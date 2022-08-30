import pandas as pd
from utils_ml import evaluate_clf, nested_k_fold, binary_clf_eval
from utils_graphs import plot_calibration_curve, plot_ecdf, shap_feature_importance, live_scatter
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shap
import joblib
import logging


# A logger for this file
log = logging.getLogger(__name__)


class PropensityModel:
    # TODO: add a shared propensity model (outcome adaptive lasso) for all outcomes between experiments

    """
    Propensity class to model probability of getting treatment
    """

    def __init__(self):

        self.propensity_model = None
        self.propensity_model_name = None

    def set_propensity_model(self, clf_type, clf_name: str):
        """
        sets which ml model the propensity score is based on
        :param clf_type:
        :param clf_name: distinct model name
        :return:
        """
        self.propensity_model = clf_type
        self.propensity_model_name = clf_name

    def fit(self, X, y, sample_weight=None):

        log.info("Start fitting propensity model {}".format(self.propensity_model_name))
        self.propensity_model = self.propensity_model.fit(X, y, sample_weight=sample_weight)
        log.info("Finish fitting propensity model {}".format(self.propensity_model_name))

    def get_predict(self, X):
        return self.propensity_model.predict(X)

    def get_predict_proba(self, X):
        return self.propensity_model.predict_proba(X)[:, 1]

    def plot_propensity_calibration(self, X, y, title):
        plot_calibration_curve(clf_dict={"propensity_check": self.propensity_model}, X=X, y=y, title=title)

    def evaluate_propensity(self, y, y_hat, y_score):
        eval_dict = evaluate_clf(y=y, y_hat=y_hat, y_score=y_score)
        return eval_dict

    def plot_overlap(self, data, y_score_col_name: str, t_col_name: str, cdf=False, cdf_stat=None):
        """
        plots overlap of propensity score between treatment arms
        :param data: holds y_score_col_name and t_col_name
        :param y_score_col_name: predict proba for positive treatment column name
        :param t_col_name: treatment col name
        :param cdf: if to plot overlap in histogram or CDF
        :param cdf_stat: if CDF is True, "proportion" for cumulative probability, or "count" for cumulative count
        :return:
        """

        plt.close()

        if cdf:
            plot_ecdf(data=data, col=y_score_col_name, hue=t_col_name, stat=cdf_stat)
        else:
            sns.histplot(data=data, x=y_score_col_name, hue=t_col_name)
        plt.show()

    def count_t_arms_per_bin(self, lower: float, upper: float):
        """
        returns number of data points per value in t_col_name for bin range <lower, upper>
        :param lower: lower propensity score bin range
        :param upper: upper propensity score bin range
        :return:
        """
        return NotImplementedError

    # temp func to clean main - TODO break to class methods
    def temp_fit_prop(self, train_data, treatment_name, modeling_path, all_chosen_features,
                      binary_classifier_models_dict, outer_fold_num,
                      inner_fold_num, score, score_name, greater_is_better, interactive_env):

        propensity_models_dict = {}

        # run modeling pipeline
        model_performance_df_list = list()
        propensity_models_cv_results_df = pd.DataFrame(columns=["params", "score_name", "score", "model_name", "model"])
        for model_name, model_dict in binary_classifier_models_dict.items():
            log.info("Working on propensity model {}".format(model_name))
            curr_propensity_model, model_cv_results_df, model_outer_cv_performance_df = \
                nested_k_fold(X=train_data[[x for x in all_chosen_features if x != treatment_name]],
                              y=train_data[treatment_name],
                              model=model_dict["model"],
                              model_name=model_name,
                              space=model_dict["space"],
                              outer_folds_num=outer_fold_num,
                              inner_folds_num=inner_fold_num,
                              score=score,
                              score_name=score_name,
                              greater_is_better=greater_is_better,
                              nested_step=True,
                              fit_params=model_dict["fit_params"] if "fit_params" in model_dict.keys() else None)

            propensity_models_dict[model_name] = curr_propensity_model
            joblib.dump(curr_propensity_model, os.path.join(modeling_path, "Propensity_{}.pkl".format(model_name)))
            propensity_models_cv_results_df = propensity_models_cv_results_df.append(model_cv_results_df,
                                                                                     ignore_index=True)
            # todo: fix cv performance calc during CV when nested step
            # model_outer_cv_performance_df = model_outer_cv_performance_df.to_frame().transpose()
            model_outer_cv_performance_df["model"] = "Propensity_{}".format(model_name)
            model_outer_cv_performance_df["dataset"] = "nested_cv_outer_loop"
            # evaluate propensity
            # train
            log.info("Propensity performance on all Train:")
            eval_dict = binary_clf_eval(model=curr_propensity_model,
                                        data=train_data,
                                        chosen_ftrs=[x for x in all_chosen_features if x != treatment_name],
                                        outcome_name=treatment_name,
                                        model_name="Propensity_{}".format(model_name),
                                        dataset="train",
                                        path=modeling_path)
            model_train_performance_df = pd.DataFrame(eval_dict, index=[0])
            model_train_performance_df["model"] = "Propensity_{}".format(model_name)
            model_train_performance_df["dataset"] = "all_train"

            model_performance_df_list.append(pd.concat([model_train_performance_df, model_outer_cv_performance_df]))

        pd.concat(model_performance_df_list).to_csv(
            os.path.join(modeling_path, "propensity_models_model_performance_df.csv"))

        # save all models performance in order to pick the most stable model type
        propensity_models_cv_results_df.to_csv(os.path.join(modeling_path, "propensity_models_cv_results_df.csv"))

        # calibration curves for all models
        # TODO: add plot precision-recall curve
        plot_calibration_curve(clf_dict=propensity_models_dict,
                               X=train_data[[x for x in all_chosen_features if x != treatment_name]],
                               y=train_data[treatment_name],
                               title="Propensity models {} - Train".format(treatment_name),
                               path=modeling_path)

        if interactive_env:
            # show score metric in best hyper parameter space - interactive html graph
            live_scatter(df=propensity_models_cv_results_df,
                         x="model_name",
                         y="score",
                         name="Propensity model: {}".format(treatment_name),
                         color="model_name",
                         size="score",
                         hover_data=["model_name", "score", "params"],
                         cols_to_string=["params"])

        # choose propensity model # TODO: define dynamically
        chosen_propensity_model_name = "lgbm"
        log.info("Chosen propensity model: {}".format(chosen_propensity_model_name))
        propensity_model = propensity_models_dict[chosen_propensity_model_name]

        # SHAP
        explainer_propensity_train = shap.Explainer(propensity_model, train_data[[x for x in all_chosen_features
                                                                                  if x != treatment_name]])
        shap_values_train_propensity = explainer_propensity_train(train_data[[x for x in all_chosen_features
                                                                              if x != treatment_name]],
                                                                  check_additivity=False)
        # Global level
        shap_feature_importance(shap_values=shap_values_train_propensity,
                                title="Train propensity {} ".format(chosen_propensity_model_name),
                                order_max=False, path=modeling_path)

        # Local level - chose data point iloc to explain
        data_point_id_iloc_to_explain = 12
        shap.plots.force(shap_values_train_propensity[data_point_id_iloc_to_explain],
                         show=False,
                         matplotlib=True).savefig(os.path.join(modeling_path,
                                                               'train_propensity_{}_iloc_{}.png'.format(
                                                                   chosen_propensity_model_name,
                                                                   data_point_id_iloc_to_explain)),
                                                  format="png", dpi=150, bbox_inches='tight')
        plt.close()
        return propensity_model

