import pandas as pd
import logging
import numpy as np
from collections import defaultdict
from functools import reduce
import matplotlib.pyplot as plt
import os
# A logger for this file
log = logging.getLogger(__name__)


class CateEvaluation:

    """
    Class for evaluations of CATE estimators and their ML models
    """

    def __init__(self):
        """
        """

    def calc_cate_df(self, y1_score: np.ndarray, y0_score: np.ndarray):
        """
        Assuming same length
        :param y1_score: probabilities for entire data to be classified as y1
        :param y0_score:  probabilities for entire data to be classified as y0
        :return:
        """
        return pd.DataFrame(data=y1_score-y0_score)

    def temp_calc_cate_by_learner(self, causal_learner_type, data, treatment_name, chosen_features_dict,
                                  chosen_outcome_model_name_dict, chosen_outcome_model_name, top_outcome_shap_features,
                                  rlearner_obj, xlearner_obj):

        cate_df_dict = defaultdict(dict)
        for learner in causal_learner_type:
            # TODO: refactor
            if learner == "slearner":
                # create copy with T=0 for all
                trimmed_train_data_chosen_features_t0 = \
                    data[chosen_features_dict[0]].copy(deep=True)
                trimmed_train_data_chosen_features_t0.loc[:, treatment_name] = 0
                # create copy with T=1 for all
                trimmed_train_data_chosen_features_t1 = \
                    data[chosen_features_dict[1]].copy(deep=True)
                trimmed_train_data_chosen_features_t1.loc[:, treatment_name] = 1

                cate_df = self.calc_cate_df(y1_score=chosen_outcome_model_name_dict["all"][
                                                         chosen_outcome_model_name].predict_proba(
                    trimmed_train_data_chosen_features_t1)[:, 1],
                                            y0_score=chosen_outcome_model_name_dict["all"][
                                                         chosen_outcome_model_name].predict_proba(
                                                trimmed_train_data_chosen_features_t0)[:, 1])
                # predict y hat from S learner
                y0_hat = chosen_outcome_model_name_dict["all"][chosen_outcome_model_name].predict_proba(
                    trimmed_train_data_chosen_features_t0)[:, 1]
                y1_hat = chosen_outcome_model_name_dict["all"][chosen_outcome_model_name].predict_proba(
                    trimmed_train_data_chosen_features_t1)[:, 1]

            elif learner == "tlearner":
                cate_df = self.calc_cate_df(y1_score=chosen_outcome_model_name_dict[1][
                                                         chosen_outcome_model_name].predict_proba(
                    data[chosen_features_dict[1]])[:, 1],
                                            y0_score=chosen_outcome_model_name_dict[0][
                                                         chosen_outcome_model_name].predict_proba(
                                                data[chosen_features_dict[0]])[:, 1])
                # predict y hat from T learner
                y0_hat = chosen_outcome_model_name_dict[0][chosen_outcome_model_name].predict_proba(
                    data[chosen_features_dict[0]])[:, 1]
                y1_hat = chosen_outcome_model_name_dict[1][chosen_outcome_model_name].predict_proba(
                    data[chosen_features_dict[1]])[:, 1]

            elif learner == "rlearner":
                # predict Rlearner CATE train
                cate_df = \
                    rlearner_obj.predict_rlearner(
                        X=data[[x for x in top_outcome_shap_features if x != treatment_name]])
                cate_df = pd.DataFrame(data=cate_df)
                # TODO: understand how the effect() method works to predict y among cv folds models
                # predict potential outcomes
                #  - probably predict from model that data point was in validation fold
                y0_hat = rlearner_obj.GridSearchCV_R_est.models_y[0][0].predict_proba(
                    data[[x for x in top_outcome_shap_features if x != treatment_name]])[:, 1]
                y1_hat = y0_hat  # TODO: understand how to extract a corrected E[Y|X] as the Rlearner uses it...

            elif learner == "xlearner":

                # predict Xlearner CATE train
                cate_df = \
                    xlearner_obj.predict_xlearner(
                        X=data[[x for x in top_outcome_shap_features if x != treatment_name]])
                cate_df = pd.DataFrame(data=cate_df)
                # predict potential outcomes
                # TODO: understand how the effect() method works to predict y among cv folds models
                #  - probably predict from model that data point was in validation fold
                y0_hat = xlearner_obj.x_est.models[0].predict_proba(
                    data[[x for x in top_outcome_shap_features if x != treatment_name]])[:, 1]
                y1_hat = xlearner_obj.x_est.models[1].predict_proba(
                    data[[x for x in top_outcome_shap_features if x != treatment_name]])[:, 1]

            else:
                raise NotImplementedError

            cate_df.rename({0: "cate_{}".format(learner)}, axis=1, inplace=True)
            # add ids
            cate_df.index = data.index
            cate_df_dict[learner]["cate_df"] = cate_df
            cate_df_dict[learner]["y0_hat"] = y0_hat
            cate_df_dict[learner]["y1_hat"] = y1_hat

        return cate_df_dict

    # def ml_performance(self):
    #     """
    #     Prediction performance measurements
    #     :return:
    #     """
    #     return NotImplementedError
    #
    # def ml_feature_importance(self):
    #     """
    #     :return:
    #     """
    #     return NotImplementedError
    #
    # def ml_error_distribution(self):
    #     """
    #     plot histogram of model error distribution
    #     :return:
    #     """
    #     return NotImplementedError
    #
    # def ml_model_logic_analysis(self):
    #     """
    #     checks how different feature values affect the model's prediction, to validate with domain experts
    #     :return:
    #     """
    #     return NotImplementedError

    def cate_correlations(self, cates_df: pd.DataFrame, dataset):
        """
        checks different correlation measurements between different CATE estimators: Pearson, Spearman, ICC
        :return:
        """

        log.info("calculating correlations between CATE methods: {}".format(dataset))
        log.info("pearson:")
        log.info(cates_df.corr(method='pearson'))
        log.info("spearman:")
        log.info(cates_df.corr(method='spearman'))
        # TODO: add ICC

        return

    def cate_ate_validation(self, cates_df: pd.DataFrame, dataset: str):
        """
        return ate and se to see that match known figures in literature
        :return:
        """

        log.info("calculating ATEs based on CATEs {}:".format(dataset))
        log.info("ATE (mean(CATE)):")
        log.info(cates_df.mean())
        log.info("Standard error of ATE (mean(CATE)):")
        log.info(cates_df.sem())
        return

    def get_cate_df(self, cate_df_dict):

        # merge all CATEs
        cate_df_list = list()
        for learner in cate_df_dict.keys():
            cate_df_list.append(cate_df_dict[learner]["cate_df"])
        cates_df = reduce(lambda df1, df2: pd.merge(df1, df2, left_index=True, right_index=True), cate_df_list)

        return cates_df

    def get_avg_cate(self, cates_df):

        # add average CATE
        log.info("adding average CATE of {} : cate_avg_all".format(cates_df.columns))
        cates_df["cate_avg_all"] = cates_df.mean(axis=1)
        return cates_df

    def plot_cates(self, path, cates_df, title):
        x_rng = np.arange(-1.0, 1.1, 0.1)
        ax = cates_df.plot.hist(bins=len(x_rng), xticks=x_rng, alpha=0.5)

        plt.title(title)

        min_ylim, max_ylim = plt.ylim()
        i = 0
        for cate_col in cates_df.columns:
            cate_mean = round(cates_df[cate_col].mean(), 2)
            plt.axvline(cate_mean, color='k', linestyle='dashed', linewidth=1)
            plt.text(cate_mean * 1.1, max_ylim * (0.9 - i), '{} Mean {}'.format(cate_col.strip('cate_'), cate_mean))
            i += 0.1

        fig = ax.get_figure()
        fig.savefig(os.path.join(path, "{}.png".format(title)))
        plt.close()

    def cate_cdf_abs_effect(self):
        """
        see CDF of absolute treatment effect to estimate potential value
        :return:
        """
        return NotImplementedError

    def cate_calibration_plots(self, cates_df: pd.DataFrame, n_folds=5):
        """
        compare calibration of different CATE methods like AIPW with others to see agreement on CATE sub populations,
        compared to a random sub population as well.
        :return:
        """

        # per CATE column
        # order rows
        # divide to n folds + take a random subset with same fold size
        # per fold calc ATE with TMLE
        # calc CI
        # plot

        return NotImplementedError

    def cate_regress_features_on_doubly_robust(self):
        """
        Regression on doubly robust score to see coefficients for X features of interest -
        in expectation the DR score is true CATE, so the regression coefficients are estimation of causal explanations
        :return:
        """
        return NotImplementedError

    def cate_define_sub_populations(self):
        """
        Characterize in terms of X features the different CATE sub populations
        :return:
        """
        return NotImplementedError

    def cate_define_agreement_set(self, cates_df):
        """
        returns index of rows where all methods agree on recommendation
        :param cates_df:
        :return:
        """

        cates_num = len(cates_df.columns)
        cates_sign_df = cates_df.transform(lambda x: np.sign(x))
        cates_vote = cates_sign_df.sum(axis=1)
        cates_vote = cates_vote.transform(lambda x: np.abs(x))
        all_cates_agreement_set_ids = cates_vote.loc[cates_vote == cates_num].index
        log.info("# of patients where all cate agree on {}".format(len(all_cates_agreement_set_ids)))
        return all_cates_agreement_set_ids

    def cate_sub_populations_agreement_sets(self):
        """
        Characterize in terms of X features the agreement/disagreement populations with current care and between CATEs
        :return:
        """


