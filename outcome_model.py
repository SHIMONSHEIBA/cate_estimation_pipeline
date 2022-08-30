import shap
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from utils_ml import EconmlRlearner, EconMlXlearner, nested_k_fold, binary_clf_eval
from utils_graphs import plot_calibration_curve, live_scatter, shap_feature_importance
import logging

# A logger for this file
log = logging.getLogger(__name__)


def temp_outcome_modeling(causal_learner_type, train_data, test_data, all_chosen_features,
                          treatment_name, outcome_name, greater_is_better, score, outcome_path, chosen_features_dict,
                          treatment_values_list, binary_classifier_models_dict, outer_fold_num, inner_fold_num,
                          score_name, upsample, interactive_env):

    rlearner_obj = None
    xlearner_obj = None
    chosen_outcome_model_name = None
    chosen_outcome_model_name_dict = None
    
    for causal_learner in causal_learner_type:
    
        # TODO: change if else patch to be consistent in outcome modeling for different learners
        # TODO: surface hyper params, switch to DoWhy class now that they support train / test fit() / predict()
        # TODO: pass binary_classifiers_features_dict to R/X learners
        # TODO: add r/x learners evaluation
        if causal_learner == "rlearner":
            # EconML Rlearner
            rlearner_obj = EconmlRlearner()
            rlearner_obj.fit_rlearner(X=train_data[[x for x in all_chosen_features if x != treatment_name]],
                                      T=train_data[treatment_name],
                                      Y=train_data[outcome_name],
                                      score=score,
                                      greater_is_better=greater_is_better,
                                      outer_fold_num=outer_fold_num)
    
            joblib.dump(rlearner_obj, os.path.join(outcome_path, "rlearner_obj.pkl"))
    
        elif causal_learner == "xlearner":

            # EconML Xlearner
            xlearner_obj = EconMlXlearner()
            xlearner_obj.fit_xlearner(X=train_data[[x for x in all_chosen_features if x != treatment_name]],
                                      T=train_data[treatment_name],
                                      Y=train_data[outcome_name],
                                      score=score,
                                      greater_is_better=greater_is_better,
                                      outer_fold_num=outer_fold_num)
    
            joblib.dump(xlearner_obj, os.path.join(outcome_path, "xlearner_obj.pkl"))
    
        elif causal_learner in ["slearner", "tlearner"]:

        # TODO: debug for Slearner with first tlearner step
            # prepare training data according to chosen causal learner
            outcome_data_causal_learner_dict = defaultdict(dict)
            if causal_learner == "slearner":
                outcome_data_causal_learner_dict["all"]["X"] = \
                    train_data[list(set(all_chosen_features).union([treatment_name]))]
                outcome_data_causal_learner_dict["all"]["Y"] = train_data[outcome_name]
            elif causal_learner == "tlearner":
                for t_arm in treatment_values_list:
                    outcome_data_causal_learner_dict[t_arm]["X"] = train_data.loc[train_data[treatment_name]
                                                                                          == t_arm,
                                                                                          chosen_features_dict[t_arm]]
                    outcome_data_causal_learner_dict[t_arm]["Y"] = train_data.loc[train_data[treatment_name]
                                                                                          == t_arm,
                                                                                          outcome_name]
    
            # dict of chosen outcome models
            chosen_outcome_model_name_dict = defaultdict(dict)
            # -- choose ML outcome model
            log.info("training outcome models for causal learner: {}".format(causal_learner))
            outcome_models_dict = {}
            outcome_models_cv_results_df = pd.DataFrame(columns=["params", "score_name", "score", "model_name", "model",
                                                                 "dataset"])
            model_performance_df_list = list()
    
            for arm in outcome_data_causal_learner_dict.keys():
                if outcome_name == 'AE_neutropenia_1line_timeline' and arm == 1.0:
                    continue
                log.info("modeling outcome {} for causal learner {} arm {}".format(outcome_name, causal_learner, arm))
                # iterate on all models
                for model_name, model_dict in binary_classifier_models_dict.items():
                    log.info("Working on outcome model {}".format(model_name))
                    curr_outcome_model, model_cv_results_df, model_outer_cv_performance_df = \
                        nested_k_fold(X=outcome_data_causal_learner_dict[arm]["X"],
                                      y=outcome_data_causal_learner_dict[arm]["Y"],
                                      model=model_dict["model"],
                                      model_name=model_name,
                                      space=model_dict["space"],
                                      outer_folds_num=outer_fold_num,
                                      inner_folds_num=inner_fold_num,
                                      score=score,
                                      score_name=score_name,
                                      greater_is_better=greater_is_better,
                                      upsample=upsample,
                                      nested_step=True,
                                      fit_params=model_dict["fit_params"] if "fit_params" in model_dict.keys() else None)
    
                    outcome_models_dict[model_name] = curr_outcome_model
                    joblib.dump(curr_outcome_model, os.path.join(outcome_path,
                                                                 "T_{}_Outcome_{}.pkl".format(arm, model_name)))
                    outcome_models_cv_results_df = outcome_models_cv_results_df.append(model_cv_results_df,
                                                                                       ignore_index=True)
    
                    # todo: fix cv performance calc during CV when nested step
                    # model_outer_cv_performance_df = model_outer_cv_performance_df.to_frame().transpose()
                    model_outer_cv_performance_df["model"] = "T_{}_Outcome_{}.pkl".format(arm, model_name)
                    model_outer_cv_performance_df["dataset"] = "nested_cv_outer_loop"
    
                    # train
                    eval_dict_train = binary_clf_eval(model=curr_outcome_model,
                                                      data=train_data,
                                                      chosen_ftrs=chosen_features_dict[arm],
                                                      outcome_name=outcome_name,
                                                      model_name="T_{}_Outcome_{}".format(arm, model_name),
                                                      dataset="train",
                                                      path=outcome_path)
    
                    model_train_performance_df = pd.DataFrame(eval_dict_train, index=[0])
                    model_train_performance_df["model"] = "T_{}_Outcome_{}".format(arm, model_name)
                    model_train_performance_df["dataset"] = "all_train"
                    # test
                    eval_dict_test = binary_clf_eval(model=curr_outcome_model,
                                                     data=test_data,
                                                     chosen_ftrs=chosen_features_dict[arm],
                                                     outcome_name=outcome_name,
                                                     model_name="T_{}_Outcome_{}".format(arm, model_name),
                                                     dataset="trimmed_test",
                                                     path=outcome_path)
    
                    model_test_performance_df = pd.DataFrame(eval_dict_test, index=[0])
                    model_test_performance_df["model"] = "T_{}_Outcome_{}".format(arm, model_name)
                    model_test_performance_df["dataset"] = "trimmed_test"
    
                    model_performance_df_list.append(pd.concat([model_train_performance_df, model_outer_cv_performance_df,
                                                                model_test_performance_df]))
    
                # save all models performance in order to pick the most stable one
                outcome_models_cv_results_df.to_csv(os.path.join(outcome_path,
                                                                 "T_{}_outcome_models_cv_results_df.csv".format(arm)))
                # calibration train
                plot_calibration_curve(clf_dict=outcome_models_dict,
                                       X=train_data[chosen_features_dict[arm]],
                                       y=train_data[outcome_name],
                                       title="T {} Outcome models {} - Train".format(arm, outcome_name),
                                       path=outcome_path)
    
                # calibration test
                plot_calibration_curve(clf_dict=outcome_models_dict,
                                       X=test_data[chosen_features_dict[arm]],
                                       y=test_data[outcome_name],
                                       title="T {} Outcome models {} - Test".format(arm, outcome_name),
                                       path=outcome_path)
    
                # show auc in best hyper parameter space
                if interactive_env:
                    live_scatter(df=outcome_models_cv_results_df, x="model_name", y="score",
                                 name="Outcome model: {}".format(outcome_name),
                                 color="model_name", size="score",
                                 hover_data=["model_name", "score", "params"], cols_to_string=["params"])

                # TODO: define dynamically
                # choose best model
                chosen_outcome_model_name = "lgbm"
                chosen_outcome_model_name_dict[arm][chosen_outcome_model_name] = \
                    outcome_models_dict[chosen_outcome_model_name]

                # add "neutral" models for y_hat in policy value
                chosen_outcome_model_name_dict["y_hat_policy_{}".format(arm)]["GradientBoostingClassifier"] = \
                    outcome_models_dict["GradientBoostingClassifier"]
    
            pd.concat(model_performance_df_list).to_csv(os.path.join(outcome_path, "outcome_models_performance_df.csv"))
    
            # explain chosen models
            for arm in chosen_outcome_model_name_dict.keys():
                if 'y_hat_policy_' in str(arm):  # TODO: CLEAN HACK
                    continue
                arm_chosen_outcome_model_name = list(chosen_outcome_model_name_dict[arm].keys())[0]
                arm_chosen_outcome_model = chosen_outcome_model_name_dict[arm][arm_chosen_outcome_model_name]
                # SHAP
                # train
                explainer_outcome_train = shap.Explainer(arm_chosen_outcome_model,
                                                         train_data[chosen_features_dict[arm]])
                shap_values_train_outcome = \
                    explainer_outcome_train(train_data[chosen_features_dict[arm]],
                                            check_additivity=False)  # TODO: debug check_additivity
                # test
                explainer_outcome_test = shap.Explainer(arm_chosen_outcome_model,
                                                        test_data[chosen_features_dict[arm]])
                shap_values_test_outcome = \
                    explainer_outcome_test(test_data[chosen_features_dict[arm]],
                                           check_additivity=False)  # TODO: debug check_additivity
    
                # Global level
                shap_feature_importance(shap_values=shap_values_train_outcome,
                                        title="T {} Train outcome {} ".format(arm, arm_chosen_outcome_model_name),
                                        order_max=False, path=outcome_path)
    
                shap_feature_importance(shap_values=shap_values_test_outcome,
                                        title="T {} Test outcome {} ".format(arm, arm_chosen_outcome_model_name),
                                        order_max=False, path=outcome_path)

                # Local level
                data_point_id_iloc_to_explain = 5
                shap.plots.force(shap_values_train_outcome[data_point_id_iloc_to_explain],
                                 show=False,
                                 matplotlib=True).savefig(
                    os.path.join(outcome_path, 'T_{}_train_outcome_{}_iloc_{}.png'.format(arm, arm_chosen_outcome_model_name,
                                                                                          data_point_id_iloc_to_explain)),
                    format="png", dpi=150, bbox_inches='tight')
                plt.close()
    
        else:
            raise NotImplementedError

    return chosen_outcome_model_name, chosen_outcome_model_name_dict, rlearner_obj, xlearner_obj
