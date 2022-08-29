import pandas as pd
import numpy as np
import os
import joblib
import logging

# A logger for this file
log = logging.getLogger(__name__)


class PolicyCreation:

    """
    Class for defining an intervention policy based on CATE estimators for a binary intervention decision
    assuming CATE(X) = E[Y|X,T=1] - E[Y|X,T=0]
    """
    # TODO: deal with no_rec_value
    # TODO: add more uncertainty steps for no recommendation 
    def __init__(self, outcome_is_cost: bool = False, no_rec_value: int = -999):
        """
        :param outcome_is_cost: True if lower outcome Y is better, then if CATE > 0 policy recommends T=0
        else if CATE < 0 policy recommends T=1. And vice versa if outcome_is_cost = False.
        :param no_rec_value: special value that indicates no recommendation due to various reasons
        """

        # self.policy_type_dict = {0: "majority", 1: "weighted", 2: "consensus", 3: "dictator", 4: "average"}
        # log.info("supported policy types are: {}".format(self.policy_type_dict))
        # self.chosen_policy_type = None
        self.no_rec_value = no_rec_value
        log.info("No recommendation special valus is: {}".format(self.no_rec_value))
        self.outcome_is_cost = outcome_is_cost
        log.info("Policy for outcome_is_cost: {}".format(self.outcome_is_cost))

    # def set_policy_type(self, policy_type_key: int):
    #     """
    #     define policy type: majority, weighted, consensus, dictator. Depends if there are multiple CATE estimators
    #     :return:
    #     """
    #
    #     try:
    #         self.chosen_policy_type = self.policy_type_dict[policy_type_key]
    #         log.info("chosen type is: {}".format(self.chosen_policy_type))
    #
    #     except LookupError:
    #         log.info("No such key in self.policy_type_dict, please choose a supported policy type:\n "
    #               "{}".format(self.policy_type_dict))

    def filter_uncertainty_cate_estimations(self):
        """
        do not consider CATE estimations that have high uncertainty
        :return:
        """
        raise NotImplementedError
    #
    # def calc_cate_df(self, y1_score: np.ndarray, y0_score: np.ndarray):
    #     """
    #     Assuming same length
    #     :param y1_score: probabilities for entire data to be classified as y1
    #     :param y0_score:  probabilities for entire data to be classified as y0
    #     :return:
    #     """
    #     # TODO: move to a CATE class and add multiple cate columns
    #     return pd.DataFrame(data=y1_score-y0_score)

    def calc_policy(self, cates_df: pd.DataFrame, cols_to_exclude: list):
        """
        calculates policy recommendations for each row
        :param cates_df: rows are patient-level, columns are cate estimator names, values are CATE estimations
        :return:
        """

        if not isinstance(cates_df, pd.DataFrame):
            cates_df = pd.DataFrame(cates_df)

        cates_df.drop(columns=cols_to_exclude, inplace=True)

        policies_df = pd.DataFrame()

        majority_policy = self.calc_majority_policy(cates_df)
        average_policy = self.calc_average_policy(cates_df)

        policies_df["average_policy"] = average_policy
        policies_df["majority_policy"] = majority_policy

        for cate_col in cates_df.columns:
            cate_policy = self.calc_average_policy(cates_df[cate_col])
            policies_df["{}_policy".format(cate_col.replace('cate_', ''))] = cate_policy

        log.info("policy distributions:")
        log.info(policies_df.describe())

        return policies_df

    def turn_cate_to_sign_policy(self, policy_cate_df: pd.DataFrame):
        # TODO: add custom threshold other than CATE sign
        """
        Turns cate values to intervention decision by 0 as a threshold according to self.outcome_is_cost flag
        :param policy_cate_df: df of one or more columns (CATE estimators) of CATE values depending on the policy call
        :return: Binary intervention policy per row
        """
        policy_cate_sign_df = policy_cate_df.transform(lambda x: np.sign(x))
        policy = np.sign(policy_cate_sign_df.sum(axis=1))
        policy.replace({0: self.no_rec_value}, inplace=True)

        if self.outcome_is_cost:
            policy.replace({-1: 1, 1: 0}, inplace=True)
        else:
            policy.replace({-1: 0, 1: 1}, inplace=True)
        return policy

    def calc_majority_policy(self, cate_df: pd.DataFrame):
        """
        Take a majority vote per row among all CATE estimators
        :param cate_df: rows are patient-level, columns are cate estimator names, values are CATE estimations
        :return:
        """

        majority_policy = self.turn_cate_to_sign_policy(cate_df)

        return majority_policy

    def calc_average_policy(self, cate_df: pd.DataFrame):
        """
        Take an average CATE value per row among all CATE estimators
        :param cate_df: rows are patient-level, columns are cate estimator names, values are CATE estimations
        :return:
        """

        if not isinstance(cate_df, pd.DataFrame):
            cate_df = pd.DataFrame(cate_df)

        cate_mean_df = cate_df.mean(axis=1).to_frame()
        average_policy = self.turn_cate_to_sign_policy(cate_mean_df)

        return average_policy

    def add_baseline_policies(self, policies_df: pd.DataFrame, curr_policy: np.ndarray):

        if curr_policy.shape[0] != policies_df.shape[0]:
            log.error("cur_policy is not in shape of policies_df")

        policies_df["current_policy"] = curr_policy
        policies_df["treat_0"] = np.zeros(policies_df.shape[0])
        policies_df["treat_1"] = np.ones(policies_df.shape[0])
        policies_df["random"] = np.random.randint(2, size=policies_df.shape[0])

        return policies_df
    
    def temp_calc_policy_by_learner(self, causal_learner_type, data, treatment_name, chosen_features_dict,
                                    chosen_outcome_model_name_dict, chosen_outcome_model_name, 
                                    top_outcome_shap_features, rlearner_obj, xlearner_obj):
        # TODO: refactor
        # # TODO: clean chosen_features_dict_shap_step from causal_learner_type
        # if causal_learner_type == "slearner":
        #     # create copy with T=0 for all
        #     trimmed_train_data_chosen_features_t0 = \
        #         data[chosen_features_dict[0]].copy(deep=True)
        #     trimmed_train_data_chosen_features_t0.loc[:, treatment_name] = 0
        #     # create copy with T=1 for all
        #     trimmed_train_data_chosen_features_t1 = \
        #         data[chosen_features_dict[1]].copy(deep=True)
        #     trimmed_train_data_chosen_features_t1.loc[:, treatment_name] = 1
        #
        #     cate_df = self.calc_cate_df(y1_score=
        #                                                    chosen_outcome_model_name_dict["all"][
        #                                                        chosen_outcome_model_name]
        #                                                    .predict_proba(trimmed_train_data_chosen_features_t1)[:, 1],
        #                                                    y0_score=
        #                                                    chosen_outcome_model_name_dict["all"][
        #                                                        chosen_outcome_model_name].
        #                                                    predict_proba(trimmed_train_data_chosen_features_t0)[:, 1])
        #     # predict y hat from S learner
        #     y0_hat = chosen_outcome_model_name_dict["all"][chosen_outcome_model_name].predict(
        #         trimmed_train_data_chosen_features_t0)
        #     y1_hat = chosen_outcome_model_name_dict["all"][chosen_outcome_model_name].predict(
        #         trimmed_train_data_chosen_features_t1)
        #
        # elif causal_learner_type == "tlearner":
        #     cate_df = self.calc_cate_df(y1_score=
        #                                                    chosen_outcome_model_name_dict[1][chosen_outcome_model_name]
        #                                                    .predict_proba(
        #                                                        data[chosen_features_dict[1]])[:, 1],
        #                                                    y0_score=
        #                                                    chosen_outcome_model_name_dict[0][chosen_outcome_model_name].
        #                                                    predict_proba(
        #                                                        data[chosen_features_dict[0]])[:, 1])
        #     # predict y hat from T learner
        #     y0_hat = chosen_outcome_model_name_dict[0][chosen_outcome_model_name].predict(
        #         data[chosen_features_dict[0]])
        #     y1_hat = chosen_outcome_model_name_dict[1][chosen_outcome_model_name].predict(
        #         data[chosen_features_dict[1]])
        #
        # elif causal_learner_type == "rlearner":
        #     # predict Rlearner CATE train
        #     cate_df = \
        #         rlearner_obj.predict_rlearner(
        #             X=data[[x for x in top_outcome_shap_features if x != treatment_name]])
        #     cate_df = pd.DataFrame(data=cate_df)
        #     # predict potential outcomes # TODO: understand how the effect() method works to predict y among cv folds models
        #     #  - probably predict from model that data point was in validation fold
        #     y0_hat = rlearner_obj.GridSearchCV_R_est.models_y[0][0].predict(
        #         data[[x for x in top_outcome_shap_features if x != treatment_name]])
        #     y1_hat = y0_hat  # TODO: understand how to extract a corrected E[Y|X] as the Rlearner uses it...
        #
        # elif causal_learner_type == "xlearner":
        #
        #     # predict Xlearner CATE train
        #     cate_df = \
        #         xlearner_obj.predict_xlearner(
        #             X=data[[x for x in top_outcome_shap_features if x != treatment_name]])
        #     cate_df = pd.DataFrame(data=cate_df)
        #     # predict potential outcomes
        #     # TODO: understand how the effect() method works to predict y among cv folds models
        #     #  - probably predict from model that data point was in validation fold
        #     y0_hat = xlearner_obj.x_est.models[0].predict(
        #         data[[x for x in top_outcome_shap_features if x != treatment_name]])
        #     y1_hat = xlearner_obj.x_est.models[1].predict(
        #         data[[x for x in top_outcome_shap_features if x != treatment_name]])
        #
        # else:
        #     raise NotImplementedError

        log.info("--------SAVE CATE DF")
        cate_df.index = data.index
        policy_learner = self.calc_policy(cate_df=cate_df)

        return cate_df, policy_learner, y0_hat, y1_hat


if __name__ == '__main__':

    cate_df = 2*np.random.random_sample((20, 3))-1

    policy_creat_obj = PolicyCreation(outcome_is_cost=False)
    policy_creat_obj.set_policy_type(0)
    maj_policy = policy_creat_obj.calc_policy(cate_df=cate_df)

    policy_creat_obj.set_policy_type(4)
    avg_policy = policy_creat_obj.calc_policy(cate_df=cate_df)
    log.info("ok")
