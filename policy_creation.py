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

        self.no_rec_value = no_rec_value
        log.info("No recommendation special valus is: {}".format(self.no_rec_value))
        self.outcome_is_cost = outcome_is_cost
        log.info("Policy for outcome_is_cost: {}".format(self.outcome_is_cost))

    def filter_uncertainty_cate_estimations(self):
        """
        do not consider CATE estimations that have high uncertainty
        :return:
        """
        raise NotImplementedError

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
