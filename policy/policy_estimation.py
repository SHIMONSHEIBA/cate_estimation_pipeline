import pandas as pd
import numpy as np
from collections import defaultdict
from utils.utils_graphs import weighted_kaplan_meier
from utils.utils_ml import calc_ipw
import logging


# A logger for this file
log = logging.getLogger(__name__)


class PolicyEstimation:

    """
    Class for estimating a binary intervention policy value - supports only binary outcome
    """

    def __init__(self):

        log.info("calculating policy value")

    def create_y_hat_cols(self, y0_hat, y1_hat, orig_policy, new_policy):

        # create y hat columns per orig and new policies, assumes binary treatment with 0 and 1

        y_hat_policies = pd.DataFrame({"new_policy": new_policy,
                                      "orig_policy": orig_policy,
                                       "y0_hat": y0_hat,
                                       "y1_hat": y1_hat})
        # new
        y_hat_policies["y_hat_new_policy"] = y_hat_policies["y0_hat"]
        y_hat_policies.loc[y_hat_policies["new_policy"] == 1, "y_hat_new_policy"] = \
            y_hat_policies.loc[y_hat_policies["new_policy"] == 1, "y1_hat"]
        # orig
        y_hat_policies["y_hat_orig_policy"] = y_hat_policies["y0_hat"]
        y_hat_policies.loc[y_hat_policies["orig_policy"] == 1, "y_hat_orig_policy"] = \
            y_hat_policies.loc[y_hat_policies["orig_policy"] == 1, "y1_hat"]

        return y_hat_policies

    def policy_actions_propensity(self, policy, propensity_for_positive_class):
        " take propensity of action a in policy - assuming binary treatment with 0/1 actions, "
        " and propensity_for_positive_class is probability for action 1 "
        policy_actions_propensity = propensity_for_positive_class.copy(deep=True)
        policy_actions_propensity[policy == 0] = (1 - propensity_for_positive_class.loc[policy == 0]).values
        return policy_actions_propensity

    def estimate_policy_value(self, treatment, outcome, policy, policy_actions_propensity, y_hat_policies):
        """
        estimate policy value by chosen estimation method
        :param treatment: treatment column of data to evaluate
        :param outcome: outcome column of data to evaluate
        :param policy: actions that the new policy will choose
        :param propensity: Estimated probabilities for positive class in binary treatment
        :return:
        """

        policy_value_est_dr = self.estimate_policy_value_doubly_robust(treatment.values, outcome.values, policy.values,
                                                                    policy_actions_propensity.values,
                                                                    y_hat_policies[["y_hat_new_policy",
                                                                                    "y_hat_orig_policy"]].astype(
                                                                        'float64').values)

        policy_value_est_ipw = self.estimate_policy_value_ipw(treatment.values, outcome.values, policy.values,
                                                              policy_actions_propensity.values)
        return policy_value_est_dr, policy_value_est_ipw

    def estimate_policy_value_doubly_robust(self, treatment, outcome, policy, policy_actions_propensity, y_hat_new_old):
        """
        stabilized doubly robust policy value estimation
        :return:
        """

        agreement_set = policy == treatment

        y_hat_new_mean = np.mean(y_hat_new_old[:, 0])
        weighted_residuals = np.sum((outcome[agreement_set] - y_hat_new_old[:, 1][agreement_set])
                                    / policy_actions_propensity[agreement_set].reshape(-1))
        sum_residulas_weights = 1 / (np.sum(1 / policy_actions_propensity[agreement_set]))
        policy_stabilized_dr = sum_residulas_weights * weighted_residuals + y_hat_new_mean

        policy_stabilized_dr = round(policy_stabilized_dr, 2)

        return policy_stabilized_dr

    def estimate_policy_value_ipw(self, treatment, outcome, policy, policy_actions_propensity):
        """
        stabilized IPW policy value estimation
        :return:
        """

        agreement_set = policy == treatment
        weighted_residuals = np.sum((outcome[agreement_set]) / policy_actions_propensity[agreement_set].reshape(-1))
        sum_residulas_weights = 1 / (np.sum(1 / policy_actions_propensity[agreement_set]))
        policy_stabilized_ipw = sum_residulas_weights * weighted_residuals

        policy_stabilized_ipw = round(policy_stabilized_ipw, 2)

        return policy_stabilized_ipw

    def temp_bootstrap_policy_value(self, sample_data, cfg, policies_df_list, propensity_score_name, treatment_name):
        # TODO: parallel
        sample_shape = sample_data.shape[0]

        policy_value_df = pd.DataFrame(columns=["i", "policy", "policy_value_estimator",
                                                "policy_value", "agree_set_size"])
        for i in range(1000):
            if i == 0:
                sample_data = sample_data
            else:
                sample_data = sample_data.sample(n=sample_shape, replace=True)

            for policy in policies_df_list:
                log.info("starting iteration {} for policy {}".format(i, policy))
                # get propensity by policy
                policy_actions_propensity = self.policy_actions_propensity(policy=sample_data[policy],
                                                                           propensity_for_positive_class=
                                                                           sample_data[propensity_score_name])

                # create yhat columns
                y_hat_policies = self.create_y_hat_cols(y0_hat=sample_data["y0_hat"].values,
                                                        y1_hat=sample_data["y1_hat"].values,
                                                        orig_policy=sample_data[treatment_name].values.reshape(-1),
                                                        new_policy=sample_data[policy].values)
                # calculate policy value
                policy_value_est_dr, policy_value_est_ipw = self.estimate_policy_value(treatment=sample_data[treatment_name],
                                                              outcome=sample_data[cfg.params.outcome_name],
                                                              policy=sample_data[policy],
                                                              policy_actions_propensity=policy_actions_propensity,
                                                              y_hat_policies=y_hat_policies)

                log.info("finished iteration {} for policy {}: SDR policy value: {} and SIPW policy value: {}".format(
                    i, policy, policy_value_est_dr, policy_value_est_ipw))

                # INSERT SDR I
                i_sdr_policy_value_df = pd.DataFrame(data=[[i,
                                                            policy,
                                                            "SDR",
                                                            policy_value_est_dr,
                                                            round(sum(sample_data[policy].values == sample_data[treatment_name].values) /
                                                                  sample_data.shape[0], 2)]],
                                                     columns=["i", "policy", "policy_value_estimator", "policy_value",
                                                              "agree_set_size"])
                policy_value_df = policy_value_df.append(i_sdr_policy_value_df, ignore_index=True)
                # INSERT SIPW I
                i_sipw_policy_value_df = pd.DataFrame(data=[[i,
                                                            policy,
                                                            "SIPW",
                                                            policy_value_est_ipw,
                                                            round(sum(sample_data[policy].values
                                                                      == sample_data[treatment_name].values) /
                                                                  sample_data.shape[0], 2)]],
                                                      columns=["i", "policy", "policy_value_estimator", "policy_value",
                                                               "agree_set_size"])
                policy_value_df = policy_value_df.append(i_sipw_policy_value_df, ignore_index=True)

            i_y_mean_df = pd.DataFrame(data=[[i,
                                              "current_policy",
                                              "average_y",
                                              sample_data[cfg.params.outcome_name].mean().round(2),
                                              round(sum(sample_data[policy].values == sample_data[treatment_name].values) /
                                                              sample_data.shape[0], 2)]],
                                                 columns=["i", "policy", "policy_value_estimator", "policy_value",
                                                          "agree_set_size"])
            policy_value_df = policy_value_df.append(i_y_mean_df, ignore_index=True)

        return policy_value_df

    def ipw_km(self, data, propensity_score_name, treatment_name, policy_names, policy_path, title, sipw: bool = True):

        # TODO: apply CATE certainty threshold, generalize method, correct stabilized version

        # create time column for Kaplan Meier with censoring as last seen in data
        # assuming overall_survival_MONTHS is null also for deceased after data cutoff minus buffer
        data["death_time"] = data["overall_survival_MONTHS"]
        data.loc[data["death_time"].isnull(), "death_time"] = data.loc[
            data["death_time"].isnull(), "data_model_update_date_to_1line_MONTHS"]

        # define right censoring
        data["death_event"] = 1
        data.loc[data["overall_survival_MONTHS"].isnull(), "death_event"] = 0

        # set IPW weights
        data["ipw"] = calc_ipw(p_treatment1=data[propensity_score_name].values,
                               treatment=data[treatment_name].values,
                               treatment1_name=1,
                               treatment0_name=0,
                               stabilized=True)

        dfs_to_plot_dict = defaultdict(dict)

        for policy_name in policy_names:
            dfs_to_plot_dict[policy_name] = {"df": data.query("{} == {}".format(treatment_name, policy_name)),
                                             "df_group_col": None,
                                             "df_weight_col": "ipw"}

        weighted_kaplan_meier(title=title,
                              dfs_to_plot_dict=dfs_to_plot_dict,
                              time_col="death_time",
                              event_col="death_event",
                              x_axis_label='Month from metastatic 1line',
                              file_path=policy_path,
                              x_lim_tuple=(0, 24), stabilized=sipw)
