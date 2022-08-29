from utils_ml import regression_importance
from utils_graphs import weighted_kaplan_meier
import os
import logging
from sklearn.preprocessing import RobustScaler

from causalis_graph import CausalisGraph

# A logger for this file
log = logging.getLogger(__name__)


class SubPopulationAnalysis:

    """
    Class of different methods to analyze CATE, Policy & sub populations
    """

    def __init__(self, data, outcome_name, treatment_name, pop_description, ftrs_for_analysis, analysis_path,
                 need_scale, scaler):
        """

        :param data:
        :param outcome_name:
        :param treatment_name:
        :param pop_description:
        :param ftrs_for_analysis:
        :param analysis_path:
        """

        self.data = data
        self.outcome_name = outcome_name
        self.treatment_name = treatment_name
        self.ftrs_for_analysis = ftrs_for_analysis
        self.pop_description = pop_description

        analysis_path = os.path.join(analysis_path, 'SubPopulationAnalysis')
        os.makedirs(analysis_path, exist_ok=True)
        self.analysis_path = analysis_path
        self.need_scale = need_scale
        self.scaler = scaler

    def pop_feature_importance(self):
        # TODO: add conversion to probabilities from log(odds ratio)
        # TODO: add from sklearn.inspection import partial_dependence, PartialDependenceDisplay

        if self.need_scale:
            if self.scaler:
                scaler = self.scaler  # assuming passed fit scaler on same features
            else:
                scaler = RobustScaler()
                scaler.fit(self.data[self.ftrs_for_analysis])

            self.data[self.ftrs_for_analysis] = scaler.transform(self.data[self.ftrs_for_analysis])

        if self.data[self.outcome_name].nunique() == 2:
            print("assuming binary outcome in population feature importance for : {}".format(self.outcome_name))
            regression_importance(data=self.data[self.ftrs_for_analysis],
                                   outcome=self.data[self.outcome_name],
                                   feature_names=self.ftrs_for_analysis,
                                   file_name="logistic_regression_{}_features_importance_pop_{}".format(
                                       self.outcome_name, self.pop_description),
                                   path=self.analysis_path,
                                  outcome_type="binary")

        elif self.data[self.outcome_name].nunique() > 10:  # TODO: change hard coded assumption
            print("assuming continuous outcome in population feature importance for : {}".format(self.outcome_name))
            regression_importance(data=self.data[self.ftrs_for_analysis],
                                   outcome=self.data[self.outcome_name],
                                   feature_names=self.ftrs_for_analysis,
                                   file_name="regression_{}_features_importance_pop_{}".format(
                                       self.outcome_name, self.pop_description),
                                   path=self.analysis_path,
                                  outcome_type='cont')

        else:
            print("outcome {} not supported".format(self.outcome_name))

        return scaler

    # def weighted_kaplan_meier(self):
    #     weighted_kaplan_meier(title="TRAIN: NSCLC Overall Survival, Women with high %PDL1, per arm",
    #                           df_with_weights=chosen_nsclc_mortality_df_high_pdl1_women_train_agree,
    #                           weight_arm_label="Causalis",
    #                           df_no_weights=chosen_nsclc_mortality_df_high_pdl1_women_train,
    #                           no_weight_arm_label="Doctors",
    #                           time_col="death_time",
    #                           event_col="death_event",
    #                           weight_col="ipw",
    #                           x_axis_label='Month from metastatic 1line',
    #                           file_path=self.analysis_path,
    #                           x_lim_tuple=(0, 29),
    #                           weighted_arm_group_col="Causalis",
    #                           no_weighted_arm_group_col="Doctors")
    #
    # def policy_decision_bar_plot_by_time(self, time_col="Metastatic_first_line_start_year"):
    #     policy_decision_bar_plot_by_time(data=self.data[self.ftrs_for_analysis],
    #                                      time_col=time_col,
    #                                      policy_col="Doctors",
    #                                      policy_col_values_order=["MONO", "COMBO"],
    #                                      title='TRAIN: Doctors treatment distribution - Women with %PDL1 > 0.49',
    #                                      file_path=self.analysis_path,
    #                                      # ylim=(0, 160)
    #                                      )
    #
    # def run_causal_discovery(self):
    #     causalis_graph_obj = CausalisGraph(data=chosen_nsclc_mortality_df_high_pdl1_women_test.loc[
    #                                             chosen_nsclc_mortality_df_high_pdl1_women_test["high_cate"] == 1, :],
    #                                        treatment=treatment_name,
    #                                        outcome=outcome_name,
    #                                        graph=None,
    #                                        common_causes=graph_ftres_temporal,
    #                                        instruments=None,
    #                                        effect_modifiers=None,
    #                                        experiment_name="Women_High_PDL1_Causalis_Recommendation_Factors",
    #                                        path=graph_path)
    #
    #     sm = causalis_graph_obj.run_causalnex_notears()
