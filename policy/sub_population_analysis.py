from utils.utils_ml import regression_importance
import os
import logging
from sklearn.preprocessing import RobustScaler

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