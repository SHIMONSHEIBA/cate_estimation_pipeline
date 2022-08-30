"""
Module containing the main class for managing the causal discovery outputs for the Causalis engine
"""
import pandas as pd
from utils.utils_ml import run_causalnex


class CausalisGraph:

    """
    Main class for managing the causal graph information out of the causal discovery outputs,
    merges causal graph outputs to a unified representation, solves controversies of two sources if exist and prepares
    format for CausalisModel class
    """

    def __init__(self, data: pd.DataFrame(), treatment: str, outcome: str, graph=None,
                 common_causes: list = None, instruments: list = None,
                 effect_modifiers: list = None, experiment_name: str = None, path: str = None, **kwargs):

        """ Create a causalis graph instance.

        :param data: a pandas dataframe containing treatment, outcome and other
        variables.
        :param treatment: name of the treatment variable
        :param outcome: name of the outcome variable
        :param graph: a string containing a DAG specification in DOT format
        :param common_causes: names of common causes of treatment and outcome
        :param instruments: names of instrumental variables for the effect of treatment on outcome
        :param effect_modifiers: names of variables that can modify the treatment effect.
        :param experiment_name: name of current run
        Estimators will return multiple different estimates based on each value of effect_modifiers.
        :returns: an instance of CausalisGraph class

        """
        self._data = data
        self._treatment = treatment
        self._outcome = outcome
        self._common_causes = common_causes
        self._instruments = instruments
        self._effect_modifiers = effect_modifiers
        self._experiment_name = experiment_name
        self._path = path
        # combine all_trimmed_high_cate variable types with no duplicates
        self._all_variables = list()
        for x in [[self._treatment], [self._outcome], self._common_causes, self._effect_modifiers, self._instruments]:
            if x is not None:
                self._all_variables = list(set(self._all_variables).union(x))

    def run_causalnex_notears(self):
        """
        run the notears algorithm of causal discovery from causalnex, coloring treatment and outcome nodes in graph
        :return:
        """
        sm = run_causalnex(discover_data=self._data.loc[:, self._all_variables],
                           file_name=self._path,
                           label="{} "
                           "\n \t Graph discovery".format(self._experiment_name),
                           color_nodes_name_dict={self._treatment: "#013220", self._outcome: "#6a0dad"})
        return sm
