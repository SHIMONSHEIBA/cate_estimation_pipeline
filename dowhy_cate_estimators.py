""" A dump of DoWhy CATE examples"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from dowhy import CausalModel
from sklearn.linear_model import LogisticRegressionCV
from econml.inference import BootstrapInference
import numpy as np
import pandas as pd
from utils import generate_controls_outcome, generate_data


class CausalLearner:
    """
    Base class to define shared parameters of all_trimmed_high_cate CATE learners
    """

    def __init__(self, method_name: str, test_significance: bool = True,
                 evaluate_effect_strength: bool = True, confidence_intervals: bool = True):
        """

        :param method_name: chosen method, see DoWhy CausalModel.estimate_effect() documentation for supported methods
        :param test_significance: Binary flag on whether to additionally do a statistical signficance test for the
        estimate.
        :param evaluate_effect_strength: (Experimental) Binary flag on whether to estimate the relative strength of the
        treatment's effect. This measure can be used to compare different treatments for the same outcome
        (by running this method with different treatments sequentially).
        :param confidence_intervals: (Experimental) Binary flag indicating whether confidence intervals should be
        computed.
        """

        self.method_name = method_name
        self.method_params = dict()
        self.method_params["init_params"] = dict()
        self.method_params["fit_params"] = dict()
        self.test_significance = test_significance
        self.evaluate_effect_strength = evaluate_effect_strength
        self.confidence_intervals = confidence_intervals

        return

    def set_method_fit_params_dict(self, inference=None):

        """
        Define inference method for learner. see SoWhy documentation for inference types such as bootstrap
        :param inference:
        :return:
        """

        self.method_params["fit_params"]["inference"] = inference

    def get_all_attributes(self):
        """
        :return: class object dict with instance attributes
        """
        all_attributes_dict = self.__dict__
        return all_attributes_dict


class SCausalLearner(CausalLearner):
    """
    Slearner CATE learner class
    """

    def __init__(self, test_significance: bool = False, evaluate_effect_strength: bool = False,
                 confidence_intervals: bool = False):
        super().__init__(method_name="backdoor.econml.metalearners.SLearner", test_significance=test_significance,
                         evaluate_effect_strength=evaluate_effect_strength, confidence_intervals=confidence_intervals)

    def set_method_init_params_dict(self, overall_model):

        self.method_params["init_params"]["overall_model"] = overall_model


class TCausalLearner(CausalLearner):
    """
    Tlearner CATE learner class
    """

    def __init__(self, test_significance: bool = False, evaluate_effect_strength: bool = False,
                 confidence_intervals: bool = False):
        super().__init__(method_name="backdoor.econml.metalearners.TLearner", test_significance=test_significance,
                         evaluate_effect_strength=evaluate_effect_strength, confidence_intervals=confidence_intervals)

    def set_method_init_params_dict(self, models):

        self.method_params["init_params"]["models"] = models


class XCausalLearner(CausalLearner):
    """
    Xlearner CATE learner class
    """

    def __init__(self, test_significance: bool = False, evaluate_effect_strength: bool = False,
                 confidence_intervals: bool = False):
        super().__init__(method_name="backdoor.econml.metalearners.XLearner", test_significance=test_significance,
                         evaluate_effect_strength=evaluate_effect_strength, confidence_intervals=confidence_intervals)

    def set_method_init_params_dict(self, models, propensity_model, cate_models):

        self.method_params["init_params"]["models"] = models
        self.method_params["init_params"]["propensity_model"] = propensity_model
        self.method_params["init_params"]["cate_models"] = cate_models

    def estimate_weight_func(self):
        """
        Taken from the Xlearner paper: https://arxiv.org/pdf/1706.03461.pdf
         "For some estimators, it might even be possible to estimate the covariance matrix of τˆ1 and τˆ0. One may then
          wish to choose g to minimize the variance of τˆ."
        :return:
        """
        return NotImplementedError


class RCausalLearner(CausalLearner):
    """
    Rlearner CATE learner class
    """

    def __init__(self, test_significance: bool = False, evaluate_effect_strength: bool = False,
                 confidence_intervals: bool = False):
        super().__init__(method_name="backdoor.econml.metalearners.RLearner", test_significance=test_significance,
                         evaluate_effect_strength=evaluate_effect_strength, confidence_intervals=confidence_intervals)

    def set_method_init_params_dict(self, model_y, model_t, model_final):

        self.method_params["init_params"]["model_y"] = model_y
        self.method_params["init_params"]["model_t"] = model_t
        self.method_params["init_params"]["model_final"] = model_final

    def set_method_fit_params_dict(self, inference=None, sample_weight=None, groups=None):

        self.method_params["fit_params"]["inference"] = inference
        self.method_params["fit_params"]["sample_weight"] = sample_weight
        self.method_params["fit_params"]["groups"] = groups


def run_dowhy_examples():

    # # Load some sample data
    # n = 10000
    # data = datasets.linear_dataset(
    #     beta=10,
    #     num_common_causes=5,
    #     num_instruments=2,
    #     num_samples=n,
    #     treatment_is_binary=True)

    d = 5
    n = 1000
    n_test = 250
    controls_outcome = generate_controls_outcome(d)
    X_test = np.random.multivariate_normal(np.zeros(d), np.diag(np.ones(d)), n_test)
    delta = 6/n_test
    X_test[:, 1] = np.arange(-3, 3, delta)
    treatment_effect = lambda x: (1 if x[1] > 0.1 else 0)*8  # x[1] is an instrument
    propensity = lambda x: (0.8 if (x[2]>-0.5 and x[2]<0.5) else 0.2)  # x[2] is an instrument

    Y, T, X = generate_data(n, d, controls_outcome, treatment_effect, propensity)
    data_df = pd.DataFrame()
    data_df["Y"] = Y
    data_df["T"] = T
    common_causes_names_list = ["X1", "X2", "X3", "X4", "X5"]
    data_df[common_causes_names_list] = X

    # I. Create a causal model from the data and given graph.
    model = CausalModel(
        data=data_df,
        treatment="T",
        outcome="Y",
        common_causes=common_causes_names_list)

    # II. Identify causal effect expression and return target estimands under assumption
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    # print("True causal estimate is", data["ate"])

    print("--------------- SLearner")
    Slearner_estimate = model.estimate_effect(identified_estimand,
                                                            method_name="backdoor.econml.metalearners.SLearner",
                                                            confidence_intervals=False,
                                                            method_params={"init_params": {
                                                                'overall_model': GradientBoostingRegressor()
                                                            },
                                                                "fit_params": {}
                                                            })

    print(Slearner_estimate)  # DoWhy CausalEstimate
    print(Slearner_estimate.cate_estimates)  # train CATE predictions
    print(Slearner_estimate.estimator.estimator)  # econml CATE estimator
    print(Slearner_estimate.estimator.estimator.overall_model)

    print("--------------- TLearner")
    Tlearner_estimate = model.estimate_effect(identified_estimand,
                                                            method_name="backdoor.econml.metalearners.TLearner",
                                                            confidence_intervals=False,
                                                            method_params={"init_params": {
                                                                'models': GradientBoostingRegressor()
                                                            },
                                                                "fit_params": {}
                                                            })

    print(Tlearner_estimate)
    print(Tlearner_estimate.cate_estimates)
    print(Tlearner_estimate.estimator.estimator)
    print(Tlearner_estimate.estimator.estimator.models)

    print("--------------- XLearner")
    Xlearner_estimate = model.estimate_effect(identified_estimand,
                                                            method_name="backdoor.econml.metalearners.XLearner",
                                                            confidence_intervals=False,
                                                            method_params={"init_params": {
                                                                'models': GradientBoostingRegressor(),
                                                                "propensity_model": GradientBoostingClassifier()
                                                                ,"cate_models": GradientBoostingRegressor()}
                                                                ,"fit_params": {"inference": 'bootstrap'}})

    print(Xlearner_estimate)
    print(Xlearner_estimate.cate_estimates)
    print(Xlearner_estimate.estimator.estimator)
    print(Xlearner_estimate.estimator.estimator.models)
    print(Xlearner_estimate.estimator.estimator.propensity_models)
    print(Xlearner_estimate.estimator.estimator.cate_controls_models)
    print(Xlearner_estimate.estimator.estimator.cate_treated_models)

    print("--------------- DML")
    dml_estimate = model.estimate_effect(identified_estimand,
                                         method_name="backdoor.econml.dml.DML",
                                         target_units="ate",
                                         confidence_intervals=True,
                                         method_params={"init_params": {'model_y': GradientBoostingRegressor(),
                                                                        'model_t': GradientBoostingRegressor(),
                                                                        "model_final": LassoCV(fit_intercept=False),
                                                                        'featurizer': PolynomialFeatures(degree=1,
                                                                                                         include_bias=True)},
                                                        "fit_params": {
                                                            'inference': BootstrapInference(n_bootstrap_samples=100,
                                                                                            n_jobs=-1),
                                                        }
                                                        })
    print(dml_estimate)
    print(dml_estimate.cate_estimates)

    print("--------------- LinearDML")
    lineardml_estimate = model.estimate_effect(identified_estimand,
                                         method_name="backdoor.econml.dml.LinearDML",
                                         target_units="ate",
                                         confidence_intervals=True,
                                         method_params={"init_params": {'model_y': LassoCV(),
                                                                        'model_t': LassoCV()},
                                                        "fit_params": {
                                                            'inference': BootstrapInference(n_bootstrap_samples=100,
                                                                                            n_jobs=-1),
                                                        }
                                                        })
    print(lineardml_estimate)
    print(lineardml_estimate.cate_estimates)

    print("--------------- LinearDRLearner")
    linear_drlearner_estimate = model.estimate_effect(identified_estimand,
                                                      method_name="backdoor.econml.dr.LinearDRLearner",
                                                      confidence_intervals=False,
                                                      method_params={"init_params": {
                                                          'model_propensity': LogisticRegressionCV(cv=3,
                                                                                                   solver='lbfgs',
                                                                                                   multi_class='auto')
                                                      },
                                                          "fit_params": {}
                                                      })
    print(linear_drlearner_estimate)
    print(linear_drlearner_estimate.cate_estimates)

    # # TODO: debug DRLearner ValueError: Expected 2D array, got scalar array instead: array=nan.
    # print("--------------- DRLearner")
    # drlearner_estimate = model.estimate_effect(identified_estimand,
    #                                      method_name="backdoor.econml.dr.DRLearner",
    #                                      target_units="ate",
    #                                      confidence_intervals=False,
    #                                      method_params={"init_params":
    #                                                         {'model_regression':
    #                                                              GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100)),
    #                                                          'model_propensity':
    #                                                             RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100)),
    #                                                          'model_final':
    #                                                              GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100)),
    #                                                          "cv": 5},
    #                                                     "fit_params": {}
    #                                                     })
    # print(drlearner_estimate)
    # print(drlearner_estimate.cate_estimates)
    #
    # # TODO: DEBUG CausalForestDML ValueError: This estimator does not support X=None!
    # print("--------------- CausalForestDML")
    # causal_forest_dml_estimate = model.estimate_effect(identified_estimand,
    #                                                   method_name="backdoor.econml.dml.CausalForestDML",
    #                                                   confidence_intervals=False,
    #                                                   method_params={"init_params": {
    #                                                       "criterion": 'het', "n_estimators": 500,
    #                                                       "min_samples_leaf": 10,
    #                                                       "max_depth": 10,
    #                                                       "max_samples": 0.5,
    #                                                       "discrete_treatment": True,
    #                                                       "model_t": LassoCV(),
    #                                                       "model_y": LassoCV()},
    #                                                       "fit_params": {}})
    # print(causal_forest_dml_estimate)
    # print(causal_forest_dml_estimate.cate_estimates)

    print("--------------- DomainAdaptationLearner")
    domain_adaptation_estimate = model.estimate_effect(identified_estimand,
                                                      method_name="backdoor.econml.metalearners.DomainAdaptationLearner",
                                                      confidence_intervals=False,
                                                       method_params={"init_params": {
                                                           "models": GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100)),
                                                           "final_models": GradientBoostingRegressor(n_estimators=100,
                                                                                                    max_depth=6,
                                                                                                    min_samples_leaf=int(
                                                                                                        n / 100)),
                                                           "propensity_model": RandomForestClassifier(n_estimators=100, max_depth=6,
                                                      min_samples_leaf=int(n/100))},
                                                          "fit_params": {}})
    print(domain_adaptation_estimate)
    print(domain_adaptation_estimate.cate_estimates)


if __name__ == '__main__':
    # run_dowhy_examples()
    tlearner_obj = TCausalLearner()
    tlearner_obj.set_method_init_params_dict(GradientBoostingRegressor())
    print("debug")