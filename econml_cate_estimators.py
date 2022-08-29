""" A dump of EconMl CATE examples"""

import numpy as np
from econml.dml import DML, LinearDML, CausalForestDML, NonParamDML
from econml.inference import BootstrapInference
from econml.metalearners import XLearner, SLearner, TLearner, DomainAdaptationLearner
from econml.dr import DRLearner, LinearDRLearner
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
import shap
from sklearn.model_selection import GridSearchCV
from utils import generate_controls_outcome, generate_data


def run_econ_examples():
    print("running econml examples")

    # define data
    d = 5
    n = 1000
    n_test = 250
    controls_outcome = generate_controls_outcome(d)
    X_test = np.random.multivariate_normal(np.zeros(d), np.diag(np.ones(d)), n_test)
    delta = 6/n_test
    X_test[:, 1] = np.arange(-3, 3, delta)
    treatment_effect = lambda x: (1 if x[1] > 0.1 else 0)*8
    propensity = lambda x: (0.8 if (x[2]>-0.5 and x[2]<0.5) else 0.2)

    Y, T, X = generate_data(n, d, controls_outcome, treatment_effect, propensity)

    # S-learner
    s_est = SLearner(overall_model=GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100)))
    s_est.fit(Y, T, X=X, inference='bootstrap') #X=np.hstack([X, W]))
    treatment_effects = s_est.effect(X_test) # np.hstack([X_test, W_test]))
    print("--------------------SLearner")
    print(treatment_effects)

    # T-learner
    t_est = TLearner(models=GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100)))
    t_est.fit(Y, T, X=X) #np.hstack([X, W]))
    treatment_effects = t_est.effect(X_test) #np.hstack([X_test, W_test]))
    print("--------------------TLearner")
    print(treatment_effects)

    # X-learner
    x_est = XLearner(models=GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100)),
                  propensity_model=GradientBoostingClassifier(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100)),
                  cate_models=GradientBoostingRegressor())
    x_est.fit(Y, T, X=X) #np.hstack([X, W]))
    X_treatment_effects = x_est.effect(X_test) #np.hstack([X_test, W_test]))
    print("--------------------XLearner")
    print(X_treatment_effects)

    # R-learner
    r_est = NonParamDML(model_y=GradientBoostingRegressor(),
                        model_t=GradientBoostingRegressor(),
                        model_final=RandomForestRegressor())
    r_est.fit(Y=Y, T=T, X=X)
    treatment_effects = r_est.effect(X_test)
    print("--------------------R_learner")
    print(treatment_effects)

    # R-learner grid search
    cv_reg = lambda: GridSearchCV(
                estimator=RandomForestRegressor(),
                param_grid={
                        'max_depth': [3, None],
                        'n_estimators': (10, 30, 50, 100, 200, 400, 600, 800, 1000),
                        'max_features': (1, 2, 3)
                    }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                )
    GridSearchCV_R_est = NonParamDML(model_y=cv_reg(), model_t=cv_reg(), model_final=cv_reg())
    GridSearchCV_R_est.fit(Y=Y, T=T, X=X)
    GridSearchCV_R_est_te = GridSearchCV_R_est.effect(X_test)
    print("--------------------R_learner grid search")
    print(GridSearchCV_R_est_te)

    # Double machine learning
    dml_est = DML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingRegressor(),
                  model_final=LassoCV(fit_intercept=False), featurizer=PolynomialFeatures(degree=1, include_bias=True))
    dml_est.fit(Y=Y, T=T, X=X, inference=BootstrapInference(n_bootstrap_samples=100,
                                                            n_jobs=-1)) #, W=W) # W -> high-dimensional confounders, X -> features
    treatment_effects = dml_est.effect(X_test)
    print("--------------------DML")
    print(treatment_effects)

    # Linear double machine learning
    linear_dml_est = LinearDML(model_y=LassoCV(), model_t=LassoCV())
    linear_dml_est.fit(Y, T, X=X) #, W=W) # W -> high-dimensional confounders, X -> features
    treatment_effects = linear_dml_est.effect(X_test)
    lb, ub = linear_dml_est.effect_interval(X_test, alpha=0.05)  # OLS confidence intervals
    print("--------------------LinearDML")
    print(treatment_effects)
    print("--------------------LinearDML OLS confidence intervals")
    print("lb: {}, ub: {} ".format(lb, ub))

    # Estimate with bootstrap confidence intervals
    linear_dml_est.fit(Y, T, X=X, inference='bootstrap')  # with default bootstrap parameters
    treatment_effects = linear_dml_est.effect(X_test)
    print("--------------------LinearDML inference='bootstrap'")
    print(treatment_effects)
    linear_dml_est.fit(Y, T, X=X, inference=BootstrapInference(n_bootstrap_samples=100))  # or customized
    treatment_effects = linear_dml_est.effect(X_test)
    print("--------------------LinearDML inference=BootstrapInference(n_bootstrap_samples=100)")
    print(treatment_effects)

    # Linear doubly robust learner
    linear_dr_est = LinearDRLearner(model_propensity=LogisticRegressionCV(cv=3, solver='lbfgs', multi_class='auto'))
    linear_dr_est.fit(Y, T, X=X)  # , W=W) # W -> high-dimensional confounders, X -> features
    treatment_effects = linear_dr_est.effect(X_test)
    print("--------------------LinearDRLearner")
    print(treatment_effects)

    # Doubly robust learner
    dr_est = DRLearner(model_regression=
                       GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100)),
                       model_propensity=
                       RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100)),
                       model_final=
                       GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100)),
                       cv=5)
    dr_est.fit(Y, T, X=X)  # , W=W) # W -> high-dimensional confounders, X -> features
    treatment_effects = dr_est.effect(X_test)
    print("--------------------DRLearner")
    print(treatment_effects)

    # Causal forest learner
    cf_est = CausalForestDML(criterion='het', n_estimators=500,
                          min_samples_leaf=10,
                          max_depth=10, max_samples=0.5,
                          discrete_treatment=False,
                          model_t=LassoCV(), model_y=LassoCV())
    cf_est.fit(Y, T, X=X)
    treatment_effects = cf_est.effect(X_test)
    print("--------------------CausalForestDML")
    print(treatment_effects)

    # Confidence intervals via Bootstrap-of-Little-Bags for forests
    lb, ub = cf_est.effect_interval(X_test, alpha=0.05)
    shap_values = cf_est.shap_values(X)
    shap.summary_plot(shap_values['Y0']['T0'])

    # Domain adaptation learner
    da_learner = DomainAdaptationLearner(models=GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100)),
                                         final_models=GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100)),
                                         propensity_model=RandomForestClassifier(n_estimators=100, max_depth=6,
                                                      min_samples_leaf=int(n/100)))
    da_learner.fit(Y, T, X=X)
    treatment_effects = da_learner.effect(X_test)
    print("--------------------Domain Adaptation learner")
    print(treatment_effects)


if __name__ == '__main__':
    run_econ_examples()
