# cate_estimation_pipeline
Configurable experimentation code for Conditional Average Treatment Effect (CATE) estimation of binary intervention on a binary outcome
CATE defined as the difference in intervention response: For binary Y & T, and for feature vector X as CATE(X) = E[Y|T=1,X] - E[Y|T=0,X]

1. data_pre_processing.py: Basic data processing (simulated data created with save_simulated_data() as an example run: saved in data/simulated_data_dict.pkl)
manage experiment configurations in config/data_process_config.yaml, conf/files/simulated_data.yaml & utils_config.py

2. main.py: manage calls of all pipeline steps
manage experiment configurations in config/cate_config.yaml, conf/files/simulated_data_processed.yaml & utils_config.py

pipeline steps:
- Data scaling
- Feature selection
- Domain knowledge user input validation
- Propensity estimation and ml models evaluation (nested CV fit iterated via top shapley values for reducing overfitting)
- Common support causal assumption validation via propensity trimming
- Causal meta learners fit: outcome estimation and ml models evaluation (nested CV fit iterated via top shapley values for reducing overfitting)
- Causal discovery: graph generation 
- CATE analysis (plots, agreement between different estimators, ATE, correlations)
- Intervention policy creation
- Intervention policy estimation: doubly robust policy value shown in bootstrapped box plots for confidence
- Sub population analysis - helper regression model to explain intervention policy decisions
