# cate_estimation_pipeline
Configurable experimentation code for Conditional Average Treatment Effect (CATE) estimation of binary intervention on a binary outcome
CATE defined as the difference in intervention response: For binary Y & T, and for feature vector X as CATE(X) = E[Y|T=1,X] - E[Y|T=0,X]

1. data_pre_processing.py: Basic data processing (missing values, collinearity, outliers handling, data split, categorical handling)
Simulated data created with save_simulated_data() as an example run: saved in data/simulated_data_dict.pkl
Manage experiment configurations in config/data_process_config.yaml, conf/files/simulated_data.yaml & utils_config.py

2. main.py: Manage calls of all pipeline steps
Manage experiment configurations in config/cate_config.yaml, conf/files/simulated_data_processed.yaml & utils_config.py

Pipeline steps:
- Data load
- Data scaling
- Feature selection (domain knowledge user input validation)
- Propensity estimation and ml models evaluation (nested CV fit iterated via top shapley values for reducing overfitting)
- Common support causal assumption validation via propensity trimming
- Causal meta learners fit: outcome estimation and ml models evaluation (nested CV fit iterated via top shapley values for reducing overfitting)
- Causal discovery: graph generation 
- CATE analysis (plots, agreement between different estimators, ATE, correlations)
- Intervention policy creation
- Intervention policy estimation: doubly robust policy value shown in bootstrapped box plots for confidence
- Sub population analysis - helper regression model to explain intervention policy decisions
