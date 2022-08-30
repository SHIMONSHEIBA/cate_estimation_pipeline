# cate_estimation_pipeline
Configurable experimentation code for Conditional Average Treatment Effect (CATE) for binary intervention and binary outcome


1. data_pre_processing.py: Handles data (simulated data created for example and debug runs: saved in data/simulated_data_dict.pkl)
manage experiment configurations in config/data_process_config.yaml, conf/files/simulated_data.yaml & utils_config.py

2. main.py: manage all pipeline steps
manage experiment configurations in config/cate_config.yaml, conf/files/simulated_data_processed.yaml & utils_config.py

pipeline steps:
- data scaling
- feature selection
- domain knowledge user input validation
- propensity estimation and ml models evaluation (nested CV fit iterated via top shapley values for reducing overfitting)
- common support causal assumption validation via propensity trimming
- causal meta learners fit: outcome estimation and ml models evaluation (nested CV fit iterated via top shapley values for reducing overfitting)
- causal discovery: graph generation 
- CATE analysis (plots, agreement between different estimators, ATE, correlations)
- Intervention policy creation
- Intervention policy estimation: doubly robust policy value shown in bootstrapped box plots for confidence
- Sub population analysis - helper regression model to explain intervention policy decisions
