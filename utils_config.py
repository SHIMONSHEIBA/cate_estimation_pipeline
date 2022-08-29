from dataclasses import dataclass
# from typing import List

# ---- Data pre processing data classes


@dataclass
class Paths:
    log: str
    data: str


@dataclass
class Files:
    data_object: str


@dataclass
class DataProcessParams:
    train_test_split_method: str
    min_test_year: int
    test_size: float
    impute: bool
    categorical: str


@dataclass
class DataProcessConfig:
    paths: Paths
    files: Files
    params: DataProcessParams

# ---- CATE modeling pipeline data classes


@dataclass
class Params:
    data_model_update_date: str
    update_date_buffer_months: int
    index_date_col: str
    no_covid: bool
    outcome_name: str
    outcome_type: str
    outcome_is_cost: bool
    scaler_name: str
    causal_learner_type: str
    run_feature_selection: bool
    fs_method: str
    mutual_confounders: bool
    add_domain_expert_features: bool
    n_selected_features: int
    boruta_clf_arg_dict: dict
    boruta_perc: int
    boruta_alpha: float
    clf_name_list: list
    score_name: str
    inner_fold_num: int
    outer_fold_num: int
    skip_propensity_trimming: bool
    test_propensity: bool
    interactive_env: bool
    upsample: bool
    causal_discovery: bool
    d_top_outcome_shap: int
    d_top_prop_shap: int
    ftrs_to_exclude: list
    mandatory_features: list


@dataclass
class CATEconfig:
    paths: Paths
    files: Files
    params: Params
