import joblib
import pandas as pd
import os
from hydra.core.config_store import ConfigStore
import hydra
from copy import deepcopy
import logging
# internal
from causalis_graph import CausalisGraph
from propensity_model import PropensityModel
from outcome_model import outcome_modeling
from cate_evaluation import CateEvaluation
from policy_creation import PolicyCreation
from policy_estimation import PolicyEstimation
from sub_population_analysis import SubPopulationAnalysis
from utils_config import CATEconfig
from utils_ml import feature_selection, sort_features_by_shap_values, get_score_params, \
    get_scaler_params, get_binary_classifier_models_dict, binary_clf_eval
from utils_domain import domain_expert_features_lists
from utils_graphs import plot_boxplot
# display
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


@hydra.main(config_path="conf", config_name="cate_config.yaml")
def run_cate_pipeline(cfg: CATEconfig) -> None:
    """
    func responsible on all calls of different steps in CATE estimation pipeline
    """

    log.info("starting experiment on outcome {} with causal meta learners {}".format(cfg.params.outcome_name, 
                                                                                     cfg.params.causal_learner_type))

    # load data
    processed_data_obj_list = joblib.load(os.path.join(cfg.paths.data, cfg.files.data_object))
    train_data, test_data, treatment_name, treatment0_name, treatment1_name, confounder_names_list, \
    post_treatment_ftrs, train_impute_numeric_df, categorical_feature_names, cat_cols_mapping_dict \
        = processed_data_obj_list
    
    log.info("train shape: {}".format(train_data.shape))
    log.info("test shape: {}".format(test_data.shape))

    # exclude features from experiment
    confounder_names_list = [x for x in confounder_names_list if not any([substring in x for substring in
                                                                          cfg.params.ftrs_to_exclude])]

    # see outcome counts per treatment arm in train/test
    log.info("{} outcome distribution per {} treatment arm (train):".format(cfg.params.outcome_name, treatment_name))
    log.info(train_data.groupby(treatment_name)[cfg.params.outcome_name].value_counts(dropna=False))
    log.info("{} outcome distribution per {} treatment arm (test):".format(cfg.params.outcome_name, treatment_name))
    log.info(test_data.groupby(treatment_name)[cfg.params.outcome_name].value_counts(dropna=False))
    log.info(train_data[cfg.params.outcome_name].value_counts(dropna=False))
        
    # set score
    score, greater_is_better = get_score_params(cfg.params.score_name)

    # ---------- Scale data ----------
    if cfg.params.scaler_name:
        scaler = get_scaler_params(cfg.params.scaler_name)
        ftrs_to_scale = [x for x in confounder_names_list if x != treatment_name]
        log.info("STARTING scaling data with {} scaler".format(cfg.params.scaler_name))
        scaler.fit(train_data[ftrs_to_scale])
        joblib.dump(scaler, os.path.join(os.getcwd(), "scaler_{}.pkl".format(cfg.params.scaler_name)))
        joblib.dump(ftrs_to_scale, os.path.join(os.getcwd(), "ftrs_to_scale_{}.pkl".format(cfg.params.scaler_name)))
        train_data[ftrs_to_scale] = scaler.transform(train_data[ftrs_to_scale])
        test_data[ftrs_to_scale] = scaler.transform(test_data[ftrs_to_scale])
        log.info("FINISHED scaling data with {} scaler".format(cfg.params.scaler_name))

    joblib.dump(confounder_names_list, "confounder_names_list_t_{}_y_{}.pkl".format(
        treatment_name, cfg.params.outcome_name))

    # ---------- Feature selection ----------
    # feature selection
    chosen_features_dict = dict()
    treatment_values_list = list(train_data[treatment_name].unique())

    if cfg.params.run_feature_selection:
        # run FS for each arm
        for t_arm in treatment_values_list:
            if t_arm == 1:
                log.info("running features selection {}".format(cfg.params.fs_method))
                chosen_features_dict[t_arm] = \
                    feature_selection(x=train_data[confounder_names_list],
                                      y=train_data[cfg.params.outcome_name],
                                      arg_dict=cfg.params.boruta_clf_arg_dict,
                                      col_names_list=confounder_names_list,
                                      method=cfg.params.fs_method,
                                      n_selected_features=cfg.params.n_selected_features,
                                      outcome=cfg.params.outcome_type,
                                      perc=cfg.params.boruta_perc,
                                      alpha=cfg.params.boruta_alpha)
        # TODO: CLEAN HACK IN ALL PIPELINE CODE ABOUT chosen_features_dict
        chosen_features_dict[0] = chosen_features_dict[1]

    # no FS
    else:
        for t_arm in treatment_values_list:
            chosen_features_dict[t_arm] = list()

    for arm in chosen_features_dict.keys():
        # add mandatory features
        chosen_features_dict[arm].extend(cfg.params.mandatory_features)
        chosen_features_dict[arm] = list(set(chosen_features_dict[arm]))

        if cfg.params.add_domain_expert_features:

            domain_expert_features_treatment0 = domain_expert_features_lists(outcome_name=cfg.params.outcome_name,
                                                                             treatment0_name=treatment0_name,
                                                                             treatment1_name=treatment1_name,
                                                                             cur_treatment=treatment0_name)
            domain_expert_features_treatment1 = domain_expert_features_lists(outcome_name=cfg.params.outcome_name,
                                                                             treatment0_name=treatment0_name,
                                                                             treatment1_name=treatment1_name,
                                                                             cur_treatment=treatment1_name)
            # add domain expert features
            if arm == 0:
                log.info("adding domain expert features treatment 0: {}".format(domain_expert_features_treatment0))
                chosen_features_dict[arm].extend(domain_expert_features_treatment0)
                chosen_features_dict[arm] = \
                    list(set(chosen_features_dict[arm]))
            elif arm == 1:
                log.info("adding domain expert features treatment 1: {}".format(domain_expert_features_treatment1))
                chosen_features_dict[arm].extend(domain_expert_features_treatment1)
                chosen_features_dict[arm] = list(set(chosen_features_dict[arm]))
            elif arm == 'all':
                log.info("adding domain expert features both treatments: {}, {}".format(
                    domain_expert_features_treatment0, domain_expert_features_treatment1))
                chosen_features_dict[arm].extend(domain_expert_features_treatment0)
                chosen_features_dict[arm].extend(domain_expert_features_treatment1)
                chosen_features_dict[arm] = list(set(chosen_features_dict[arm]))
            else:
                log.info("NO DOMAIN expert FEATURES ADDED - CHECK TREATMENT ARMS IN CHOSEN_FEATURES_DICT")

    joblib.dump(chosen_features_dict, "chosen_features_dict_method_{}_t_{}_y_{}.pkl".format(cfg.params.fs_method,
                                                                                            treatment_name,
                                                                                            cfg.params.outcome_name))

    # ------- ML models estimation ------------
    # TODO: customize objective metric - add F1
    # TODO: validate when no FS
    if cfg.params.mutual_confounders:
        # take only mutual between arms
        all_chosen_features = list(set(chosen_features_dict[0]).intersection(chosen_features_dict[1]))
    else:
        # all chosen per all arms
        all_chosen_features = list(set([item for sublist in chosen_features_dict.values() for item in sublist]))

    joblib.dump(all_chosen_features, "all_chosen_features.pkl".format(treatment_name))
    binary_classifier_models_dict = get_binary_classifier_models_dict(clf_name_list=cfg.params.clf_name_list,
                                                                      categorical_names_list=
                                                                      list(set(categorical_feature_names).intersection(
                                                                          all_chosen_features)))

    # ---------- Propensity model ----------
    log.info("training propensity models")
    propensity_model_obj = PropensityModel()
    # fit initial propensity
    propensity_model = propensity_model_obj.fit_prop(train_data,
                                                     treatment_name,
                                                     os.getcwd(),
                                                     all_chosen_features,
                                                     binary_classifier_models_dict,
                                                     cfg.params.outer_fold_num,
                                                     cfg.params.inner_fold_num,
                                                     score,
                                                     cfg.params.score_name,
                                                     greater_is_better,
                                                     cfg.params.interactive_env)

    # calc propensity features shap importance
    top_propensity_shap_features = sort_features_by_shap_values(model=propensity_model,
                                                                data=train_data,
                                                                features_for_model=[x for x in all_chosen_features
                                                                                    if x != treatment_name])
    # take propensity with only top features d: d_top_prop_shap
    log.info("PROPENSITY: Taking top shap {}".format(cfg.params.d_top_prop_shap))
    top_propensity_shap_features = top_propensity_shap_features[0: cfg.params.d_top_prop_shap]
    # add mandatory features
    top_propensity_shap_features.extend(cfg.params.mandatory_features)
    top_propensity_shap_features = list(set(top_propensity_shap_features))

    log.info("top_propensity_shap_features: {}".format(top_propensity_shap_features))
    # re-fit with less features
    propensity_top_shap_path = os.path.join(os.getcwd(), "top_shap_propensity")
    os.makedirs(propensity_top_shap_path, exist_ok=True)

    # update categorical features
    binary_classifier_models_dict = get_binary_classifier_models_dict(clf_name_list=cfg.params.clf_name_list,
                                                                      categorical_names_list=
                                                                      list(set(categorical_feature_names).intersection(
                                                                          top_propensity_shap_features)))

    propensity_model = propensity_model_obj.fit_prop(train_data=train_data,
                                                     treatment_name=treatment_name,
                                                     modeling_path=propensity_top_shap_path,
                                                     all_chosen_features=top_propensity_shap_features,
                                                     binary_classifier_models_dict=
                                                          binary_classifier_models_dict,
                                                     outer_fold_num=cfg.params.outer_fold_num,
                                                     inner_fold_num=cfg.params.inner_fold_num,
                                                     score=score,
                                                     score_name=cfg.params.score_name,
                                                     greater_is_better=greater_is_better,
                                                     interactive_env=cfg.params.interactive_env)

    joblib.dump(top_propensity_shap_features, "top_propensity_shap_features.pkl".format(treatment_name))

    # ---------- Propensity score trimming ----------

    # check effective sample size per treatment arm in different propensity thresholds
    log.info("adding propensity score to train data")
    propensity_score_name = "propensity_score"
    train_data[propensity_score_name] = \
        propensity_model.predict_proba(train_data[top_propensity_shap_features])[:, 1]
    log.info("adding propensity score to test data")
    test_data[propensity_score_name] = \
        propensity_model.predict_proba(test_data[top_propensity_shap_features])[:, 1]

    # check thresholds
    propensity_score_thresholds = [(0.05, 0.95), (0.1, 0.9), (0.15, 0.85), (0.2, 0.8), (0.25, 0.75)]
    for thresh_tuple in propensity_score_thresholds:
        log.info("data count per treatment arm under check_lower_thresh = {}".format(thresh_tuple[0]))
        log.info(train_data.query("{} <= {}".format(propensity_score_name,
                                                    thresh_tuple[0])).groupby(treatment_name)
                 [propensity_score_name].count())

        log.info("data count per treatment arm above check_upper_thresh = {}".format(thresh_tuple[1]))
        log.info(train_data.query("{} >= {}".format(propensity_score_name,
                                                    thresh_tuple[1])).groupby(treatment_name)
                 [propensity_score_name].count())

    # Define propensity trimming thresholds  # TODO: need to be defined dynamically
    lower_propensity_trimming_thresh = 0.1
    upper_propensity_trimming_thresh = 0.9
    log.info("propensity trimming defined between {} {}".format(lower_propensity_trimming_thresh,
                                                                upper_propensity_trimming_thresh))

    # get patient ids that are in overlap region a.k.a common support population
    common_support_train_data_ids = train_data.query(" ({} <= {}) & ({} >= {})".format(propensity_score_name,
                                                                                       upper_propensity_trimming_thresh,
                                                                                       propensity_score_name,
                                                                                       lower_propensity_trimming_thresh)
                                                     ).index

    common_support_test_data_ids = test_data.query(" ({} <= {}) & ({} >= {})".format(propensity_score_name,
                                                                                     upper_propensity_trimming_thresh,
                                                                                     propensity_score_name,
                                                                                     lower_propensity_trimming_thresh)
                                                   ).index

    log.info("trimmed {} in train".format(train_data.shape[0] - len(common_support_train_data_ids)))
    log.info("trimmed {} in test".format(test_data.shape[0] - len(common_support_test_data_ids)))

    joblib.dump(common_support_train_data_ids, "common_support_train_data_ids.pkl")
    joblib.dump(common_support_test_data_ids, "common_support_test_data_ids.pkl")

    if cfg.params.skip_propensity_trimming:
        log.info("keeping all data - no propensity trimming")
        common_support_train_data_ids = train_data.index
        common_support_test_data_ids = test_data.index

    # ---------------------- Model outcome --------------------------
    outcome_path = os.path.join(os.getcwd(), 'outcome_models')
    os.makedirs(outcome_path, exist_ok=True)

    # update categorical features
    binary_classifier_models_dict = get_binary_classifier_models_dict(clf_name_list=cfg.params.clf_name_list,
                                                                      categorical_names_list=
                                                                      list(set(categorical_feature_names).intersection(
                                                                          all_chosen_features)))

    chosen_outcome_model_name, chosen_outcome_model_name_dict, _, _ = \
        outcome_modeling(causal_learner_type=["tlearner"],
                         train_data=train_data.loc[common_support_train_data_ids, :],
                         test_data=test_data.loc[common_support_test_data_ids, :],
                         all_chosen_features=all_chosen_features,
                         treatment_name=treatment_name,
                         outcome_name=cfg.params.outcome_name,
                         greater_is_better=greater_is_better,
                         score=score,
                         outcome_path=outcome_path,
                         chosen_features_dict=chosen_features_dict,
                         treatment_values_list=treatment_values_list,
                         binary_classifier_models_dict=binary_classifier_models_dict,
                         outer_fold_num=cfg.params.outer_fold_num,
                         inner_fold_num=cfg.params.inner_fold_num,
                         score_name=cfg.params.score_name,
                         upsample=cfg.params.upsample,
                         interactive_env=cfg.params.interactive_env)

    # to reduce d of y models - take top d_top_outcome_shap features + mandatory features
    top_outcome_y0_shap_features = sort_features_by_shap_values(
        model=chosen_outcome_model_name_dict[0][chosen_outcome_model_name],
        data=train_data.loc[common_support_train_data_ids, :],
        features_for_model=chosen_features_dict[0])

    top_outcome_y1_shap_features = sort_features_by_shap_values(
        model=chosen_outcome_model_name_dict[1][chosen_outcome_model_name],
        data=train_data.loc[common_support_train_data_ids, :],
        features_for_model=chosen_features_dict[1])

    log.info("Outcome: Taking top shap {}".format(cfg.params.d_top_outcome_shap))
    top_outcome_y0_shap_features = top_outcome_y0_shap_features[0: cfg.params.d_top_outcome_shap]
    top_outcome_y1_shap_features = top_outcome_y1_shap_features[0: cfg.params.d_top_outcome_shap]

    # add mandatory features
    top_outcome_y0_shap_features.extend(cfg.params.mandatory_features)
    top_outcome_y0_shap_features = list(set(top_outcome_y0_shap_features))
    top_outcome_y1_shap_features.extend(cfg.params.mandatory_features)
    top_outcome_y1_shap_features = list(set(top_outcome_y1_shap_features))

    # re-fit with less features
    outcome_top_shap_path = os.path.join(outcome_path, "top_shap_outcome")
    os.makedirs(outcome_top_shap_path, exist_ok=True)

    # update features dict for re-modeling outcome
    # chosen_features_dict_shap_step = defaultdict(dict)
    chosen_features_dict[0] = top_outcome_y0_shap_features
    chosen_features_dict[1] = top_outcome_y1_shap_features

    # validate propensity features are in outcome models
    logging.warning("See propensity features that are not in Y0: {}".format(set(top_propensity_shap_features).difference(
        chosen_features_dict[0])))
    logging.warning("See propensity features that are not in Y1: {}".format(set(top_propensity_shap_features).difference(
        chosen_features_dict[1])))

    top_outcome_shap_features = list(set(chosen_features_dict[0]).union(chosen_features_dict[1]))

    log.info("top_outcome_shap_features: {}".format(top_outcome_shap_features))
    joblib.dump(top_outcome_shap_features, os.path.join(outcome_top_shap_path, "top_outcome_shap_features.pkl"))
    joblib.dump(chosen_features_dict, os.path.join(outcome_top_shap_path, "chosen_features_dict_shap_step.pkl"))

    # update categorical features
    binary_classifier_models_dict = get_binary_classifier_models_dict(clf_name_list=cfg.params.clf_name_list,
                                                                      categorical_names_list=
                                                                      list(set(categorical_feature_names).intersection(
                                                                          top_outcome_shap_features)))

    # model again with top features to reduce over-fitting
    chosen_outcome_model_name, chosen_outcome_model_name_dict, rlearner_obj, xlearner_obj = \
        outcome_modeling(causal_learner_type=cfg.params.causal_learner_type,
                         train_data=train_data.loc[common_support_train_data_ids, :],
                         test_data=test_data.loc[common_support_test_data_ids, :],
                         all_chosen_features=top_outcome_shap_features,
                         treatment_name=treatment_name,
                         outcome_name=cfg.params.outcome_name,
                         greater_is_better=greater_is_better,
                         score=score,
                         outcome_path=outcome_top_shap_path,
                         chosen_features_dict=chosen_features_dict,
                         treatment_values_list=treatment_values_list,
                         binary_classifier_models_dict=binary_classifier_models_dict,
                         outer_fold_num=cfg.params.outer_fold_num,
                         inner_fold_num=cfg.params.inner_fold_num,
                         score_name=cfg.params.score_name,
                         upsample=cfg.params.upsample,
                         interactive_env=cfg.params.interactive_env)

# ----------- Causal Discovery -----------------
    if cfg.params.causal_discovery:
        # run causal discovery
        graph_path = os.path.join(os.getcwd(), 'causal_graph')
        os.makedirs(graph_path, exist_ok=True)
        graph_ftrs = top_outcome_shap_features
        graph_ftrs = [x for x in graph_ftrs if x != treatment_name]
        # TODO: fix bug that post treatment are not scaled - affects NOTEARS ALG
        # graph_ftrs.extend(post_treatment_ftrs)
        causalis_graph_obj = CausalisGraph(data=train_data.loc[common_support_train_data_ids, :],
                                           treatment=treatment_name,
                                           outcome=cfg.params.outcome_name,
                                           graph=None,
                                           common_causes=graph_ftrs,
                                           instruments=None,
                                           effect_modifiers=None,
                                           experiment_name="{}_{}_{}".format(treatment0_name, treatment1_name,
                                                                             cfg.params.outcome_name),
                                           path=os.path.join(graph_path, "train"))

        _ = causalis_graph_obj.run_causalnex_notears()

        # test data
        causalis_graph_obj.data = test_data.loc[common_support_test_data_ids, :]
        causalis_graph_obj.path = os.path.join(graph_path, "test")
        _ = causalis_graph_obj.run_causalnex_notears()

# ----- Analyze CATE
    cate_path = os.path.join(os.getcwd(), "CATE")
    os.makedirs(cate_path, exist_ok=True)

    log.info("CATE evaluation")

    # TODO: add a baseline CATE like based on PD-L1, age and ECOG - ask clinician
    cate_obj = CateEvaluation()
    # train
    cate_df_dict_train = cate_obj.temp_calc_cate_by_learner(causal_learner_type=cfg.params.causal_learner_type,
                                                            data=train_data.loc[common_support_train_data_ids, :],
                                                            treatment_name=treatment_name,
                                                            chosen_features_dict=chosen_features_dict,
                                                            chosen_outcome_model_name_dict=chosen_outcome_model_name_dict,
                                                            chosen_outcome_model_name=chosen_outcome_model_name,
                                                            top_outcome_shap_features=top_outcome_shap_features,
                                                            rlearner_obj=rlearner_obj, xlearner_obj=xlearner_obj)
    cates_df_train = cate_obj.get_cate_df(cate_df_dict_train)

    # plot all cates
    cate_obj.plot_cates(path=cate_path, cates_df=cates_df_train, title="CATE_HISTOGRAM_TRAIN")

    # get agreement/disagreement sets between all learners
    all_cates_agreement_set_ids_train = cate_obj.cate_define_agreement_set(cates_df_train)

    # GET AVERAGE CATE
    cates_df_train = cate_obj.get_avg_cate(cates_df_train)
    joblib.dump(cate_df_dict_train, os.path.join(cate_path, "cate_df_dict_train.pkl"))
    joblib.dump(cates_df_train, os.path.join(cate_path, "cates_df_train.pkl"))
    joblib.dump(all_cates_agreement_set_ids_train, os.path.join(cate_path, "all_cates_agreement_set_ids_train.pkl"))

    # test
    cate_df_dict_test = cate_obj.temp_calc_cate_by_learner(causal_learner_type=cfg.params.causal_learner_type,
                                                           data=test_data.loc[common_support_test_data_ids, :],
                                                           treatment_name=treatment_name,
                                                           chosen_features_dict=chosen_features_dict,
                                                           chosen_outcome_model_name_dict=chosen_outcome_model_name_dict,
                                                           chosen_outcome_model_name=chosen_outcome_model_name,
                                                           top_outcome_shap_features=top_outcome_shap_features,
                                                           rlearner_obj=rlearner_obj,
                                                           xlearner_obj=xlearner_obj)
    cates_df_test = cate_obj.get_cate_df(cate_df_dict_test)

    cate_obj.plot_cates(path=cate_path, cates_df=cates_df_test, title="CATE_HISTOGRAM_TEST")

    all_cates_agreement_set_ids_test = cate_obj.cate_define_agreement_set(cates_df_test)

    # GET AVERAGE CATE
    cates_df_test = cate_obj.get_avg_cate(cates_df_test)
    joblib.dump(cate_df_dict_test, os.path.join(cate_path, "cate_df_dict_test.pkl"))
    joblib.dump(cates_df_test, os.path.join(cate_path, "cates_df_test.pkl"))
    joblib.dump(all_cates_agreement_set_ids_test, os.path.join(cate_path, "all_cates_agreement_set_ids_test.pkl"))

    # calc correlations between CATE methods
    cate_obj.cate_correlations(cates_df_train, dataset="TRAIN")
    cate_obj.cate_correlations(cates_df_test, dataset="TEST")

    # calc ATE between CATE methods
    cate_obj.cate_ate_validation(cates_df_train, dataset="TRAIN")
    cate_obj.cate_ate_validation(cates_df_test, dataset="TEST")

    # cate_obj.cate_calibration_plots(cates_df)

    # cate_obj.cate_regress_features_on_doubly_robust(cates_df)

    # ----- Create intervention policy
    policy_path = os.path.join(os.getcwd(), "Policy")
    os.makedirs(policy_path, exist_ok=True)

    log.info("Policy Creation")

    policy_create_obj = PolicyCreation(outcome_is_cost=cfg.params.outcome_is_cost, no_rec_value=-999)
    # train
    # create policies
    policies_df_train = policy_create_obj.calc_policy(cates_df=cates_df_train, cols_to_exclude=["cate_avg_all"])

    # add baseline policies
    policies_df_train = policy_create_obj.add_baseline_policies(policies_df=policies_df_train,
                                                                curr_policy=train_data.loc[common_support_train_data_ids, 
                                                                                           treatment_name].values)

    joblib.dump(policies_df_train, os.path.join(policy_path, "policies_df_train.pkl"))
    # test
    # create policies
    policies_df_test = policy_create_obj.calc_policy(cates_df=cates_df_test, cols_to_exclude=["cate_avg_all"])

    # add baseline policies
    policies_df_test = policy_create_obj.add_baseline_policies(policies_df=policies_df_test,
                                                               curr_policy=test_data.loc[common_support_test_data_ids,
                                                                                         treatment_name].values)

    joblib.dump(policies_df_test, os.path.join(policy_path, "policies_df_test.pkl"))

    log.info("Evaluate Policy value")

    # evaluate a policy
    policy_est_obj = PolicyEstimation()

    # add policies
    sample_data_train = train_data.loc[common_support_train_data_ids, :].merge(policies_df_train,
                                                                               left_index=True, right_index=True)

    # add predicted y columns for policy value
    sample_data_train["y0_hat"] = \
        chosen_outcome_model_name_dict["y_hat_policy_0"]["GradientBoostingClassifier"].predict_proba(
            train_data.loc[common_support_train_data_ids, chosen_features_dict[0]])[:, 1]
    sample_data_train["y1_hat"] = \
        chosen_outcome_model_name_dict["y_hat_policy_1"]["GradientBoostingClassifier"].predict_proba(
            train_data.loc[common_support_train_data_ids, chosen_features_dict[1]])[:, 1]
    # sample_data_train["y0_hat"] = cate_df_dict_train["tlearner"]["y0_hat"]
    # sample_data_train["y1_hat"] = cate_df_dict_train["tlearner"]["y1_hat"]

    policy_value_df_train = policy_est_obj.temp_bootstrap_policy_value(sample_data_train, cfg,
                                                                       policies_df_train.columns,
                                                                       propensity_score_name, treatment_name)
    # calc policy value box plots of all policies
    plot_boxplot(x="policy", y='policy_value', data=policy_value_df_train, hue="policy_value_estimator",
                 title="Stabilizied Policy Value - Train",
                 path_file=os.path.join(policy_path, "policy_value_boxplot_train.png"),
                 y_label=cfg.params.outcome_name)

    joblib.dump(policy_value_df_train, os.path.join(policy_path, "policy_value_df_train.pkl"))

    propensity_model_test = deepcopy(propensity_model)
    # refit on for conservative approach fit only on test - for reducing variance refit on train+test
    propensity_score_name_policy = propensity_score_name + "_test"
    test_data.loc[common_support_test_data_ids, propensity_score_name_policy] = \
        test_data.loc[common_support_test_data_ids, propensity_score_name]
    if cfg.params.test_propensity:
        log.info("Fit propensity on test for policy evaluation")  # TODO: ask Uri if re-fit prop only on common support?
        propensity_model_test.fit(pd.concat([train_data.loc[:, top_propensity_shap_features],
                                            test_data.loc[:, top_propensity_shap_features]]),
                                  pd.concat([train_data.loc[:, treatment_name],
                                            test_data.loc[:, treatment_name]]))
        log.info("Evaluating test propensity for offline policy evaluation")
        binary_clf_eval(model_name="Test Propensity Policy Offline Evaluation",
                        path=policy_path,
                        model=propensity_model_test,
                        data=test_data,
                        chosen_ftrs=top_propensity_shap_features,
                        outcome_name=treatment_name,
                        dataset="TEST")

        joblib.dump(propensity_model_test, os.path.join(policy_path, "propensity_model_test.pkl"))

        log.info("updating {} column: propensity score with new trained propensity on test".format(
            propensity_score_name_policy))
        test_data[propensity_score_name_policy] = \
            propensity_model_test.predict_proba(test_data[top_propensity_shap_features])[:, 1]

    sample_data_test = test_data.loc[common_support_test_data_ids, :].merge(policies_df_test,
                                                                            left_index=True, right_index=True)
    # add predicted y columns for policy value
    sample_data_test["y0_hat"] = \
        chosen_outcome_model_name_dict["y_hat_policy_0"]["GradientBoostingClassifier"].predict_proba(
            test_data.loc[common_support_test_data_ids, chosen_features_dict[0]])[:, 1]
    sample_data_test["y1_hat"] = \
        chosen_outcome_model_name_dict["y_hat_policy_1"]["GradientBoostingClassifier"].predict_proba(
            test_data.loc[common_support_test_data_ids, chosen_features_dict[1]])[:, 1]
    # sample_data_test["y0_hat"] = cate_df_dict_test["tlearner"]["y0_hat"]
    # sample_data_test["y1_hat"] = cate_df_dict_test["tlearner"]["y1_hat"]
    # calc policy value
    policy_value_df_test = policy_est_obj.temp_bootstrap_policy_value(sample_data_test, cfg, policies_df_test.columns,
                                                                      propensity_score_name_policy, treatment_name)
    # plot policy value of all policies
    plot_boxplot(x="policy", y='policy_value', data=policy_value_df_test, hue="policy_value_estimator",
                 title="Stabilizied Policy Value - Test",
                 path_file=os.path.join(policy_path, "policy_value_boxplot_test.png"),
                 y_label=cfg.params.outcome_name)
    joblib.dump(policy_value_df_test, os.path.join(policy_path, "policy_value_df_test.pkl"))

    # ----- Evaluate intervention policy - With KM curves
    chosen_policies = ["xlearner_policy", "current_policy"]  # TODO: make dynamic

    log.info("Policy Evaluation - IPW KM TRAIN")
    policy_est_obj.ipw_km(data=sample_data_train,
                          propensity_score_name=propensity_score_name,
                          treatment_name=treatment_name,
                          policy_names=policies_df_train.columns.intersection(chosen_policies),
                          policy_path=policy_path,
                          title="Flatiron NSCLC OS - TRAIN",
                          sipw=False)

    log.info("Policy Evaluation - IPW KM TEST")
    policy_est_obj.ipw_km(data=sample_data_test,
                          propensity_score_name=propensity_score_name_policy,
                          treatment_name=treatment_name,
                          policy_names=policies_df_test.columns.intersection(chosen_policies),
                          policy_path=policy_path,
                          title="Flatiron NSCLC OS - TEST",
                          sipw=False)

    log.info("Policy Explanation - regress on Policy decision TRAIN")

    sub_pop_analysis_obj = \
        SubPopulationAnalysis(data=sample_data_train,
                              outcome_name=policies_df_train.columns.intersection(chosen_policies),
                              treatment_name=treatment_name,
                              pop_description="Trimmed_train",
                              ftrs_for_analysis=top_outcome_shap_features,
                              analysis_path=policy_path,
                              need_scale=True if cfg.params.scaler_name is False else False,
                              scaler=False)  # TODO: clean this scaler mess
    scaler = sub_pop_analysis_obj.pop_feature_importance()

    log.info("Policy Explanation - regress on Policy decision TEST")
    sub_pop_analysis_obj.data = sample_data_test
    sub_pop_analysis_obj.pop_description = "Trimmed_test"
    sub_pop_analysis_obj.scaler = scaler
    sub_pop_analysis_obj.pop_feature_importance()


if __name__ == '__main__':

    # A logger for this file
    log = logging.getLogger(__name__)

    # define experiment config type
    cs = ConfigStore.instance()
    cs.store(name="cate_pipeline_config", node=CATEconfig)

    # run modeling pipeline
    run_cate_pipeline()
