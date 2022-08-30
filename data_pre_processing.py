import pandas as pd
import numpy as np
import joblib
import os
from tableone import TableOne
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import hydra
import logging
from hydra.core.config_store import ConfigStore
from collections import defaultdict
from utils import save_simulated_data, remove_quantiles
from utils_config import DataProcessConfig
from utils_ml import remove_corr_features


pd.set_option("display.max_rows", 10000)
pd.set_option("display.max_columns", 10000)
pd.set_option("display.max_colwidth", 10000)

# TODO: decompose func to class methods


@hydra.main(config_path="conf", config_name="data_process_config.yaml")
def data_pre_process(cfg: DataProcessConfig) -> None:
    """
    Function to insert all data pre process before CATE modeling pipeline
    """

    data_dict = joblib.load(os.path.join(cfg.paths.data, cfg.files.data_object))

    data = data_dict["df"]
    log.info("Running data pre process on data shape {} ".format(data.shape))

    # validate treatment is int
    data[data_dict["treatment_name"]] = data[data_dict["treatment_name"]].astype("int")
    treatment0_name, treatment1_name = sorted(data[data_dict["treatment_name"][0]].unique())

    data[data_dict["outcome_name"]] = data[data_dict["outcome_name"]].astype("int")

    # all candidate features
    confounder_names_list = \
        data_dict["common_causes_names"] + data_dict["effect_modifier_names"] + data_dict["instrument_names"]

    post_treatment_ftrs = []  # features that happened after index date (treatment time step hence will cause leakage)
    confounder_names_list = [x for x in confounder_names_list if x not in post_treatment_ftrs]

    # TODO: change column types logic, add validation of column types / human in the loop
    # get binary columns
    bool_cols = [col for col in confounder_names_list if np.isin(data[col].dropna().unique(), [0, 1]).all()]
    categorical_feature_names = data.select_dtypes(include=['object']).columns.difference(bool_cols)
    numeric_feature_names = data.select_dtypes(include=['number', 'category']).columns.difference(bool_cols)
    data[numeric_feature_names] = data[numeric_feature_names].apply(pd.to_numeric, errors='coerce')

    # Unify NULL
    data.replace({np.nan: None, "none": None}, inplace=True)

    # see if missingness patterns exist
    msno.matrix(data.iloc[:, : 50]).set_title("Missing patterns")
    plt.show()
    plt.close()
    msno.heatmap(data.iloc[:, : 50]).set_title("Missing correlations")
    plt.show()
    plt.close()

    # create missing values indicators
    log.info("adding missing indicator columns with prefix null_ind_")
    for col in confounder_names_list:
        if sum(data[col].isnull()) > 0:
            data["null_ind_{}".format(col)] = 0
            data.loc[data[col].isnull(), "null_ind_{}".format(col)] = 1
    bool_cols.extend([col for col in data.columns if 'null_ind_' in col])

    if cfg.params.train_test_split_method == 'temporal':

        temporal_train_index = data.query("{} < {}".format(data_dict["index_date"], cfg.params.min_test_year)).index
        temporal_test_index = data.query("{} >= {}".format(data_dict["index_date"], cfg.params.min_test_year)).index
        print(data[data_dict["index_date"]].value_counts(dropna=False))

        train_data = data.loc[data.index.isin(temporal_train_index), :]
        test_data = data.loc[data.index.isin(temporal_test_index), :]

        log.info("temporal train/test split according to {}: {}".format(data_dict["index_date"],
                                                                          cfg.params.min_test_year))
        log.info("temporal train: {}".format(train_data.shape))
        log.info("temporal test: {}".format(test_data.shape))

    elif cfg.params.train_test_split_method == "random":

        # split to train_data/ test
        train_data, test_data = \
            train_test_split(data, test_size=cfg.params.test_size, random_state=42)

        log.info("random train: {}".format(train_data.shape))
        log.info("random test: {}".format(test_data.shape))

    # remove highly correlated features
    corr_ftrs_to_drop = remove_corr_features(data=train_data.loc[:, train_data.columns != data_dict["treatment_name"][0]],
                                             corr_thresh=cfg.params.corr_thresh)
    log.info("removing corr features from the data: {}".format(corr_ftrs_to_drop))
    train_data.drop(columns=corr_ftrs_to_drop, axis=1, inplace=True)

    # remove rows with extreme values
    if cfg.params.outliers:
        for col in numeric_feature_names:
            train_data = remove_quantiles(df=train_data, value_col=col, min_quantile=cfg.params.outliers_range[0],
                                          max_quantile=cfg.params.outliers_range[1])

    # impute missing values
    train_impute_numeric_df = None
    if cfg.params.impute:
        print("Imputing missing data")
        # impute train
        log.info("Impute X train_data (binary with 0, numeric with mode)")
        # impute binary with 0
        train_data.loc[:, bool_cols] = train_data.loc[:, bool_cols].fillna(0.0)
        # impute numerical with the mode
        train_impute_numeric_df = train_data.loc[:, numeric_feature_names].mode().iloc[0]
        train_data.loc[:, numeric_feature_names] = \
            train_data.loc[:, numeric_feature_names].fillna(train_impute_numeric_df)

        log.info("----------------------Train data summary after imputation")
        mytable_train = TableOne(train_data)
        log.info(mytable_train.tabulate(tablefmt="github"))

        # impute test
        log.info("Impute X test data")
        # impute binary with 0
        test_data.loc[:, bool_cols] = test_data.loc[:, bool_cols].fillna(0.0)
        # impute numerical in test with the train mode
        test_data.loc[:, numeric_feature_names] = \
            test_data.loc[:, numeric_feature_names].fillna(train_impute_numeric_df)

    # code category variable
    cat_cols_mapping_dict = defaultdict(dict)
    if cfg.params.categorical == 'cat_codes':
        labelencoder = LabelEncoder()  # creating instance of labelencoder
        # Assigning numerical values
        for cat_col in categorical_feature_names:
            train_data.loc[:, cat_col] = labelencoder.fit_transform(train_data[cat_col].astype('category'))
            # get mapping dict
            dic = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
            # map test also undefined categories
            test_data.loc[:, cat_col] = test_data[cat_col].map(dic).fillna(-999)
            cat_cols_mapping_dict[cat_col] = labelencoder.get_params()

    elif cfg.params.categorical == 'dummy':
        # dummy code categorical
        log.info("train shape before dummies: {}".format(train_data.shape))
        train_data = pd.get_dummies(data=train_data, dummy_na=True, columns=categorical_feature_names)
        log.info("train shape after dummies: {}".format(train_data.shape))
        test_data = pd.get_dummies(data=test_data, dummy_na=True, columns=categorical_feature_names)

    # make dummy columns in test if does not exist
    train_cols_not_in_test = [x for x in train_data.columns if x not in test_data.columns]
    if train_cols_not_in_test:
        test_data = pd.concat([test_data, pd.DataFrame(columns=train_cols_not_in_test)], axis=1)
        test_data.loc[:, train_cols_not_in_test] = test_data.loc[:, train_cols_not_in_test].fillna(0.0)

    # take only train columns
    test_data = test_data[train_data.columns]

    # remove columns with one value
    for col in train_data.columns:
        if len(train_data[col].unique()) == 1:
            log.info("removing feature {} with single value".format(col))
            train_data.drop(col, inplace=True, axis=1)
            test_data.drop(col, inplace=True, axis=1)
    log.info("-------FINAL TRAIN SHAPE: {}".format(train_data.shape))
    log.info("-------FINAL TEST SHAPE: {}".format(test_data.shape))

    joblib.dump([train_data, test_data, data_dict["treatment_name"][0], treatment0_name, treatment1_name,
                 confounder_names_list, post_treatment_ftrs, train_impute_numeric_df, categorical_feature_names,
                 cat_cols_mapping_dict],
                "processed_data_obj_list.pkl")


if __name__ == '__main__':

    # A logger for this file
    log = logging.getLogger(__name__)

    # define experiment config type
    cs = ConfigStore.instance()
    cs.store(name="data_process_config", node=DataProcessConfig)

    save_simulated_data()  # prepare simulated data for debug
    data_pre_process()  # run pre process
