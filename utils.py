import random
from functools import reduce
from glob import glob
import os
import time
import boto3
from dowhy import datasets
import pandas as pd
import numpy as np
from numpy.random import binomial, multivariate_normal, normal, uniform
from scipy.stats import linregress
import datetime as dt
import joblib
from tableone import TableOne
from joblib import Parallel, delayed
import multiprocessing
import yaml
from yaml.loader import SafeLoader
import logging


# A logger for this file
log = logging.getLogger(__name__)


def save_simulated_data() -> None:
    """
    Imports/creates and save simulated data to data path
    """

    # Load sample data
    data_dict = datasets.linear_dataset(
        beta=10,
        num_common_causes=9,
        num_effect_modifiers=3,
        num_instruments=2,
        num_discrete_common_causes=3,
        num_discrete_instruments=1,
        num_discrete_effect_modifiers=1,
        num_samples=1000,
        treatment_is_binary=True,
        outcome_is_binary=True)

    # copy a column
    data_dict["df"]['W00'] = data_dict["df"]['W0']

    # add some random missingness between 0 to 10%
    for col in data_dict["df"].columns.difference([data_dict["treatment_name"][0], data_dict["outcome_name"][0]]):
        data_dict["df"].loc[data_dict["df"].sample(frac=round(random.uniform(0.0, 0.1), 2)).index, col] = pd.np.nan

    # add a dummy categorical feature
    data_dict["df"]['C0'] = pd.Series(random.choices(['A', 'B', 'C', 'D', 'E'],
                                                     weights=[1, 1, 1, 1, 1],
                                                     k=len(data_dict["df"])),
                                      index=data_dict["df"].index)
    data_dict["effect_modifier_names"].append('C0')
    # save data dictionary
    cur_path = create_path("data")
    joblib.dump(data_dict, os.path.join(cur_path, "simulated_data_dict.pkl"))


def load_yaml_to_dict(yaml_path):

    # Open the file and load the file
    with open(yaml_path) as f:
        yaml_dict = yaml.load(f, Loader=SafeLoader)
    return yaml_dict


def calc_weighted_table1(df, group: str, weights: str, numeric_columns: list, categorical_columns: list, path: str):
    """
    Weights features to see weighted feature distributions between groups
    :param df:
    :param group:
    :param weights:
    :param numeric_columns:
    :param categorical_columns:
    :param path:
    :return:
    """

    numeric_col_df = list()

    for ftr in numeric_columns:
        d = dict()
        weighted_ftr = ftr + '_weight'
        df[weighted_ftr] = df[ftr] * df[weights]
        # overall
        ftr_w_mean = (df[weighted_ftr].sum() / df[weights].sum())
        ftr_w_std = df[weighted_ftr].std()
        d["Overall"] = (ftr_w_mean, ftr_w_std)
        # per group
        for g in df[group].unique():
            ftr_w_mean = (df.query("{} == '{}'".format(group, g))[weighted_ftr].sum() /
                          df.query("{} == '{}'".format(group, g))[weights].sum())
            ftr_w_std = df.query("{} == '{}'".format(group, g))[weighted_ftr].std()
            d[g] = (ftr_w_mean, ftr_w_std)
        ftr_df = pd.DataFrame(data=d, index=[ftr+'_mean', ftr+'_std'])
        numeric_col_df.append(ftr_df)

    categorical_col_df = list()
    # calc weighted % of categorical values
    for ftr in categorical_columns:
        d = dict()
        for value in df[ftr].unique():
            ftr_value_prc = (df.loc[df[ftr] == value, weights].sum() / df[weights].sum())
            d["Overall"] = ftr_value_prc
            for g in df[group].unique():
                ftr_value_prc = (
                            df.query("{} == '{}'".format(group, g)).loc[df[ftr] == value, weights].sum() /
                            df.query("{} == '{}'".format(group, g))[weights].sum())
                d[g] = ftr_value_prc
                ftr_df = pd.DataFrame(data=d, index=[(ftr + ' ' +str(value))])
                categorical_col_df.append(ftr_df)

    numeric_df = pd.concat(numeric_col_df)
    numeric_df.to_csv(os.path.join(path, "weighted_numeric_ag_set.csv"))

    categorical_df = pd.concat(categorical_col_df)
    categorical_df.to_csv(os.path.join(path, "weighted_categorical_ag_set.csv"))

    return


def apply_parallel(df_grouped, func):
    """
    apply func to groups in df_grouped in parallel
    :param df_grouped:
    :param func:
    :return:
    """

    result_list = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in df_grouped)
    result_df = pd.DataFrame(index=df_grouped.indices.keys(), data=result_list)
    return result_df


def load(dir, key, df_name):
    df = pd.read_csv(dir)
    log.info("{} {} rows with {} unique patients".format(df_name, df.shape, df[key].nunique()))
    df.set_index(keys=key, inplace=True)
    return df


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def create_path(folder: str) -> str:
    """
    creates folder from working directory if not exists
    :param folder: folder name (can be a directory that ends with the folder name
    :return: full path to created folder
    """

    cur_dir = os.getcwd()
    path = os.path.join(cur_dir, folder)
    os.makedirs(path, exist_ok=True)

    return path


def generate_data(n, d, controls_outcome, treatment_effect, propensity):
    """Generates population data for given untreated_outcome, treatment_effect and propensity functions.

    Parameters
    ----------
        n (int): population size
        d (int): number of covariates
        controls_outcome (func): untreated outcome conditional on covariates
        treatment_effect (func): treatment effect conditional on covariates
        propensity (func): probability of treatment conditional on covariates
    """
    # Generate covariates
    X = multivariate_normal(np.zeros(d), np.diag(np.ones(d)), n)
    # Generate treatment
    T = np.apply_along_axis(lambda x: binomial(1, propensity(x), 1)[0], 1, X)
    # Calculate outcome
    Y0 = np.apply_along_axis(lambda x: controls_outcome(x), 1, X)
    treat_effect = np.apply_along_axis(lambda x: treatment_effect(x), 1, X)
    Y = Y0 + treat_effect * T
    return Y, T, X


# DGP constants and test data
# controls outcome, treatment effect, propensity definitions
def generate_controls_outcome(d):
    beta = uniform(-3, 3, d)
    return lambda x: np.dot(x, beta) + normal(0, 1)


def load_s3_data(bucket: str, key: str):

    client = boto3.client('s3')  # low-level functional API

    # resource = boto3.resource('s3')  # high-level object-oriented API
    # my_bucket = resource.Bucket('my-bucket')  # subsitute this for your s3 bucket name.

    obj = client.get_object(Bucket=bucket, Key=key)
    data = pd.read_csv(obj['Body'])

    return data


def read_all_csv(dir_name: str, ext: str, key: str, chosen_file_names: list = None):
    """
    import all csv in dir and subdir and joins to Pandas csv on key
    :param dir_name:
    :param ext:
    :param key:
    :param chosen_file_names: if exist reads only these files
    :return:
    """

    csv_filenames = [file for _path, subdir, files in os.walk(dir_name) for file in glob(os.path.join(_path, ext))]
    dfs = []
    for filename in csv_filenames:
        if (chosen_file_names is not None) & (filename in chosen_file_names):
            log.info("reading {}".format(filename))
            df = pd.read_csv(filename)
            log.info("df shape {} with nunique key {}".format(df.shape, df[key].nunique()))
            dfs.append(df)

    big_frame = reduce(lambda left, right: pd.merge(left, right, on=key, how='outer'), dfs)

    return big_frame


def to_str_lower(df, col_list):

    for col in df.columns.intersection(col_list):
        df[col] = df[col].astype(str)
        df[col] = df[col].str.lower()
    return df


def to_numeric(df, col_list):

    for col in col_list:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception as e:
            log.info("column {} can't be converted to numeric - check values:".format(col))
            log.info(df.loc[
                pd.to_numeric(df[col], errors='coerce').isnull(), col].unique())
    return df


def to_datetime(df, col_list):

    for col in col_list:
        if col:
            df[col] = pd.to_datetime(df[col])
    return df


def remove_nulls(df, col_list):

    for col in col_list:
        log.info("removing {} rows with null value in {}".format(df.loc[df[col].isnull(), :].shape, col))
        df = df.loc[~df[col].isnull(), :]
        df = df.loc[df[col] != 'Null', :]

    return df


def set_capped_result(df, col_list):
    """
    replaces string test value of bigger/smaller sign to the limit value itself +/- small number accordingly
    """
    # TODO: add test if both signs exist in string, validate small reduction/addition is valid in range,
    #  check value location in splited list to make sure it is not first meaning opposite direction of sign
    for col in col_list:
        # fix smaller than test result
        df.loc[df[col].str.contains('<', na=False), col] = \
            df.loc[df[col].str.contains('<', na=False), col].str.split('<').apply(
            lambda x: str(pd.to_numeric(x[1]) - 0.00001)) # 5657224 <0.01, 5676818 2.85, 5642761 >75, 5676887 2.3

        # fix bigger than test result
        df.loc[df[col].str.contains('>', na=False), col] = \
            df.loc[df[col].str.contains('>', na=False), col].str.split('>').apply(
            lambda x: str(pd.to_numeric(x[1]) + 0.00001))

    return df


def strip_char_from_result(df, col_list, char_list):
    """
    strip char_list from string test value in col_list
    """

    for col in col_list:

        df[col] = df[col].str.replace('|'.join(char_list), '')

    return df


def get_days_months_duration_cols(df, prefix, start_date_col, end_date_col):

    df.loc[:, "{}_DAYS".format(prefix)] = \
        (df[end_date_col] -
         df[start_date_col]).dt.days

    df.loc[:, "{}_MONTHS".format(prefix)] = ((df[end_date_col] - df[start_date_col]) / np.timedelta64(1, 'M'))
    df["{}_MONTHS".format(prefix)] = df["{}_MONTHS".format(prefix)].astype(int, errors='ignore')

    log.info(" {} - {} distribution (M)".format(end_date_col, start_date_col))
    log.info(df["{}_MONTHS".format(prefix)].describe(np.arange(0, 1.1, 0.1)))
    return df


def calc_mean_value_per_timeframe(df: pd.DataFrame, id_col: str, test_col: str, test_value: str, time_col: str,
                                  timeframe_start: int = 0, timeframe_end: int = 24, time_step: int = 4):

    # get average value per id, test_col and timeframe
    timeframe_mean_df = df.groupby([id_col, test_col, pd.cut(df[time_col],
                                                             np.arange(timeframe_start, timeframe_end + 1,
                                                                       time_step))])[test_value].mean().reset_index()
    # validate columns are string for names
    timeframe_mean_df[test_col] = timeframe_mean_df[test_col].apply(lambda x: str(x))
    timeframe_mean_df[time_col] = timeframe_mean_df[time_col].apply(lambda x: str(x))
    # pivot table
    timeframe_mean_df = timeframe_mean_df.reset_index().pivot_table(values=test_value, index=id_col,
                                                                    columns=[test_col, time_col])
    # flatten column
    timeframe_mean_df.columns = timeframe_mean_df.columns.map('_'.join).str.strip('_')

    return timeframe_mean_df


def diagnosis_group_indicator_col(df, df_diag_col, diag_dict):

    diagnosis_group_indicator_col_list = list()
    for diag_group_name, diag_list in diag_dict.items():
        df.loc[:, diag_group_name] = 0
        df.loc[df[df_diag_col].isin(diag_list), diag_group_name] = 1
        diagnosis_group_indicator_col_list.append(diag_group_name)

    return df, diagnosis_group_indicator_col_list


def convert_range(old_value, old_min, old_max, new_min, new_max):
    "changes old value to new value from old min/max to new min/max"
    old_range = (old_max - old_min)
    if old_range == 0:
        new_value = new_min
    else:
        new_range = (new_max - new_min)
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min
    return new_value


def remove_sub_list_from_string_list(sub_list: list, string_list: list):
    """
    removes sub strings fro, list of strings, if exist otherwise keeps string
    :param sub_list: sub strings to remove
    :param string_list: strings to clean
    :return:
    """
    # method to כfind substring in string from lst
    def substr_in_list(string, lst):
        for sub in lst:
            if sub in string:
                return sub
        return False

    string_list_clean = list()

    for x in string_list:
        log.info(x)
        sub = substr_in_list(x, sub_list)
        log.info(sub)
        if sub:
            string_list_clean.append(x.replace(sub, ""))
        else:
            string_list_clean.append(x)

    log.info(len(string_list_clean))
    string_list_clean.sort()
    log.info(string_list_clean)

    return string_list_clean


def remove_quantiles(df: pd.DataFrame, value_col: str,
                     min_quantile: float = 0.05, max_quantile: float = 0.95) -> pd.DataFrame:
    """
    keeps rows where value_col is in defined quantile range
    :param df:
    :param value_col:
    :param min_quantile:
    :param max_quantile:
    :return:
    """
    min_quantile_value = df[value_col].quantile(min_quantile)
    max_quantile_value = df[value_col].quantile(max_quantile)

    return df[(df[value_col] >= min_quantile_value) & (df[value_col] <= max_quantile_value)]


def calc_slope_per_test_date(df, group_col_list: list, time_col: str, value_col: str):
    """
    calculates slope of time series per group
    :param df:
    :param group_col_list:
    :param time_col: x axis for regression
    :param value_col: y axis for regression
    :return:
    """

    log.info("calculating slopes")
    # map date to ordinal values (days from 1/1/1)
    df[time_col+'_ordinal'] = df[time_col].map(dt.datetime.toordinal)
    # subtract days of 1970 years, x axis will be normal
    # df[time_col + '_ordinal'] = df[time_col+'_ordinal'] - 365*1970 # TODO: validate
    log.info("start")

    group_value_col_slope_df = \
        df.groupby(group_col_list).apply(lambda v: linregress(v[time_col+'_ordinal'],
                                                              v[value_col])[0]).round(4).to_frame().reset_index()

    log.info("finish")
    group_value_col_slope_df.rename({0: "slope"}, axis=1, inplace=True)
    # remove null values - probably had only 1 test at baseline
    group_value_col_slope_df = group_value_col_slope_df.loc[~group_value_col_slope_df["slope"].isnull()]

    df.drop(time_col + '_ordinal', inplace=True, axis=1)

    return group_value_col_slope_df


def unite_columns_to_one_columns(df, col_list, new_col_name, agg_func):

    # df[new_col_name] = df[col_list].transform(agg_func) # TODO: validate
    existing_cols = df.columns.intersection(col_list)
    log.info("out of {}, existing columns are: {}".format(col_list, existing_cols))
    if agg_func == 'max':
        df[new_col_name] = df[existing_cols].max(axis=1)
    else:
        log.info("agg_func not supported")
    return df, existing_cols


def features_interactions(df: pd.DataFrame, features_interactions_dict: dict):

    for interaction_type, features_list in features_interactions_dict.items():

        if interaction_type == "ratio":
            for features_tuple in features_list:
                df["{}_{}_ratio".format(features_tuple[0], features_tuple[1])] = \
                    df[features_tuple[0]] / df[features_tuple[1]]
        elif interaction_type == "multiply":
            for features_tuple in features_list:
                df["{}_{}_multiply".format(features_tuple[0], features_tuple[1])] = \
                    df[features_tuple[0]] * df[features_tuple[1]]
        else:
            log.info("interaction_type {} not supported")
    return df


def exclude_by_date_col(data, end_date, start_date_col_name, thresh_months):

    data = data.loc[((pd.Timestamp(end_date) - data[start_date_col_name]).dt.days / 30) >= thresh_months, :]
    log.info("data shape after exclusion less than follow up {}: {}".format(thresh_months, data.shape))
    return data
