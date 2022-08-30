import os
import pandas as pd
import numpy as np
from scipy.special import expit
from scipy import stats
from sklearn.calibration import _sigmoid_calibration
from sklearn.ensemble import GradientBoostingClassifier, \
    RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier
import lightgbm as lgbm
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score, roc_auc_score, make_scorer, \
    average_precision_score, accuracy_score, log_loss
from sklearn.utils import resample
from econml.dml import NonParamDML
from econml.metalearners import XLearner
# from skfeature.function.information_theoretical_based.CMIM import cmim
from boruta import BorutaPy
from BorutaShap import BorutaShap
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from causalnex.structure.notears import from_pandas
import logging
import joblib
from builtins import any as b_any
from sklearn.linear_model import LogisticRegression, LinearRegression
from yellowbrick.classifier import DiscriminationThreshold
from utils import time_print


# A logger for this file
log = logging.getLogger(__name__)


def ks_test(sample1, sample2, alternative: str = "two-sided"):

    """Performs the two-sample Kolmogorov-Smirnov test for goodness of fit"""
    print("Kolmogorov-Smirnov test {}".format(alternative))
    result = stats.ks_2samp(data1=sample1, data2=sample2, alternative=alternative)
    print("statistic: {}".format(round(result[0], 4)))
    print("pvalue: {}".format(round(result[1], 4)))
    return result


def t_test(sample1, sample2, alternative: str = "two-sided", permutations=None):
    print("T-test")
    # rng = np.random.default_rng()
    result = stats.ttest_ind(sample1, sample2)  #, permutations=10000, random_state=rng)
    print("statistic: {}".format(result[0].round(4)))
    print("pvalue: {}".format(result[1].round(4)))
    return result


def feature_selection(x, y, col_names_list: list, method: str, arg_dict: dict = {"max_depth": 5}, alpha=0.05, perc=100,
                      n_selected_features=30,
                      outcome="binary"):

    if method == "boruta":
        time_print("Starting boruta feature selection on data shape: {}".format(x.shape))

        # define random forest classifier, with utilising all cores and
        # sampling in proportion to y labels
        if outcome == "binary":
            ml = RandomForestClassifier(n_jobs=-1, random_state=42, **arg_dict)# max_depth=4, n_estimators=100, criterion='gini')
        elif outcome == "multiclass":
            ml = RandomForestClassifier(n_jobs=-1, max_depth=5, n_estimators=200,
                                        random_state=42, criterion='entropy')  # objective="multi:softmax",
        elif outcome == "reg":
            ml = RandomForestRegressor(n_jobs=-1, max_depth=5, n_estimators=200, random_state=42)

        # define Boruta feature selection method
        feat_selector = BorutaPy(ml, n_estimators='auto', verbose=2, random_state=1, max_iter=100, alpha=alpha,
                                 perc=perc)

        # find all relevant features
        feat_selector.fit(x.values, y.values.astype(int))
        log.info("Finish boruta feature selection on data shape: {}".format(x.shape))

        green_area = x.columns[feat_selector.support_].to_list()
        blue_area = x.columns[feat_selector.support_weak_].to_list()
        log.info('Boruta features in the green area: {}'.format(green_area))
        log.info('Boruta features in the blue area:: {}'.format(blue_area))

        # combine Boruta's green and blue areas,
        # see https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a
        green_area.extend(blue_area)
        n_selected_features = len(green_area)
        log.info("Boruta {} chosen features are {}".format(n_selected_features, green_area))

        # if n_selected_features:
        #     log.info("Taking only top {} features".format(n_selected_features))
        #     green_area = green_area[:n_selected_features]

        return green_area

    elif method == "cmim":

        raise ValueError("Currently not supported: awaitinig installation")
        # log.info("Starting cmim feature selection on data shape: {}".format(x.shape))
        # selected_features_index = cmim(X=x.values, y=y.values, n_selected_features=n_selected_features, mode="index")
        # log.info("Finish cmim feature selection on data shape: {}".format(x.shape))
        # # TODO: validate feature names by index are correct
        # return list(np.array(col_names_list)[selected_features_index])

    elif method == "tree":
        selector = SelectFromModel(estimator=GradientBoostingClassifier(random_state=42, **arg_dict), threshold=-np.inf,
                                   max_features=n_selected_features).fit(x.values, y.values)

        selected_features = list(np.array(col_names_list)[selector.get_support()])
        log.info("XGB Select from model {} chosen features are {}".format(n_selected_features, selected_features))

        return selected_features

    elif method == "borutashap":

        log.info("Starting borutashap feature selection on data shape: {}".format(x.shape))

        if outcome == "binary":
            # If no model selected default is Random Forest,
            selector = BorutaShap(
                model=GradientBoostingClassifier(max_depth=4, n_estimators=50, random_state=42,
                                                 validation_fraction=0.2, n_iter_no_change=10, tol=0.001),
                importance_measure='shap', classification=True, percentile=perc, pvalue=alpha)
        elif outcome == "reg":
            selector = BorutaShap(model=GradientBoostingRegressor(max_depth=4, n_estimators=50, random_state=42,
                                                                  validation_fraction=0.2, n_iter_no_change=10,
                                                                  tol=0.001),
                                  importance_measure='shap', classification=False, percentile=perc, pvalue=alpha)
        else:
            raise ValueError("outcome {} is not supported".format(outcome))

        # Fits the selector
        selector.fit(X=x, y=y, n_trials=100, train_or_test='test', sample=False, normalize=True, verbose=True)  # train_or_test='test'

        # Returns Boxplot of features
        # selector.plot(which_features='all')
        boruta_shap_list = list(set(selector.accepted).union(selector.tentative))

        log.info("Borutashap chosen features are {}".format(boruta_shap_list))

        return boruta_shap_list

    else:
        raise ValueError("method {} is not supported".format(method))


def remove_corr_features(data: pd.DataFrame, corr_thresh=0.95):

    # remove highly Pearson correlated features
    time_print("Start with data shape {}".format(data.shape))
    data_corr_matrix = data.corr().abs()
    upper_tri = data_corr_matrix.where(np.triu(np.ones(data_corr_matrix.shape), k=1).astype(bool))
    corr_ftrs_to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_thresh)]
    time_print("{} highly correlated features: {}".format(len(corr_ftrs_to_drop), corr_ftrs_to_drop))
    return corr_ftrs_to_drop


def evaluate_clf(y, y_hat, y_score, print_scores=True):
    """
    evaluations for binary classifier
    :param y: labels
    :param y_hat: class predictions
    :param y_score: class prediction probabilities
    :return: dictionary with all evaluation metrics
    """

    eval_dict = dict()
    eval_dict["acc"] = round(accuracy_score(y, y_hat), 2)
    eval_dict["auc"] = round(roc_auc_score(y, y_score), 2)
    eval_dict["brier"] = round(brier_score_loss(y, y_score, pos_label=y.max()), 2)
    eval_dict["prc"] = round(precision_score(y, y_hat), 2)
    eval_dict["rcl"] = round(recall_score(y, y_hat), 2)
    eval_dict["f1"] = round(f1_score(y, y_hat), 2)

    if print_scores:
        for key, value in eval_dict.items():
            log.info("{}: {}".format(key, value))

    return eval_dict


def binary_clf_eval(model, data, chosen_ftrs, outcome_name, model_name, dataset, path):
    # TODO: add precision recall curve
    log.info("evaluating binary classifier {} on {} with shape {}".format(model_name, dataset, data[chosen_ftrs].shape))

    y_hat = model.predict(X=data[chosen_ftrs])
    y_score = model.predict_proba(X=data[chosen_ftrs])[:, 1]
    eval_dict = evaluate_clf(y=data[outcome_name].values, y_hat=y_hat, y_score=y_score)
    y_score_treatment_df = pd.DataFrame(data=np.column_stack((y_score, data[outcome_name].values)),
                                        columns=["y_score", outcome_name])
    sns.histplot(data=y_score_treatment_df,
                 x="y_score", hue=outcome_name, stat="percent").set_title("{} {} {}".format("Histogram", model_name, dataset))
    # plt.show()
    plt.tight_layout()
    plt.ioff()
    plt.savefig(os.path.join(path,
                             "{} {} {}".format("Histogram", model_name, dataset)+".png"))#, bbox_inches='tight', pad_inches=1)
    # plt.show()
    plt.close()
    sns.ecdfplot(data=y_score_treatment_df, x="y_score", hue=outcome_name,
                 stat="proportion").set_title("{} {} {}".format("CDF", model_name, dataset))
    # plt.show()
    plt.tight_layout()
    plt.ioff()
    plt.savefig(os.path.join(path, "{} {} {}".format("CDF", model_name, dataset)+".png"))#, bbox_inches='tight', pad_inches=1)
    plt.close()
    # plot_calibration_curve(clf_dict={model_name: model},
    #                        X=data[chosen_ftrs],
    #                        y=data[outcome_name],
    #                        title=dataset,
    #                        path=path)
    return eval_dict


def nested_k_fold(X, y, model, score_name, score, greater_is_better, model_name, space, outer_folds_num=8,
                  inner_folds_num=5, upsample=False, nested_step: bool = False, fit_params: dict = None):
    """
    Nested CV - to reduce bias of model selection and hyper-parameter GRID SEARCH
    :param X:
    :param y:
    :param model: model object
    :param score_name: string of scoring metric
    :param score: score_name metric object
    :param greater_is_better: defines if score is cost or reward
    :param model_name: model type name
    :param space: model specific hyper parameters search space dictionary
    :param outer_folds_num: number of train splits to train/test for stratified KFold model selection
    :param inner_folds_num: number of outer's train fold splits for gridsearch KFold hyper-parameter optimization
    :param upsample: up sample minority class (assuming to be Y=1)
    :return:
    """
    # TODO: add Mcnemar test and Bonferroni

    outer_cv_results_df = pd.DataFrame(columns=["params", "score_name", "score", "model_name", "model", "dataset"])
    outer_cv_test_scores_list = list()
    cv_outer = StratifiedKFold(n_splits=outer_folds_num, shuffle=True, random_state=42)

    if nested_step:
        for outer_cv_idx, (train_ix, test_ix) in enumerate(cv_outer.split(X=X, y=y), 1):
            log.info("running {} iteration of nested cv outer loop".format(outer_cv_idx))
            # split data
            X_train, X_test = X.values[train_ix, :], X.values[test_ix, :]
            y_train, y_test = y.values[train_ix], y.values[test_ix]

            if upsample:
                "balance outcome classes in train set of current fold"
                minority_df = pd.DataFrame(data=X_train)
                minority_df["y_train"] = y_train
                # get minority y value
                y_min_val = minority_df["y_train"].value_counts(ascending=True).index[0]
                # set other classes to another dataframe
                majority_df = minority_df.loc[minority_df["y_train"] != y_min_val]
                # set the minority class to a seperate dataframe
                minority_df = minority_df.loc[minority_df["y_train"] == y_min_val]
                # upsample the minority class
                minority_df_upsampled = resample(minority_df,
                                                 random_state=42,
                                                 n_samples=int((majority_df.shape[0]/minority_df.shape[0])
                                                               *minority_df.shape[0]), replace=True)
                # concatenate the upsampled dataframe
                upsampled_df = pd.concat([minority_df_upsampled, majority_df])
                y_train = upsampled_df["y_train"].values
                upsampled_df.drop("y_train", inplace=True, axis=1)
                X_train = upsampled_df.values

            # configure the cross-validation procedure
            cv_inner = KFold(n_splits=inner_folds_num, shuffle=True, random_state=42)

            # define search
            search = GridSearchCV(model, space, scoring=make_scorer(score, greater_is_better=greater_is_better),
                                  cv=cv_inner, refit=True, n_jobs=-1)
            # execute search
            log.info("Starting grid search hyper parameters optimization with score: {}".format(score_name))
            # check if need to pass fit_params
            if fit_params:
                result = search.fit(X_train, y_train, **fit_params)
            else:
                result = search.fit(X_train, y_train)
            # get the best performing model fit on the whole training set
            best_model = result.best_estimator_

            # evaluate model on the hold out dataset
            # if binary_outcome:
            y_score = best_model.predict_proba(X=X_test)[:, 1]
            y_pred = best_model.predict(X=X_test)
            score_val = round(score(y_test, y_score), 2)  # TODO: change assumption that score gets predict_proba

            # else:
            #     y_score = best_model.predict_proba(X=X_test)
            #     score_val = round(score(y_test, y_score, multi_class="ovr"), 2)

            outer_cv_test_scores_dict = evaluate_clf(y=y_test, y_hat=y_pred, y_score=y_score, print_scores=False)
            outer_cv_test_scores_list.append(outer_cv_test_scores_dict)

            # store the result
            res_df = pd.DataFrame(data=[[result.best_params_, score_name, score_val, model_name, best_model]],
                                  columns=["params", "score_name", "score", "model_name", "model"])
            outer_cv_results_df = outer_cv_results_df.append(res_df, ignore_index=True)

            # report progress
            log.info('outer CV fold test {}={}, best inner CV mean score {}={}, cfg={}'.format(score_name, score_val,
                                                                                                 score_name,
                                                                             round(result.best_score_, 2),
                                                                             result.best_params_))
        # summarize the estimated performance of the model by the outer cv
        log.info('Performance on outer CV')
        log.info('{}: mean: {}, std: {}'.format(score_name, round(outer_cv_results_df.loc[:, "score"].mean(), 2),
                                                              round(outer_cv_results_df.loc[:, "score"].std(), 2)))
        # Merge outer cv performance dicts
        total_outer_cv_test_scores_dict = dict()
        for sub in outer_cv_test_scores_list:
            for key, val in sub.items():
                total_outer_cv_test_scores_dict.setdefault(key, []).append(val)
        outer_cv_performance_df = pd.DataFrame.from_dict(total_outer_cv_test_scores_dict).mean()
        log.info("outer CV {} results:".format(model_name))
        log.info(outer_cv_performance_df)
        log.info("Observe if performance is stable across outer CV folds and best params are stable? "
                   "if so choose final model:")

    # run final cv with grid search on entire train
    # configure the cross-validation procedure
    log.info("Running Final CV with {} folds on all train data and refit for {} parameter + model "
               "selection".format(inner_folds_num, model_name))
    final_cv_train = StratifiedKFold(n_splits=inner_folds_num, shuffle=True, random_state=42)
    # define search
    final_search = GridSearchCV(model, space, scoring={"auc": 'roc_auc',
                                                       "prc_rcl_curve": 'average_precision',
                                                       "log_loss": 'neg_log_loss',
                                                        "brier": 'neg_brier_score',
                                                        "accuracy": 'accuracy',
                                                        "precision": 'precision',
                                                        "recall": 'recall',
                                                        "F1": 'f1',
                                                        },
                                cv=final_cv_train, refit=score_name,
                                n_jobs=-1, return_train_score=True)
    # execute search
    final_result = final_search.fit(X, y)

    # get the best performing model on the whole training set
    final_model = final_result.best_estimator_

    res_df_cv = pd.DataFrame(data=[[final_result.best_params_, score_name, final_result.best_score_, model_name +
                                    '_final', final_model, "CV_val"]],
                             columns=["params", "score_name", "score", "model_name", "model", "dataset"])
    outer_cv_results_df = outer_cv_results_df.append(res_df_cv, ignore_index=True)
    res_df_train = pd.DataFrame(data=[[final_result.best_params_, score_name,
                                final_result.cv_results_["mean_train_{}".format(score_name)][final_result.best_index_],
                                       model_name + '_final', final_model, "CV_Train"]],
                                columns=["params", "score_name", "score", "model_name", "model", "dataset"])
    outer_cv_results_df = outer_cv_results_df.append(res_df_train, ignore_index=True)

    log.info("Final grid search hyper parameters optimization for model {} with all train mean cv {} score "
             "is: {}".format(model_name, score_name, round(final_result.best_score_, 2)))
    # cv performance
    cv_performance_dict = {
                            "acc": round(final_result.cv_results_["mean_test_accuracy"][final_result.best_index_], 2),
                            "auc": round(final_result.cv_results_["mean_test_auc"][final_result.best_index_], 2),
                            "brier": round(final_result.cv_results_["mean_test_brier"][final_result.best_index_], 2),
                            "prc": round(final_result.cv_results_["mean_test_precision"][final_result.best_index_], 2),
                            "rcl": round(final_result.cv_results_["mean_test_recall"][final_result.best_index_], 2),
                            "f1": round(final_result.cv_results_["mean_test_F1"][final_result.best_index_], 2)
    }
    outer_cv_performance_df = pd.DataFrame(cv_performance_dict, index=[0])
    log.info("CV performance for final {}".format(model_name))
    log.info(outer_cv_performance_df)

    log.info("Final model params are: {}".format(final_result.best_params_))
    return final_model, outer_cv_results_df, outer_cv_performance_df


def k_fold(X, y, model, space, score_name, score, greater_is_better, model_name, folds_num=5):

    """
     CV
    :param X:
    :param y:
    :param model: model object
    :param score_name: string of scoring metric
    :param score: score_name metric object
    :param model_name: model type name
    :param folds_num: number of train splits to train/test for stratified KFold model selection
    :return:
    """

    cv = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=42)
    cv_results_df = pd.DataFrame(columns=["score_name", "score", "model_name", "model"])
    cv_test_scores_list = list()

    for cv_idx, (train_ix, test_ix) in enumerate(cv.split(X=X, y=y), 1):

        time_print("running {} iteration of cv loop".format(cv_idx))

        # split data
        X_train, X_test = X.values[train_ix, :], X.values[test_ix, :]
        y_train, y_test = y.values[train_ix], y.values[test_ix]

        model.fit(X_train, y_train)

        # evaluate model on the hold out
        y_score = model.predict_proba(X=X_test)[:, 1]
        y_pred = model.predict(X=X_test)
        score_val = round(score(y_test, y_score), 2)

        outer_cv_test_scores_dict = evaluate_clf(y=y_test, y_hat=y_pred, y_score=y_score, print_scores=False)
        cv_test_scores_list.append(outer_cv_test_scores_dict)

        # store the result
        res_df = pd.DataFrame(data=[[score_name, score_val, model_name, model]],
                              columns=["score_name", "score", "model_name", "model"])
        cv_results_df = cv_results_df.append(res_df, ignore_index=True)

    # summarize the estimated performance of the model by the outer cv
    time_print('Performance on outer CV')
    time_print('{}: mean: {}, std: {}'.format(score_name, round(cv_results_df.loc[:, "score"].mean(), 2),
                                                          round(cv_results_df.loc[:, "score"].std(), 2)))
    # Merge outer cv performance dicts
    cv_test_scores_dict = dict()
    for sub in cv_test_scores_list:
        for key, val in sub.items():
            cv_test_scores_dict.setdefault(key, []).append(val)
    cv_performance_df = pd.DataFrame.from_dict(cv_test_scores_dict).mean()
    time_print("outer CV {} results:".format(model_name))
    print(cv_performance_df)
    time_print("Observe if performance is stable across outer CV folds and best params are stable? "
               "if so choose final model:")

    # fit on all train
    model.fit(X, y)

    return model, cv_results_df, cv_performance_df


def run_causalnex(discover_data, file_name, label, color_nodes_name_dict, default_plot=False):
    # TODO: CLEAN
    time_print("Starting causal graph discovery with causalnex on data shape: {}".format(discover_data.shape))
    # causalnex graph
    sm = from_pandas(discover_data, max_iter=100000)
    time_print("Finish causal graph discovery with causalnex on data shape: {}".format(discover_data.shape))
    sm.remove_edges_below_threshold(0.5)

    graph_attributes = {
        "splines": "spline",  # I use splies so that we have no overlap
        "ordering": "out",
        "ratio": "fill",  # This is necessary to control the size of the image
        "size": "16,9!",  # Set the size of the final image. (this is a typical presentation size)
        "label": label,
        "fontcolor": "#FFFFFFD9",
        "fontname": "Helvetica",
        "fontsize": 45,
        "labeljust": "l",
        "labelloc": "t",
        "pad": "2,2",
        "dpi": 200,
        "nodesep": 0.8,
        "ranksep": ".5 equally",
    }
    # graph_attributes["nodesep"] = 2
    # graph_attributes["ranksep"] = "1.1 equally"
    # Making all nodes hexagonal with black coloring
    node_attributes = {
        node: {
            "shape": "hexagon",
            "width": 2.2,
            "height": 2,
            "fillcolor": "#000000",
            "penwidth": "10",
            "color": "#4a90e2d9",
            "fontsize": 45,
            "labelloc": "c",
        }
        for node in sm.nodes
    }

    # Target nodes are colored differently
    for node in sm.nodes:
        if b_any(node in x for x in color_nodes_name_dict.keys()):
            node_attributes[node]["fillcolor"] = color_nodes_name_dict[node]

    min_w = 0
    max_w = 0
    for u, v, w in sm.edges(data="weight"):
        if w < min_w:
            min_w = w
        if w > max_w:
            max_w = w

    def scale_w(w, min_w, max_w):
        return (w-min_w)/(max_w-min_w)

    # Customising edges
    edge_attributes = {
        (u, v): {
            "penwidth": scale_w(w, min_w=min_w, max_w=max_w) * 20 + 2,  # Setting edge thickness
            "weight": int(5 * scale_w(w, min_w=min_w, max_w=max_w)),  # Higher "weight"s mean shorter edges
            "arrowsize": 2 - 2.0 * scale_w(w, min_w=min_w, max_w=max_w),  # Avoid too large arrows
            "arrowtail": "dot",
        }
        for u, v, w in sm.edges(data="weight")
    }

    # if default_plot:
    #     viz = plot_structure(
    #         sm,
    #         graph_attributes={"scale": "0.5"},
    #         all_node_attributes=NODE_STYLE.WEAK,
    #         all_edge_attributes=EDGE_STYLE.WEAK)
    #     Image(viz.draw(format='png'))
    # else:
    #     viz = plot_structure(
    #         sm,
    #         prog="dot",
    #         graph_attributes=graph_attributes,
    #         node_attributes=node_attributes,
    #         edge_attributes=edge_attributes
    #         )

    # joblib.dump(Image(viz.draw(format='jpg',  prog="circo")), os.path.join(file_name, "causal_graph_image.pkl"))
    joblib.dump(sm, os.path.join(file_name, "causal_graph_obj.pkl"))
    joblib.dump(graph_attributes, os.path.join(file_name, "graph_attributes.pkl"))
    joblib.dump(node_attributes, os.path.join(file_name, "node_attributes.pkl"))
    joblib.dump(edge_attributes, os.path.join(file_name, "edge_attributes.pkl"))

    # cdt graph # TODO: install R dependencies
    # glasso = cdt.independence.graph.Glasso()
    # skeleton = glasso.predict(discover_data)
    # print(skeleton)
    # remove indirect links in the graph using the Aracne algorithm
    # new_skeleton = cdt.utils.graph.remove_indirect_links(skeleton, alg='aracne')
    # print(nx.adjacency_matrix(new_skeleton).todense())
    # GES algorithm
    # model = cdt.causality.graph.GES()
    # output_graph = model.predict(discover_data, new_skeleton)
    # print(nx.adjacency_matrix(output_graph).todense())
    # scores = [metric(graph, output_graph) for metric in (precision_recall, SID, SHD)]
    # now we compute the CAM graph without constraints and the associated scores
    # model2 = cdt.causality.graph.CAM()
    # output_graph_nc = model2.predict(discover_data)
    # scores_nc = [metric(graph, output_graph_nc) for metric in (precision_recall, SID, SHD)]
    # print(scores_nc)

    return sm


class EconmlRlearner():

    def __init__(self):
        self.GridSearchCV_R_est = None

    def fit_rlearner(self, X, T, Y, score, greater_is_better, outer_fold_num):
        # TODO: pass categorical features to fit_params of gridsearch
        lgbm_space = dict()
        lgbm_space['learning_rate'] = [0.005, 0.01, 0.05, 0.1, 0.2, 1.]
        lgbm_space['n_estimators'] = [20, 30, 40, 50, 70, 150, 200, 500]
        lgbm_space['min_data_in_leaf'] = [10, 20, 30, 50, 100]
        lgbm_space['max_depth'] = [3, 4, 5, 6, 7]

        strf_cv = StratifiedKFold(n_splits=outer_fold_num, shuffle=True, random_state=42)
        cv_clf_space = lambda: GridSearchCV(
            estimator=lgbm.LGBMClassifier(objective="binary",  # early_stopping_rounds=15,
                                          random_state=500, silent=True, n_jobs=4),
            # GradientBoostingClassifier(validation_fraction=0.12, n_iter_no_change=10, tol=0.001)
            param_grid=lgbm_space,
            # param_grid={
            #     'max_depth': [3, 4, 5],
            #     'n_estimators': [30, 40, 50, 60]},
            cv=strf_cv, n_jobs=-1, scoring=make_scorer(score, greater_is_better=greater_is_better)
            , refit=True
        )

                 # "fit_params": {'categorical_feature': categorical_names_list}}

        cv_reg_space = lambda: GridSearchCV(
            estimator=lgbm.LGBMRegressor(random_state=500, silent=True, n_jobs=4),
            # GradientBoostingRegressor(validation_fraction=0.2, n_iter_no_change=10, tol=0.001),
            param_grid=lgbm_space
            # {
            #     'max_depth': [3, 4, 5],
            #     'n_estimators': [30, 40, 50, 60],}
            , cv=outer_fold_num, n_jobs=-1, scoring='neg_root_mean_squared_error'
            , refit=True
        )
        # cv_reg_space = GradientBoostingRegressor(max_depth=3, n_estimators=50)

        self.GridSearchCV_R_est = NonParamDML(model_y=cv_clf_space(),
                                              model_t=cv_clf_space(),
                                              model_final=cv_reg_space(),
                                              discrete_treatment=True,
                                              random_state=42)
        self.GridSearchCV_R_est.fit(Y=Y, T=T, X=X)
        return self.GridSearchCV_R_est

    def predict_rlearner(self, X):
        return self.GridSearchCV_R_est.effect(X)


class EconMlXlearner():

    def __init__(self):
        self.x_est = None

    def fit_xlearner(self, X, T, Y, score, outer_fold_num, greater_is_better):

        # TODO: pass categorical features to fit_params of gridsearch
        lgbm_space = dict()
        lgbm_space['learning_rate'] = [0.005, 0.01, 0.05, 0.1, 0.2, 1.]
        lgbm_space['n_estimators'] = [20, 30, 40, 50, 70, 150, 200, 500]
        lgbm_space['min_data_in_leaf'] = [10, 20, 30, 50, 100]
        lgbm_space['max_depth'] = [3, 4, 5, 6, 7]
        strf_cv = StratifiedKFold(n_splits=outer_fold_num, shuffle=True, random_state=42)
        cv_clf_space = lambda: GridSearchCV(
            estimator=lgbm.LGBMClassifier(objective="binary",  # early_stopping_rounds=15,
                                          random_state=500, silent=True, n_jobs=4),
            # GradientBoostingClassifier(validation_fraction=0.12, n_iter_no_change=10, tol=0.001)
            param_grid=lgbm_space
            # param_grid={
            #     'max_depth': [3, 4, 5],
            #     'n_estimators': [30, 40, 50, 60]},
        , cv=strf_cv, n_jobs=-1, scoring=make_scorer(score, greater_is_better=greater_is_better)
            , refit=True
        )

        cv_reg_space = lambda: GridSearchCV(
            estimator=lgbm.LGBMRegressor(random_state=500, silent=True, n_jobs=4),
            # GradientBoostingRegressor(validation_fraction=0.2, n_iter_no_change=10, tol=0.001),
            param_grid=lgbm_space
            # {
            #     'max_depth': [3, 4, 5],
            #     'n_estimators': [30, 40, 50, 60, 70],
            # }
            , cv=outer_fold_num, n_jobs=-1, scoring='neg_root_mean_squared_error'
            , refit=True)

        self.x_est = XLearner(models=cv_clf_space(),
                              propensity_model=cv_clf_space(),
                              cate_models=cv_reg_space())

        self.x_est.fit(Y, T, X=X)
        return self.x_est

    def predict_xlearner(self, X):
        return self.x_est.effect(X)


def sigmoid_calib(y, y_score):
    """ re-calibration of prediction, using Plat 2000

    using implantation by sci-kit learn

    Args:
        y: the target variable
        y_score: the prediction


    Returns:

    """
    a, b = _sigmoid_calibration(y_score, y)
    return expit(-(a * y_score + b))


def regression_importance(data: pd.DataFrame, outcome: pd.DataFrame,
                                   feature_names: list, file_name: str, path: str, outcome_type: str = "binary"):

    if outcome_type == "binary":
        lr = LogisticRegression(C=10, penalty='l2', solver="liblinear")
        lr.fit(X=data, y=outcome)
        importance = lr.coef_[0]
    elif outcome_type == "cont":
        lr = LinearRegression()
        lr.fit(X=data, y=outcome)
        importance = lr.coef_
    else:
        print("outcome_type {} not supported".format(outcome_type))
        return

    # get importance
    lr_ftrs_importance_df = pd.DataFrame()
    lr_ftrs_importance_df["importance"] = importance
    lr_ftrs_importance_df["features"] = feature_names
    lr_ftrs_importance_df["abs_importance"] = lr_ftrs_importance_df.importance.abs()
    result = lr_ftrs_importance_df.reindex(lr_ftrs_importance_df["abs_importance"].sort_values(
        ascending=False).index).round(2)
    print(result)
    result.to_csv(os.path.join(path, "{}.csv".format(file_name)))

    # shap_explainer = shap.LinearExplainer(model=lr_explain, data=chosen_nsclc_mortality_df[all_chosen_features],
    #                                       masker=shap.TabularMasker(chosen_nsclc_mortality_df[all_chosen_features],
    #                                                                 hclustering="correlation"))
    # shap_values = shap_explainer.shap_values(chosen_nsclc_mortality_df[all_chosen_features])
    # shap.summary_plot(shap_values, chosen_nsclc_mortality_df[all_chosen_features],
    #                   color=plt.get_cmap("cool"), show=False)
    #
    # plt.title(plt_title)
    # shap.plots.beeswarm(shap_values,
    #                     color=plt.get_cmap("cool"), show=False)
    # plt.tight_layout()
    # plt.ioff()
    # plt.savefig(plt_title + "_feature_importance.png")  # , bbox_inches='tight', pad_inches=1)
    # # plt.show()
    # plt.close()

    return


def set_binary_clf_threshold(model, X, y, model_name, fig_path):
    """
    finds the best discriminating threshold for classifier
    :param model:
    :param X:
    :param y:
    :param model_name:
    :param fig_path:
    :return:
    """

    visualizer = DiscriminationThreshold(estimator=model, is_fitted=True)
    visualizer.fit(X, y)
    visualizer.show()
    visualizer.fig.savefig(os.path.join(fig_path, "{}_threshold.png".format(model_name)))
    plt.close()


def calc_ipw(p_treatment1: np.array, treatment: np.array, treatment1_name, treatment0_name, stabilized=True):
    """
    calculates inverse propensity weights for binary treatment allocation - assuming 2 treatments
    :param p_treatment1: probability to get treatment1 P(T=treatment1_name|X) # TODO: change name p_treatment1
    :param treatment: actual treatment allocated
    :param treatment1_name:
    :param treatment0_name:
    :return:
    """
    p_treated = np.where(treatment == treatment1_name, p_treatment1, 1-p_treatment1)
    ipw = 1. / p_treated

    if stabilized:
        sum_treated1 = sum(ipw[treatment == treatment1_name])
        sum_treated0 = sum(ipw[treatment == treatment0_name])
        demon = np.where(treatment == treatment1_name, sum_treated1, sum_treated0)
        ipw = ipw / demon

    return ipw


def sort_features_by_shap_values(model, data: pd.DataFrame, features_for_model, max_order=False):
    """
    Calculates SHAP values and returns by features importance
    :param model:
    :param data:
    :param features_for_model:
    :param max_order: decide if to sort by max absolute value or else by average absolute value
    :return:
    """
    explainer = shap.Explainer(model, data[features_for_model])
    shap_values = explainer(data[features_for_model], check_additivity=False)  # TODO: debug check_additivity
    # TODO: validate logic again against shap graphs
    if max_order:
        # get shap ordered features by max absolute effect - high impact for individuals
        top_shap_features = [features_for_model[i] for i in np.argsort(shap_values.abs.max(0).values)[::-1]]
    else:
        # get shap ordered features by mean absolute effect - broad average impact
        top_shap_features = [features_for_model[i] for i in np.argsort(shap_values.abs.mean(0).values)[::-1]]

    return top_shap_features


def get_score_params(score_name: str):
    # metric performance definition
    score_types_dict = {"auc": roc_auc_score,
                        "prc_rcl_curve": average_precision_score,
                        "log_loss": log_loss,
                        "brier": brier_score_loss
                        }
    score = score_types_dict[score_name]
    # set score is cost or rewrad for optimization
    if score_name in ["auc", "prc_rcl_curve"]:
        greater_is_better = True
    elif score_name in ["log_loss", "brier"]:
        greater_is_better = False
    else:
        print("greater_is_better NOT DEFINED")
        greater_is_better = None
    return score, greater_is_better


def get_scaler_params(scaler_name: str):

    scaler_name_dict = {"MinMaxScaler": MinMaxScaler(),
                        "StandardScaler": StandardScaler(),
                        "RobustScaler": RobustScaler()}
    try:
        scaler = scaler_name_dict[scaler_name]
        return scaler
    except KeyError:
        print("Scaler {} not supported".format(scaler_name))


def get_binary_classifier_models_dict(clf_name_list: list, categorical_names_list: list = None) -> dict:
    """
    prepares classifiers dict by given names
    :param clf_name_list:
    :return:
    """

    # define search spaces
    space_trees = dict()
    space_trees['tol'] = [0.005, 0.01, 0.05, 0.1, 0.2, 1.]
    space_trees['n_estimators'] = [20, 30, 40, 50, 70, 150, 200, 500]
    space_trees['min_samples_leaf'] = [10, 20, 30, 50, 100]
    space_trees['max_depth'] = [3, 4, 5, 6, 7]

    lgbm_space = dict()
    lgbm_space['learning_rate'] = [0.005, 0.01, 0.05, 0.1, 0.2, 1.]
    lgbm_space['n_estimators'] = [20, 30, 40, 50, 70, 150, 200, 500]
    lgbm_space['min_data_in_leaf'] = [10, 20, 30, 50, 100]
    lgbm_space['max_depth'] = [3, 4, 5, 6, 7]

    # lgbm_space['max_bin'] = [255, 510]  # large max_bin helps improve accuracy but might slow down training progress
    # lgbm_space['num_leaves'] = [6, 8, 12, 16]  # large num_leaves helps improve accuracy but might lead to over-fitting
    # lgbm_space['boosting_type']= ['gbdt', 'dart']  # for better accuracy -> try dart
    # lgbm_space['objective']= ['binary']
    # lgbm_space['colsample_bytree']= [0.64, 0.65, 0.66]
    # lgbm_space['subsample']=[0.7, 0.75]
    # lgbm_space['reg_alpha']=[1, 1.2]
    # lgbm_space['reg_lambda']=[1, 1.2, 1.4]
    # lgbm_space['early_stopping_rounds'] = [15]

    space_log_reg = dict()
    space_log_reg["C"] = [0.001, 0.009, 0.01, 0.09, 1, 5, 10, 25]
    space_log_reg["penalty"] = ["l1", "l2"]

    space_svm = dict()
    space_svm['kernel'] = ['linear', 'rbf']
    space_svm['C'] = [0.1, 1, 5, 10]

    space_gp = dict()
    # space_gp['multi_class'] = ['one_vs_rest', 'one_vs_one']
    space_gp['kernel'] = [10 * RBF(), 1.0 * Matern()]

    space_sgd = dict()
    space_sgd['loss'] = ["hinge", "log", "squared_hinge", "modified_huber"]
    space_sgd['alpha'] = [0.0001, 0.001, 0.01, 0.1]

    binary_classifier_models_dict = {
        "LogisticRegression": {"model": LogisticRegression(solver="liblinear"
                               # , multi_class="ovr",
                               #                             max_iter=100,
                               #                             C=1, penalty='l1'
                                    ),
                               "space": space_log_reg},
        "GradientBoostingClassifier": {"model": GradientBoostingClassifier(validation_fraction=0.15,
                                                                           n_iter_no_change=15),
                                       "space": space_trees},
        "SVC": {"model": SVC(probability=True),
                "space": space_svm},
        "RandomForestClassifier": {"model": RandomForestClassifier(criterion='entropy'),
                                   "space": space_trees},
        "GaussianProcessClassifier": {"model": GaussianProcessClassifier(),
                                      "space": space_gp},
        "SGDClassifier": {"model": SGDClassifier(max_iter=1000, tol=1e-3),
                          "space": space_sgd},

        "lgbm": {"model": lgbm.LGBMClassifier(objective="binary",  # early_stopping_rounds=15,
                                              random_state=500, silent=True, n_jobs=4),
                 "space": lgbm_space,
                 "fit_params": {'categorical_feature': categorical_names_list}}
    } # TODO: add early stopping for lgbm
    return {key: binary_classifier_models_dict[key]
            for key in set(clf_name_list).intersection(binary_classifier_models_dict.keys())}
