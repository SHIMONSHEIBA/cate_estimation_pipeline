import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.calibration import calibration_curve
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times, restricted_mean_survival_time
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging
import plotly.io as pio
from lifelines.plotting import add_at_risk_counts


sns.set_theme(style="whitegrid")
pio.renderers.default = "browser"

# A logger for this file
log = logging.getLogger(__name__)


def plot_boxplot(x, y, data, title, path_file, hue=None, y_label=None):
    boxplot_test = sns.boxplot(x=x,
                               y=y,
                               hue=hue,
                               data=data)
    boxplot_test.set_title(title)
    if y_label:
        boxplot_test.set_ylabel(y_label)
    boxplot_test.set_xticklabels(boxplot_test.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    boxplot_test.figure.savefig(path_file)
    plt.close()


def hist(data, x, title, path, hue=None):

    ax = sns.histplot(data=data, x=x, hue=hue)
    fig = ax.get_figure()
    fig.savefig(os.path.join(path, "{}.png".format(title)))
    plt.close()


def violin_plot(data, title, file_path, x_col = None, y_col = None, hue_col = None, column_col = None,
                xlim = None, palette = None, col_order = None):

    if y_col:
        g = sns.catplot(x=x_col, y=y_col, hue=hue_col, col=column_col, data=data, kind="violin", split=True, height=4,
                        aspect=1.2, palette=palette, INNER="quartile", bw=.2, col_order=col_order)
    else:
        g = sns.displot(data=data, x=x_col, hue=hue_col, kind='kde', fill=True) # col=column_col,, palette=palette)
    if xlim:
        g.set(xlim=xlim)
    g.fig.suptitle(title, size=14)
    g.fig.subplots_adjust(top=.9)
    g.savefig(os.path.join(file_path, "{}.png".format(title)))

    return


def weighted_kaplan_meier(dfs_to_plot_dict: dict, title: str, time_col: str, event_col: str,
                          x_axis_label: str, file_path: str, x_lim_tuple: tuple = None, y_axis_label: str = None,
                          stabilized: bool = False, y_lim_tuple: tuple = (0, 1), add_counts: bool = False, y_label: str = 'Survival'):
    """
    estimates the kaplan-meier curve with optional re-weighted and grouped
    :param dfs_to_plot_dict: {df label: {df: pd.DataFrame, df_group_col: str = None, df_weight_col: str = None}}
    :param title:
    :param time_col:
    :param event_col:
    :param x_axis_label:
    :param: y_axis_label
    :param file_path:
    :param x_lim_tuple:
    :param stabilized: to normalize weights to sum to 1 (stabilizied version of IPW agreement set weighting)
    :return:
    """
    #  TODO: fix warning of variance (CI) bias estimation
    log.info(title)

    ax = plt.subplot(111)
    ax.set_title(title)
    fitters = list()

    for df_label, df_dict in dfs_to_plot_dict.items():

        log.info("df {} shape: {}".format(df_label, df_dict["df"].shape))
        if df_dict["df_group_col"]:
            for group in df_dict["df"][df_dict["df_group_col"]].unique():
                group_df = df_dict["df"].loc[df_dict["df"][df_dict["df_group_col"]] == group, :]

                # stabilizied version of IPW weighs - normalize by agreement set sum weights
                if df_dict["df_weight_col"] is not None and stabilized:
                    group_df[df_dict["df_weight_col"]] = group_df[df_dict["df_weight_col"]] * \
                                                         (group_df[df_dict["df_weight_col"]].sum()/group_df.shape[0])
                    # TODO: split the marginal probability of actual treatment per arm and make sure which n is right
                    # marginal probability of actual treatment

                kmf = KaplanMeierFitter()
                kmf.fit(group_df[time_col],
                        group_df[event_col],
                        label=df_label+'_'+group,
                        weights=group_df[df_dict["df_weight_col"]] if df_dict["df_weight_col"] is not None else None)
                ax = kmf.plot_survival_function(ax=ax)
                fitters.append(kmf)
                median_ci = median_survival_times(kmf.confidence_interval_)
                print("{} {} group {} median survival time: {} with ci {}".format(df_label,
                                                                                  df_dict["df_group_col"],
                                                                                  group,
                                                                                  kmf.median_survival_time_,
                                                                                  median_ci))
                rmst_exp = restricted_mean_survival_time(kmf, t=x_lim_tuple[1])
                print("restricted_mean_survival_time: {}".format(rmst_exp))

        else:

            # stabilizied version of IPW weighs - normalize by agreement set sum weights
            if df_dict["df_weight_col"] is not None and stabilized:
                df_dict["df"][df_dict["df_weight_col"]] = \
                    df_dict["df"][df_dict["df_weight_col"]] / df_dict["df"][df_dict["df_weight_col"]].sum()

            kmf = KaplanMeierFitter()
            kmf.fit(df_dict["df"][time_col],
                    df_dict["df"][event_col],
                    label=df_label,
                    weights=df_dict["df"][df_dict["df_weight_col"]] if df_dict["df_weight_col"] is not None else None)
            ax = kmf.plot_survival_function(ax=ax)
            fitters.append(kmf)
            # calc statistics
            median_ci = median_survival_times(kmf.confidence_interval_)
            print("{} median survival time: {} with ci {}".format(df_label,
                                                                  kmf.median_survival_time_,
                                                                  median_ci))
            rmst_exp = restricted_mean_survival_time(kmf, t=x_lim_tuple[1])
            print("restricted_mean_survival_time: {}".format(rmst_exp))

    # add counts
    if add_counts:
        add_at_risk_counts(*fitters, ax=ax)
    # adjust plot
    if x_lim_tuple:
        ax.set(xlim=(x_lim_tuple[0], x_lim_tuple[1]))
    if y_lim_tuple:
        ax.set(ylim=(y_lim_tuple[0], y_lim_tuple[1]))
    ax.set_xlabel(x_axis_label)
    if y_axis_label:
        ax.set_ylabel(y_axis_label)
    else:
        ax.set_ylabel('{} Probability'.format(y_label))

    plt.tight_layout()
    ax.figure.savefig(os.path.join(file_path, "{}.png".format(title)))
    plt.close()
    return


def policy_decision_bar_plot_by_time(data: pd.DataFrame, time_col: str, policy_col: str, policy_col_values_order: list,
                                     title: str, file_path: str, ylim: tuple = None):
    """
    plots decisions of policy of which action to recommend per time col value
    :param data:
    :param time_col:
    :param policy_col:
    :param policy_col_values_order:
    :param title:
    :param file_path:
    :param ylim:
    :return:
    """
    g = sns.catplot(x=time_col, hue=policy_col,
                    data=data, kind="count", hue_order=policy_col_values_order,
                    palette="ch:.25")
    if ylim:
        g.set(ylim=(ylim[0], ylim[1]))
    g.fig.suptitle(title, size=14)
    g.fig.subplots_adjust(top=.9)
    g.savefig(os.path.join(file_path, "{}.png".format(title)))
    return


def kaplanmeier(title, df, time_col, event_col, file_path, group_col=None, weights_col=None, x_lim_tuple=None,
                x_axis_label="Timeline"):
    ax = plt.subplot(111)
    ax.set_title(title)
    kmf = KaplanMeierFitter()

    if group_col:
        for group in df[group_col].unique():
            if group is not None:
                group_df = df.loc[df[group_col] == group, :]
                log.info("calculating Kaplan Meier for group {} with shape {}".format(group, group_df.shape))
                if weights_col:
                    kmf.fit(group_df[time_col], group_df[event_col], label=group, weights=group_df[weights_col])
                else:
                    kmf.fit(group_df[time_col], group_df[event_col], label=group)
                kmf.plot_survival_function(ax=ax)

    else:
        log.info("calculating Kaplan Meier for data shape: {} ".format(df.shape))
        if weights_col:
            kmf.fit(df[time_col], df[event_col], weights=df[weights_col])
        else:
            kmf.fit(df[time_col], df[event_col])
        kmf.plot_survival_function(ax=ax)

    # adjust plot
    if x_lim_tuple:
        ax.set(xlim=(x_lim_tuple[0], x_lim_tuple[1]))
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel('Survival Probability')
    ax.figure.savefig(os.path.join(file_path, "{}.png".format(title)))
    plt.close()
    return


def plot_calibration_curve(clf_dict: dict, X, y, title, path):
    """

    :param clf_dict: {clf_name: fitted clf}
    :param X: features for clf
    :param y: outcome
    :return: plots calibration curves of all clf in clf_list
    """

    plt.close()

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for name, clf in clf_dict.items():
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % (name,))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.title(title)
    plt.tight_layout()
    plt.ioff()
    plt.savefig(os.path.join(path, title + ".png"))  # , bbox_inches='tight', pad_inches=1)
    plt.close()

    return


def plot_ecdf(data, col: str, hue=None, stat=None):
    """
    plots empirical CDF
    :param data:
    :param col: the column to plot
    :param hue: stratify based on this col
    :param stat: calculate proportion or count
    :return:
    """
    plt.close()
    sns.ecdfplot(data=data, x=col, hue=hue, stat=stat)
    plt.show()
    plt.tight_layout()
    plt.ioff()
    plt.savefig("ECDF" + ".png")  # , bbox_inches='tight', pad_inches=1)
    plt.close()


def live_scatter(df, x, y, name, color=None, size=None, hover_data=None, cols_to_string=None):
    plt.close()
    if cols_to_string:
        for col in cols_to_string:
            df[col] = df[col].apply(lambda x: json.dumps(x))
    fig = px.scatter(df, x=x, y=y, color=color, size=size, hover_data=hover_data,
                     title="{} Scatter plot X: {} ~ Y: {} ".format(name, x, y))
    fig.show()
    plt.close()


def shap_feature_importance(shap_values, title, path, order_max=False, max_display=15):
    # TODO: for non XGB models ValueError: The beeswarm plot does not support plotting explanations with instances that
    #  have more than one dimension!

    if order_max:
        plt_title = title + " \n SHAP ordered by max absolute value - high impacts for individual"
        plt.title(plt_title)
        shap.plots.beeswarm(shap_values, order=shap_values.abs.max(0),
                            color=plt.get_cmap("cool"), show=False, max_display=max_display)
    else:
        plt_title = title + " \n SHAP ordered by mean absolute value - broad average impact"
        plt.title(plt_title)
        shap.plots.beeswarm(shap_values,
                            color=plt.get_cmap("cool"), show=False, max_display=max_display)
    plt.tight_layout()
    plt.ioff()
    plt.savefig(os.path.join(path, plt_title + "_feature_importance.png"))  # , bbox_inches='tight', pad_inches=1)
    plt.close()


def shap_features_interaction(model, data, title, path, max_display=15):
    """
    plots features interactions
    :param model:
    :param data:
    :param title:
    :param path:
    :param max_display:
    :return:
    """
    explainer = shap.TreeExplainer(model)
    shap_interaction = explainer.shap_interaction_values(data)
    plt_title = title + " \n SHAP features interaction"
    plt.title(plt_title)
    shap.summary_plot(shap_interaction, data, show=False, max_display=max_display)
    plt.tight_layout()
    plt.ioff()
    plt.savefig(os.path.join(path, plt_title + "_features_interaction.png"))
    plt.close()


def shap_dependence_plots(model, data, ftr_1, ftr_2, path, title):

    explainer = shap.TreeExplainer(model)
    shap_interaction = explainer.shap_interaction_values(data)
    # plt_title = title + " \n SHAP features interaction"
    # plt.title(plt_title)
    shap.dependence_plot((ftr_1, ftr_2), shap_interaction, data, display_features=data, show=False)

    plt.tight_layout()
    plt.ioff()
    plt.savefig(os.path.join(path, title + '_' + ftr_1.replace("/", '_') + '_' +
                             ftr_2.replace("/", '_') + "_features_dependence.png"))
    plt.close()


def plot_cates(cate_x_axis: pd.DataFrame, cate_y_axis: pd.DataFrame, title: str, path: str, is_probability=True):
    """
    scatterplot two CATE for same patients - assuming index is data point key, colors by sum of CATE
    :param cate_x_axis:
    :param cate_y_axis:sc
    :param title:
    :param path:
    :param is_probability: indicator weather cate between Y of probability, then axis are between -1 to 1
    :return:
    """
    plt.close()
    # merge cate df
    data = cate_x_axis.merge(cate_y_axis, how='inner', left_index=True, right_index=True)
    # sum cate for coloring by levels
    data["sum_cates"] = data[cate_x_axis.columns[0]] + data[cate_y_axis.columns[0]]

    f, ax = plt.subplots()
    sns.despine(f, left=True, bottom=True)
    g = sns.scatterplot(x=cate_x_axis.columns[0], y=cate_y_axis.columns[0],
                        hue="sum_cates", data=data, ax=ax, legend=False)
    if is_probability:
        ax.set_xticks(np.arange(-1.0, 1.1, 0.2))
        ax.set_yticks(np.arange(-1.0, 1.1, 0.2))
    plt.title(title)
    g.figure.savefig(os.path.join(path, "{}.png".format(title)))
    plt.close()


def scatterplot(title: str, data, x_col: str, y_col: str, filename: str,
                hue=None, hue_order=None, is_probability=False):
    """
    Scatter plot
    :param title:
    :param data:
    :param x_col:
    :param y_col:
    :param filename: full path + file name
    :param hue:
    :param hue_order:
    :param is_probability: set axis between 0 to 1
    :return:
    """

    plt.close()
    f, ax = plt.subplots(figsize=(10, 10))
    sns.despine(f, left=True, bottom=True)
    g = sns.scatterplot(x=x_col,
                        y=y_col,
                        data=data,
                        hue=hue,
                        hue_order=hue_order,
                        ax=ax)
    # Put the legend out of the figure
    plt.legend(bbox_to_anchor=(1.01, 1),
               borderaxespad=0)
    plt.title(title)
    plt.tight_layout()
    if is_probability:
        ax.set_xticks(np.arange(0.0, 1.1, 0.1))
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))

    g.figure.savefig("{}.png".format(filename), format='png', dpi=150)
    plt.close()


def plot_multiple_metrics_GridSearchCV(results, scoring, filename):
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)

    plt.xlabel("min_samples_split")
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(0, 402)
    ax.set_ylim(0.73, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results["param_min_samples_split"].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ["g", "k"]):
        for sample, style in (("train", "--"), ("test", "-")):
            sample_score_mean = results["mean_%s_%s" % (sample, scorer)]
            sample_score_std = results["std_%s_%s" % (sample, scorer)]
            ax.fill_between(
                X_axis,
                sample_score_mean - sample_score_std,
                sample_score_mean + sample_score_std,
                alpha=0.1 if sample == "test" else 0,
                color=color,
            )
            ax.plot(
                X_axis,
                sample_score_mean,
                style,
                color=color,
                alpha=1 if sample == "test" else 0.7,
                label="%s (%s)" % (scorer, sample),
            )

        best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
        best_score = results["mean_test_%s" % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot(
            [
                X_axis[best_index],
            ]
            * 2,
            [0, best_score],
            linestyle="-.",
            color=color,
            marker="x",
            markeredgewidth=3,
            ms=8,
        )

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    # plt.show()
    plt.savefig("{}.png".format(filename), format='png', dpi=150)
    plt.close()
