import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib
#import seaborn as sns
from adjustText import adjust_text
import re


def plot_hist_exp_1(results, household_size, pool_size, prevalence):
    fnr_indep = results[:, 0]
    fnr_correlated = results[:, 1]
    eff_indep = results[:, 2]
    eff_correlated = results[:, 3]
    test_indep = results[:, 4]
    test_correlated = results[:, 5]

    fig, [ax0, ax1] = plt.subplots(1,2, figsize=(10,6))
    ax0.hist([fnr_indep, fnr_correlated], label=['randomized pooling', 'correlated pooling'], color=['mediumaquamarine', 'mediumpurple'])
    ax0.legend(loc='upper right')
    ax0.set_xlabel('$FNR$')
    ax0.set_ylabel('Frequency')
    ax0.set_title('FNR values under randomized and\ncorrelated pooling')

    ax1.hist(fnr_indep - fnr_correlated, color='lightskyblue', rwidth=0.7)
    #ax1.legend(loc='upper right')
    ax1.set_title('difference in FNR values')
    #ax1.set_xlabel('$FNR_{indep} - FNR_{correlated}$')
    ax1.set_ylabel('Frequency')

    #plt.title('Histogram of FNR values for one-stage group testing \n under household size = {}, pool size = {}, prevalence = {}'.format(household_size, pool_size, prevalence))
    plt.tight_layout()
    plt.savefig('../figs/experiment_1/fnr_diff_pool-size={}_household-size={}_prevalence={}.pdf'.format(pool_size, household_size, prevalence))
    plt.close()


    fig, [ax0, ax1] = plt.subplots(1,2, figsize=(10,6))
    ax0.hist([test_indep, test_correlated], label=['randomized pooling', 'correlated pooling'], color=['mediumaquamarine', 'mediumpurple'])
    ax0.legend(loc='upper right')
    ax0.set_xlabel('$\#$ followup tests per positive identified')
    ax0.set_ylabel('Frequency')
    ax0.set_title('$\#$ followup tests per positive identified under\nrandomized and correlated pooling')

    ax1.hist(test_indep - test_correlated, color='lightskyblue', rwidth=0.7)
    #ax1.legend(loc='upper right')
    ax1.set_title('difference in $\#$ followup tests per positive identified')
    #ax1.set_xlabel('$\#_{followup, corr, per} - \#_{follorup, indep, per}$')
    ax1.set_ylabel('Frequency')

    #plt.title('Histogram of FNR values for one-stage group testing \n under household size = {}, pool size = {}, prevalence = {}'.format(household_size, pool_size, prevalence))
    plt.tight_layout()
    plt.savefig('../figs/experiment_1/relative_test_consumption_pool-size={}_household-size={}_prevalence={}.pdf'.format(pool_size, household_size, prevalence))
    plt.close()

    # plt.hist([eff_indep, eff_correlated], label=['independent group testing', 'correlated group testing'], alpha=0.5)
    # plt.legend(loc='upper right')
    # plt.xlabel('Efficiency')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of testing efficiency for one-stage group testing \n under household size = {}, pool size = {}, prevalence = {}'.format(household_size, pool_size, prevalence))
    # plt.savefig('../figs/experiment_1/eff_pool-size={}_household-size={}_prevalence={}.pdf'.format(pool_size, household_size, prevalence))
    # plt.close()
    return


def generate_heatmap_plots_for_exp_1():
    dir = '../results/experiment_1'

    aggregate_results = {}

    for filename in os.listdir(dir):
        if filename == ".DS_Store" or not filename.endswith('.data'):
            continue

        parts = re.split('=|[.](?!\d)|_', filename)
        print(parts)
        household_size = int(parts[4])
        prevalence = float(parts[6])

        filedir = os.path.join(dir, filename)
        with open(filedir) as f:
            results = np.loadtxt(f)

        avgs = np.mean(results, axis=0)
        aggregate_results[(prevalence, household_size)] = avgs

    df_agg = pd.DataFrame.from_dict(aggregate_results, orient='index', columns=['indep fnr', 'corr fnr', 'indep eff', 'corr eff', 'indep test', 'corr test'])
    df_agg.index = pd.MultiIndex.from_tuples(df_agg.index, names=['prevalence', 'household size'])
    df_agg = df_agg.reset_index()
    df_agg = df_agg.sort_values(by=['prevalence', 'household size'])
    df_agg['indep sn'] = 1 - df_agg['indep fnr']
    df_agg['corr sn'] = 1 - df_agg['corr fnr']
    df_agg['sn diff'] = df_agg['corr sn'] - df_agg['indep sn']
    df_agg['rel test consumption'] = df_agg['corr test'] / df_agg['indep test']

    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(8, 4))
    table_sn = pd.pivot_table(df_agg, values='sn diff', index=['household size'], columns=['prevalence'])
    print(table_sn)
    heatmap = ax0.pcolor(table_sn, cmap=cm.BuPu)
    ax0.set_aspect('equal')
    ax0.set_yticks(np.arange(0.5, len(table_sn.index), 1))
    ax0.set_yticklabels(table_sn.index)
    ax0.set_xticks(np.arange(0.5, len(table_sn.columns), 1))
    ax0.set_xticklabels(table_sn.columns)
    ax0.set_xlabel('prevalence')
    ax0.set_ylabel('household size')
    ax0.set_title('Difference in FNR')
    fig.colorbar(heatmap, ax=ax0, orientation="horizontal")
    #plt.savefig('../figs/experiment_2/pareto_plots/tmp_heapmap_for_fnr.pdf')
    #plt.close()

    table_test = pd.pivot_table(df_agg, values='rel test consumption', index=['household size'], columns=['prevalence'])
    heatmap = ax1.pcolor(table_test, cmap=cm.YlGn_r)
    ax1.set_aspect('equal')
    ax1.set_yticks(np.arange(0.5, len(table_test.index), 1))
    ax1.set_yticklabels(table_test.index)
    ax1.set_xticks(np.arange(0.5, len(table_test.columns), 1))
    ax1.set_xticklabels(table_test.columns)
    ax1.set_xlabel('prevalence')
    ax1.set_ylabel('household size')
    ax1.set_title('Relative test consumption')
    fig.colorbar(heatmap, ax=ax1, orientation="horizontal")
    # plt.savefig('../figs/experiment_2/pareto_plots/tmp_heapmap_for_eff.pdf')
    # plt.close()
    fig.tight_layout()
    fig.savefig('../figs/experiment_1/tmp_heapmap_for_fnr_and_test.pdf', bbox_inches='tight')
    plt.clf()


def plot_hist_exp_2(results, param, val=None):
    fnr_indep = results[:, 0]
    fnr_correlated = results[:, 1]
    eff_indep = results[:, 2]
    eff_correlated = results[:, 3]

    #num_iters = size(fnr_indep)

    plt.hist([fnr_indep, fnr_correlated], label=['randomized', 'correlated'], color=['mediumaquamarine', 'mediumpurple'])
    plt.legend(loc='upper right')
    plt.xlabel('False negative rate')
    plt.ylabel('Frequency')
    if param == 'nominal':
        plt.title('Histogram of FNR values under {} scenario'.format(param))
        plt.savefig('../figs/experiment_2/fnr_{}_scenario.pdf'.format(param))
    else:
        plt.title('Histogram of FNR values for one-stage group testing \n under {} = {}'.format(param, val))
        plt.savefig('../figs/experiment_2/fnr_{}={}.pdf'.format(param, val))
    plt.close()

    plt.hist([eff_indep, eff_correlated], label=['randomized', 'correlated'], color=['mediumaquamarine', 'mediumpurple'])
    plt.legend(loc='upper right')
    plt.xlabel('Efficiency')
    plt.ylabel('Frequency')
    if param == 'nominal':
        plt.title('Histogram of testing efficiency under {} scenario'.format(param))
        plt.savefig('../figs/experiment_2/eff_{}_scenario.pdf'.format(param))
    else:
        plt.title('Histogram of testing efficiency for one-stage group testing \n under {} = {}'.format(param, val))
        plt.savefig('../figs/experiment_2/eff_{}={}.pdf'.format(param, val))
    plt.close()
    return


def generate_sensitivity_plots(param):
    dir = '../results/experiment_2/sensitivity_analysis/'

    fnr_indep = []
    fnr_corr = []
    eff_indep = []
    eff_corr = []
    index = []

    for filename in os.listdir(dir):
        if param in filename:
            val = filename.split(param, 1)[1][:-5]
            val = val.split('_', 1)[0][1:]
            val = int(val) if param == 'pool size' else val if param == 'household dist' else float(val)

            filedir = os.path.join(dir, filename)
            with open(filedir) as f:
                results = np.loadtxt(f)

            avgs = np.mean(results, axis=0)
            #print('{} = {}'.format(param, val))
            fnr_indep.append(avgs[0])
            fnr_corr.append(avgs[1])
            eff_indep.append(avgs[2])
            eff_corr.append(avgs[3])
            index.append(val)

    df = pd.DataFrame({'FNR (randomized)': fnr_indep, 'FNR (correlated)': fnr_corr, 'efficiency (randomized)': eff_indep,'efficiency (correlated)': eff_corr}, index=index)
    df = df.sort_index()
    df = df.rename_axis(param).reset_index()
    #print(df)
    fig, [ax0, ax1] = plt.subplots(2, 1, sharex=False, figsize=(6, len(df)+ 3))
    df[['FNR (randomized)', 'FNR (correlated)']].plot.barh(ax=ax0, legend=False, color=['mediumaquamarine', 'mediumpurple'])
    ax0.set_xlabel('FNR')
    df[['efficiency (randomized)', 'efficiency (correlated)']].plot.barh(ax=ax1, legend=False, color=['mediumaquamarine', 'mediumpurple'])
    ax1.set_xlabel('efficiency')
    ax0.legend(['randomized', 'correlated'], loc='lower left', bbox_to_anchor=(0, 1.02, 0.6, 1.02), ncol=2)
    fig.text(0.02, 0.5, param, va='center', rotation='vertical')
    fig.savefig('../figs/experiment_2/sensitivity_plots/tmp_sensitivity_for_{}.pdf'.format(param), bbox_inches='tight')
    fig.tight_layout()
    plt.clf()

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    fnrs = df[['FNR (randomized)', 'FNR (correlated)']].plot.bar(ax=ax, legend=False, color=['mediumaquamarine', 'mediumpurple'], alpha=0.5)
    effs = df[['efficiency (randomized)', 'efficiency (correlated)']].plot.line(ax=ax2, legend=False, marker='o', markeredgecolor='w', color=['mediumaquamarine', 'mediumpurple'])
    ax.set_xticklabels(df[param])
    ax.set_ylabel('FNR')
    ax2.set_ylabel('efficiency')
    ax2.set_ylim(1) if param in ['prevalence', 'pool size'] else ax2.set_ylim(4.5)
    ax.set_xlabel(param) if param != 'FNR' else ax.set_xlabel('individual testing average FNR')
    h, l = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h + h2, l + l2, loc='lower left', bbox_to_anchor=(0, 1.02, 0.6, 1.02), ncol=2)
    fig.savefig('../figs/experiment_2/sensitivity_plots/tmp_sensitivity_for_{}_alternative.pdf'.format(param), bbox_inches='tight')
    plt.clf()
    return


def generate_pareto_fontier_plots():
    dir = '../results/experiment_2/pareto_analysis/'

    aggregate_results = {}

    for filename in os.listdir(dir):
        if filename == ".DS_Store":
            continue

        parts = re.split('=|[.](?!\d)|_', filename)
        prev = float(parts[2])
        pool_size = int(parts[4])

        filedir = os.path.join(dir, filename)
        with open(filedir) as f:
            results = np.loadtxt(f)

        avgs = np.mean(results, axis=0)
        aggregate_results[(prev, pool_size)] = avgs

    df_agg = pd.DataFrame.from_dict(aggregate_results, orient='index', columns=['fnr (randomized)', 'fnr (correlated)', 'eff (randomized)', 'eff (correlated)'])
    df_agg.index = pd.MultiIndex.from_tuples(df_agg.index, names=['prevalence', 'pool size'])
    df_agg = df_agg.reset_index()
    df_agg = df_agg.sort_values(by=['prevalence', 'pool size'])
    df_agg['sn (randomized)'] = 1 - df_agg['fnr (randomized)']
    df_agg['sn (correlated)'] = 1 - df_agg['fnr (correlated)']

    for prev in df_agg['prevalence'].unique():
        df = df_agg[df_agg['prevalence'] == prev]
        ax = df.sort_values(by='pool size').plot(x='sn (randomized)', y = 'eff (randomized)', sort_columns=True, color='mediumpurple', marker='^')
        df.sort_values(by='pool size').plot(x='sn (correlated)', y = 'eff (correlated)', sort_columns=True, ax=ax, color='mediumaquamarine', marker='o')

        # for i, point in df.iterrows():
        #     ax.text(point['indep sn'], point['indep eff'], str(int(point['pool size'])), color='dimgrey')
        #     ax.text(point['corr sn'], point['corr eff'], str(int(point['pool size'])), color='dimgrey')

        texts = []
        for i, point in df.iterrows():
            texts.append(ax.text(point['sn (randomized)'], point['eff (randomized)'], str(int(point['pool size'])), color='dimgrey'))
            texts.append(ax.text(point['sn (correlated)'], point['eff (correlated)'], str(int(point['pool size'])), color='dimgrey'))

        adjust_text(texts, only_move={'points':'y', 'texts':'xy'})


        plt.legend(['randomized', 'correlated'])
        plt.xlabel('Sensitivity = 1 - FNR')
        plt.ylabel('Efficiency')
        plt.title('Tradeoff between test efficiency and sensitivity\n under prevalence = {}'.format(prev))
        plt.grid(True, ls=':')
        plt.savefig('../figs/experiment_2/pareto_plots/tmp_pareto_for_prev_{}.pdf'.format(prev), bbox_inches='tight')
        plt.close()

    df_agg['sn diff'] = df_agg['sn (correlated)'] - df_agg['sn (randomized)']
    df_agg['eff diff'] = df_agg['eff (correlated)'] - df_agg['eff (randomized)']


    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(8, 4))
    table_sn = pd.pivot_table(df_agg, values='sn diff', index=['prevalence'], columns=['pool size'])
    print(table_sn)
    heatmap = ax0.pcolor(table_sn, cmap=cm.BuPu)
    ax0.set_aspect('equal')
    ax0.set_yticks(np.arange(0.5, len(table_sn.index), 1))
    ax0.set_yticklabels(table_sn.index)
    ax0.set_xticks(np.arange(0.5, len(table_sn.columns), 1))
    ax0.set_xticklabels(table_sn.columns)
    ax0.set_xlabel('pool size')
    ax0.set_ylabel('prevalence')
    ax0.set_title('Difference in sensitivity')
    fig.colorbar(heatmap, ax=ax0, orientation="horizontal")

    textcolors = ["k", "w"]
    threshold = 0.056
    for i, prev in enumerate(table_sn.index):
        for j, pool_size in enumerate(table_sn.columns):
            text = ax0.text(j+0.5, i+0.5, "{:.3f}".format(table_sn.iloc[i,j]).replace("0.", "."), \
                ha="center", va="center", color=textcolors[table_sn.iloc[i, j] > threshold], size=7)


    #plt.savefig('../figs/experiment_2/pareto_plots/tmp_heapmap_for_fnr.pdf')
    #plt.close()

    table_eff = pd.pivot_table(df_agg, values='eff diff', index=['prevalence'], columns=['pool size'])
    heatmap = ax1.pcolor(table_eff, cmap=cm.YlGn)
    ax1.set_aspect('equal')
    ax1.set_yticks(np.arange(0.5, len(table_eff.index), 1))
    ax1.set_yticklabels(table_eff.index)
    ax1.set_xticks(np.arange(0.5, len(table_eff.columns), 1))
    ax1.set_xticklabels(table_eff.columns)
    ax1.set_xlabel('pool size')
    ax1.set_ylabel('prevalence')
    ax1.set_title('Difference in efficiency')
    fig.colorbar(heatmap, ax=ax1, orientation="horizontal")
    # plt.savefig('../figs/experiment_2/pareto_plots/tmp_heapmap_for_eff.pdf')
    textcolors = ["k", "w"]
    threshold = 0.8
    for i, prev in enumerate(table_eff.index):
        for j, pool_size in enumerate(table_eff.columns):
            text = ax1.text(j+0.5, i+0.5, "{:.3f}".format(table_eff.iloc[i,j]).replace("0.", "."), \
                ha="center", va="center", color=textcolors[table_eff.iloc[i, j] > threshold], size=7)

    # plt.close()
    fig.tight_layout()
    fig.savefig('../figs/experiment_2/pareto_plots/tmp_heapmap_for_fnr_and_eff.pdf', bbox_inches='tight')
    plt.clf()
    return



if __name__ == '__main__':
    # generate_heatmap_plots_for_exp_1()

    # for param in ['prevalence', 'pool size', 'SAR', 'FNR', 'household dist']:
    #     generate_sensitivity_plots(param)
    #
    # generate_pareto_fontier_plots()

    filedir = "../results/experiment_2/sensitivity_analysis/results_prevalence=0.01_SAR=0.188_pool size=6_FNR=0.05_household dist=US.data"
    with open(filedir) as f:
        results = np.loadtxt(f)
    plot_hist_exp_2(results, 'nominal')
