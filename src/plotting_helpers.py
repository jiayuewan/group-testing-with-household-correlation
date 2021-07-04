import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib
from adjustText import adjust_text
import re
import matplotlib.patheffects as pe


# deprecated
def plot_hist_exp_1(results, household_size, pool_size, prevalence):
    fnr_indep = results[:, 0]
    fnr_correlated = results[:, 1]
    eff_indep = results[:, 2]
    eff_correlated = results[:, 3]
    test_indep = results[:, 4]
    test_correlated = results[:, 5]

    fig, [ax0, ax1] = plt.subplots(1,2, figsize=(10,6))
    ax0.hist([fnr_indep, fnr_correlated], label=['naive pooling', 'correlated pooling'], color=['mediumaquamarine', 'mediumpurple'])
    ax0.legend(loc='upper right')
    ax0.set_xlabel('$FNR$')
    ax0.set_ylabel('Frequency')
    ax0.set_title('FNR values under naive and\ncorrelated pooling')

    ax1.hist(fnr_indep - fnr_correlated, color='lightskyblue', rwidth=0.7)
    ax1.set_title('difference in FNR values')
    ax1.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('../figs/experiment_1/fnr_diff_pool-size={}_household-size={}_prevalence={}.pdf'.format(pool_size, household_size, prevalence))
    plt.close()

    fig, [ax0, ax1] = plt.subplots(1,2, figsize=(10,6))
    ax0.hist([test_indep, test_correlated], label=['naive pooling', 'correlated pooling'], color=['mediumaquamarine', 'mediumpurple'])
    ax0.legend(loc='upper right')
    ax0.set_xlabel('$\#$ followup tests per positive identified')
    ax0.set_ylabel('Frequency')
    ax0.set_title('$\#$ followup tests per positive identified under\nnaive and correlated pooling')

    ax1.hist(test_indep - test_correlated, color='lightskyblue', rwidth=0.7)
    ax1.set_title('difference in $\#$ followup tests per positive identified')
    ax1.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('../figs/experiment_1/relative_test_consumption_pool-size={}_household-size={}_prevalence={}.pdf'.format(pool_size, household_size, prevalence))
    plt.close()
    return


# deprecated
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
    fig.tight_layout()
    fig.savefig('../figs/experiment_1/tmp_heapmap_for_fnr_and_test.pdf', bbox_inches='tight')
    plt.clf()
    return


def plot_hist_exp_2(results, param, val=None):
    fnr_indep = results[:, 0]
    fnr_correlated = results[:, 1]
    eff_indep = results[:, 2]
    eff_correlated = results[:, 3]

    ax1 = plt.subplot(111)

    n, bins, patches = ax1.hist(results[:, :2], label=['naive', 'correlated'], color=['mediumaquamarine', 'mediumpurple'])

    hatches = [".", '//']
    for patch_set, hatch in zip(patches, hatches):
        for patch in patch_set.patches:
            patch.set_hatch(hatch)
            patch.set_edgecolor('k')

    plt.legend(loc='upper right')
    plt.xlabel('False negative rate')
    plt.ylabel('Frequency')
    if param == 'nominal':
        plt.title('Histogram of FNR values under {} scenario'.format(param))
        plt.savefig('../figs/experiment_2/fnr_{}_scenario.pdf'.format(param))
    else:
        plt.title('Histogram of FNR values for one-stage group testing \n under {} = {}'.format(param, val))
        plt.savefig('../figs/experiment_2/fnr_{}={}.pdf'.format(param, val), dpi=600)
    plt.close()

    ax2 = plt.subplot(111)
    n, bins, patches = ax2.hist(results[:, 2:], label=['naive', 'correlated'], color=['mediumaquamarine', 'mediumpurple'])

    hatches = ["..", '//']
    for patch_set, hatch in zip(patches, hatches):
        for patch in patch_set.patches:
            patch.set_hatch(hatch)

    plt.legend(loc='upper right')
    plt.xlabel('Efficiency')
    plt.ylabel('Frequency')
    if param == 'nominal':
        plt.title('Histogram of testing efficiency under {} scenario'.format(param))
        plt.savefig('../figs/experiment_2/eff_{}_scenario.pdf'.format(param))
    else:
        plt.title('Histogram of testing efficiency for one-stage group testing \n under {} = {}'.format(param, val))
        plt.savefig('../figs/experiment_2/eff_{}={}.pdf'.format(param, val), dpi=600)
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
            fnr_indep.append(avgs[0])
            fnr_corr.append(avgs[1])
            eff_indep.append(avgs[2])
            eff_corr.append(avgs[3])
            index.append(val)

    df = pd.DataFrame({'FNR (naive)': fnr_indep, 'FNR (correlated)': fnr_corr, 'efficiency (naive)': eff_indep,'efficiency (correlated)': eff_corr}, index=index)
    df = df.sort_index()
    df = df.rename_axis(param).reset_index()

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    fnrs = df[['FNR (naive)', 'FNR (correlated)']].plot.bar(ax=ax, legend=False, color=['mediumaquamarine', 'mediumpurple'], alpha=1)
    l = df.shape[0]

    bars = ax.patches
    hatches = [".."] * l + ['//'] * l

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    df[['efficiency (naive)']].plot.line(ax=ax2, legend=False, marker='^', markeredgecolor='w', markeredgewidth=0, \
        color=['mediumaquamarine'], path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()])

    df[['efficiency (correlated)']].plot.line(ax=ax2, legend=False, marker='o', markeredgecolor='w', markeredgewidth=0, \
                color=['mediumpurple'], path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()])

    ax.set_xticklabels(df[param])
    ax.set_ylabel('FNR')
    ax2.set_ylabel('efficiency')
    ax2.set_ylim(1) if param in ['prevalence', 'pool size'] else ax2.set_ylim(4.5)
    ax.set_xlabel(param) if param != 'FNR' else ax.set_xlabel('individual testing average FNR')
    h, l = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h + h2, l + l2, loc='lower left', bbox_to_anchor=(0, 1.02, 0.6, 1.02), ncol=2)
    fig.savefig('../figs/experiment_2/sensitivity_plots/tmp_sensitivity_for_{}_alternative.pdf'.format(param), bbox_inches='tight', dpi=600)
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

    df_agg = pd.DataFrame.from_dict(aggregate_results, orient='index', columns=['fnr (naive)', 'fnr (correlated)', 'eff (naive)', 'eff (correlated)'])
    df_agg.index = pd.MultiIndex.from_tuples(df_agg.index, names=['prevalence', 'pool size'])
    df_agg = df_agg.reset_index()
    df_agg = df_agg.sort_values(by=['prevalence', 'pool size'])
    df_agg['sn (naive)'] = 1 - df_agg['fnr (naive)']
    df_agg['sn (correlated)'] = 1 - df_agg['fnr (correlated)']

    for prev in df_agg['prevalence'].unique():
        df = df_agg[df_agg['prevalence'] == prev]
        ax = df.sort_values(by='pool size').plot(x='sn (naive)', y = 'eff (naive)', sort_columns=True, color='mediumpurple', marker='^', style='--')
        df.sort_values(by='pool size').plot(x='sn (correlated)', y = 'eff (correlated)', sort_columns=True, ax=ax, color='mediumaquamarine', marker='o', style='-')

        texts = []
        for i, point in df.iterrows():
            texts.append(ax.text(point['sn (naive)'], point['eff (naive)'], str(int(point['pool size'])), color='dimgrey'))
            texts.append(ax.text(point['sn (correlated)'], point['eff (correlated)'], str(int(point['pool size'])), color='dimgrey'))

        adjust_text(texts, only_move={'points':'y', 'texts':'xy'})

        plt.legend(['naive', 'correlated'])
        plt.xlabel('Sensitivity = 1 - FNR')
        plt.ylabel('Efficiency')
        plt.title('Tradeoff between test efficiency and sensitivity\n under prevalence = {}'.format(prev))
        plt.grid(True, ls=':')
        plt.savefig('../figs/experiment_2/pareto_plots/pareto_for_prev_{}.pdf'.format(prev), format='pdf', dpi=600, bbox_inches='tight')
        plt.close()
    return


def generate_heatmap_plots():
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

    df_agg = pd.DataFrame.from_dict(aggregate_results, orient='index', columns=['fnr (naive)', 'fnr (correlated)', 'eff (naive)', 'eff (correlated)'])
    df_agg.index = pd.MultiIndex.from_tuples(df_agg.index, names=['prevalence', 'pool size'])
    df_agg = df_agg.reset_index()
    df_agg = df_agg.sort_values(by=['prevalence', 'pool size'])
    df_agg['sn (naive)'] = 1 - df_agg['fnr (naive)']
    df_agg['sn (correlated)'] = 1 - df_agg['fnr (correlated)']

    df_agg['sn diff'] = df_agg['sn (correlated)'] - df_agg['sn (naive)']
    df_agg['eff diff'] = df_agg['eff (correlated)'] - df_agg['eff (naive)']

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

    textcolors = ["k", "w"]
    threshold = 0.8
    for i, prev in enumerate(table_eff.index):
        for j, pool_size in enumerate(table_eff.columns):
            text = ax1.text(j+0.5, i+0.5, "{:.3f}".format(table_eff.iloc[i,j]).replace("0.", "."), \
                ha="center", va="center", color=textcolors[table_eff.iloc[i, j] > threshold], size=7)

    fig.tight_layout()
    fig.savefig('../figs/experiment_2/pareto_plots/heapmap_for_fnr_and_eff.pdf', format='pdf', dpi=600, bbox_inches='tight')
    plt.clf()
    return
    
def generate_test_consumption_results():
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

    df_agg = pd.DataFrame.from_dict(aggregate_results, orient='index', columns=['fnr (naive)', 'fnr (correlated)', 'eff (naive)', 'eff (correlated)'])
    df_agg.index = pd.MultiIndex.from_tuples(df_agg.index, names=['prevalence', 'pool size'])
    df_agg = df_agg.reset_index()
    df_agg = df_agg.sort_values(by=['prevalence', 'pool size'])
    df_agg['sn (naive)'] = 1 - df_agg['fnr (naive)']
    df_agg['sn (correlated)'] = 1 - df_agg['fnr (correlated)']

    df_results = pd.DataFrame(columns=['prevalence', 'opt pool size (naive)', 'opt sn * eff (naive)',
        'opt pool size (correlated)', 'opt sn * eff (correlated)', 'tests needed reduction'])

    for prev in df_agg['prevalence'].unique():
        df = df_agg[df_agg['prevalence'] == prev].reset_index()
        df['sn*eff (naive)'] = df['sn (naive)'] * df['eff (naive)']
        df['sn*eff (correlated)'] = df['sn (correlated)'] * df['eff (correlated)']

        opt_pool_size_naive = df['pool size'].iloc[df['sn*eff (naive)'].idxmax()]
        opt_sn_eff_prod_naive = df['sn*eff (naive)'].max()
        opt_pool_size_corr = df['pool size'].iloc[df['sn*eff (correlated)'].idxmax()]
        opt_sn_eff_prod_corr = df['sn*eff (correlated)'].max()

        test_needed_reduction = 1 - opt_sn_eff_prod_naive / opt_sn_eff_prod_corr
        # (1 / naive - 1 / corr) / (1 / corr) = corr / naive - 1 # increase when using NP
        # (1 / naive - 1 / corr) / (1 / naive) = 1 - naive/corr # reduction when using CP

        results = np.array([prev, opt_pool_size_naive,
            opt_sn_eff_prod_naive, opt_pool_size_corr, opt_sn_eff_prod_corr, test_needed_reduction]).round(3)
        df_results = df_results.append(dict(zip(df_results.columns, results)), ignore_index=True)

    df_results.to_csv('../results/experiment_2/opt_pool_size_test_reduction.csv', index=False)
    return


if __name__ == '__main__':
    # plt.rcParams["font.family"] = 'serif'
    #
    # filedir = "../results/experiment_2/sensitivity_analysis/results_prevalence=0.01_SAR=0.188_pool size=6_FNR=0.05_household dist=US.data"
    # with open(filedir) as f:
    #     results = np.loadtxt(f)
    # plot_hist_exp_2(results, 'nominal')
    #
    # for param in ['prevalence', 'pool size', 'SAR', 'FNR', 'household dist']:
    #     generate_sensitivity_plots(param)
    #
    # generate_pareto_fontier_plots()
    #
    # generate_heatmap_plots()

    generate_test_consumption_results()
