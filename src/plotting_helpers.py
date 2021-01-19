import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
#import seaborn as sns
import re


def plot_hist_exp_1(results, household_size, pool_size, prevalence):
    fnr_indep = results[:, 0]
    fnr_correlated = results[:, 1]
    eff_indep = results[:, 2]
    eff_correlated = results[:, 3]

    plt.hist([fnr_indep, fnr_correlated], label=['independent group testing', 'correlated group testing'], alpha=0.5)
    plt.legend(loc='upper right')
    plt.xlabel('False negative rate')
    plt.ylabel('Frequency')
    plt.title('Histogram of FNR values for one-stage group testing \n under household size = {}, pool size = {}, prevalence = {}'.format(household_size, pool_size, prevalence))
    plt.savefig('../figs/experiment_1/fnr_pool-size={}_household-size={}_prevalence={}.pdf'.format(pool_size, household_size, prevalence))
    plt.close()

    plt.hist([eff_indep, eff_correlated], label=['independent group testing', 'correlated group testing'], alpha=0.5)
    plt.legend(loc='upper right')
    plt.xlabel('Efficiency')
    plt.ylabel('Frequency')
    plt.title('Histogram of testing efficiency for one-stage group testing \n under household size = {}, pool size = {}, prevalence = {}'.format(household_size, pool_size, prevalence))
    plt.savefig('../figs/experiment_1/eff_pool-size={}_household-size={}_prevalence={}.pdf'.format(pool_size, household_size, prevalence))
    plt.close()
    return


def plot_hist_exp_2(results, param, val=None):
    fnr_indep = results[:, 0]
    fnr_correlated = results[:, 1]
    eff_indep = results[:, 2]
    eff_correlated = results[:, 3]

    #num_iters = size(fnr_indep)

    plt.hist([fnr_indep, fnr_correlated], label=['independent group testing', 'correlated group testing'], alpha=0.5)
    plt.legend(loc='upper right')
    plt.xlabel('False negative rate')
    plt.ylabel('Frequency')
    if param == 'nominal':
        plt.title('Histogram of FNR values for one-stage group testing \n under {} scenario'.format(param))
        plt.savefig('../figs/experiment_2/fnr_{}_scenario.pdf'.format(param))
    else:
        plt.title('Histogram of FNR values for one-stage group testing \n under {} = {}'.format(param, val))
        plt.savefig('../figs/experiment_2/fnr_{}={}.pdf'.format(param, val))
    plt.close()

    plt.hist([eff_indep, eff_correlated], label=['independent group testing', 'correlated group testing'], alpha=0.5)
    plt.legend(loc='upper right')
    plt.xlabel('Efficiency')
    plt.ylabel('Frequency')
    if param == 'nominal':
        plt.title('Histogram of testing efficiency for one-stage group testing \n under {} scenario'.format(param))
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

    df = pd.DataFrame({'indep FNR': fnr_indep, 'corr FNR': fnr_corr, 'indep efficiency': eff_indep,'corr efficiency': eff_corr}, index=index)
    df = df.sort_index()
    #print(df)
    fig, axes = plt.subplots(2, 1, sharex=True)
    df[['indep FNR', 'corr FNR']].plot.bar(ax = axes[0], xlabel=param, ylabel='FNR')
    df[['indep efficiency', 'corr efficiency']].plot.bar(ax = axes[1], xlabel=param, ylabel='efficiency')
    plt.savefig('../figs/experiment_2/sensitivity_plots/tmp_sensitivity_for_{}.pdf'.format(param))
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

    df_agg = pd.DataFrame.from_dict(aggregate_results, orient='index', columns=['indep fnr', 'corr fnr', 'indep eff', 'corr eff'])
    df_agg.index = pd.MultiIndex.from_tuples(df_agg.index, names=['prevalence', 'pool size'])
    df_agg = df_agg.reset_index()
    df_agg = df_agg.sort_values(by=['prevalence', 'pool size'])

    for prev in df_agg['prevalence'].unique():
        df = df_agg[df_agg['prevalence'] == prev]
        ax = df.sort_values(by='pool size').plot(x='indep fnr', y = 'indep eff', sort_columns=True)
        df.sort_values(by='pool size').plot(x='corr fnr', y = 'corr eff', sort_columns=True, ax=ax)

        for i, point in df.iterrows():
            ax.text(point['indep fnr'], point['indep eff'], str(int(point['pool size'])))
            ax.text(point['corr fnr'], point['corr eff'], str(int(point['pool size'])))
        plt.legend(['indep', 'corr'])
        plt.xlabel('fnr')
        plt.ylabel('eff')
        plt.title('eff vs fnr, indep vs corr for prevalence={}'.format(prev))

        plt.savefig('../figs/experiment_2/pareto_plots/tmp_pareto_for_prev_{}.pdf'.format(prev))
        plt.close()

    df_agg['fnr diff'] = df_agg['indep fnr'] - df_agg['corr fnr']
    df_agg['eff diff'] = df_agg['corr eff'] - df_agg['indep eff']

    table_fnr = pd.pivot_table(df_agg, values='fnr diff', index=['prevalence'], columns=['pool size'])
    heatmap = plt.pcolor(table_fnr)
    plt.yticks(np.arange(0.5, len(table_fnr.index), 1), table_fnr.index)
    plt.xticks(np.arange(0.5, len(table_fnr.columns), 1), table_fnr.columns)
    plt.xlabel('pool size')
    plt.ylabel('prevalence')
    plt.title('heat map for (indep fnr - corr fnr)')
    plt.colorbar(heatmap)
    plt.savefig('../figs/experiment_2/pareto_plots/tmp_heapmap_for_fnr.pdf')
    plt.close()

    table_eff = pd.pivot_table(df_agg, values='eff diff', index=['prevalence'], columns=['pool size'])
    heatmap = plt.pcolor(table_eff)
    plt.yticks(np.arange(0.5, len(table_eff.index), 1), table_eff.index)
    plt.xticks(np.arange(0.5, len(table_eff.columns), 1), table_eff.columns)
    plt.xlabel('pool size')
    plt.ylabel('prevalence')
    plt.title('heat map for (corr eff - indep eff)')
    plt.colorbar(heatmap)
    plt.savefig('../figs/experiment_2/pareto_plots/tmp_heapmap_for_eff.pdf')
    plt.close()
    return


if __name__ == '__main__':
    #run_simulations_for_sensitivity_analysis()

    # for param in ['prevalence', 'pool size', 'SAR', 'FNR', 'household dist']:
    #     generate_sensitivity_plots(param)

    generate_pareto_fontier_plots()
