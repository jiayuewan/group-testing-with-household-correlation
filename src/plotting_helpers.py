import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


def plot_hist_exp_1(fnr_indep, fnr_correlated, prevalence, pool_size):
    assert len(fnr_indep) == len(fnr_correlated)

    num_iters = len(fnr_indep)
    plt.hist([fnr_indep, fnr_correlated], label=['independent group testing', 'correlated group testing'], alpha=0.5)
    plt.legend(loc='upper right')
    plt.xlabel('False negative rate')
    plt.ylabel('Frequency')
    plt.title('Histogram of FNR values for one-stage group testing \n under prevalence = {}'.format(prevalence))
    plt.savefig('../figs/experiment_1/fnr_prev_{}_pool_size_{}.pdf'.format(prevalence, pool_size))
    plt.close()

    plt.hist(fnr_correlated -  fnr_indep)
    plt.xlabel(r"$FNR_{corr} - FNR_{indep}$")
    plt.ylabel('Frequency')
    plt.title('difference in false negative rate between correlated \n and independent group testing under prevalence {}'.format(prevalence))
    plt.savefig('../figs/experiment_1/fnr_diff_prev_{}_pool_size_{}.pdf'.format(prevalence, pool_size))
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
    fig, axes = plt.subplots(2, 1)
    df[['indep FNR', 'corr FNR']].plot.bar(ax = axes[0])
    df[['indep efficiency', 'corr efficiency']].plot.bar(ax = axes[1])
    plt.savefig('../figs/experiment_2/sensitivity_plots/tmp_sensitivity_for_{}.pdf'.format(param))
    return


def generate_pareto_fontier_plots():
    return
