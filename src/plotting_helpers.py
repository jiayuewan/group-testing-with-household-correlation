import matplotlib.pyplot as plt


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
