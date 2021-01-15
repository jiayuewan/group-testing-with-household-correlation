import matplotlib.pyplot as plt
import numpy as np
import pickle
from household_dist import US_DIST, CN_DIST, AUS_DIST, FR_DIST
from generate_infection_states import generate_correlated_infections
from one_stage_hierarchical_group_testing import one_stage_group_testing


def plot_hist(fnr_indep, fnr_correlated, prevalence, pool_size):
    assert len(fnr_indep) == len(fnr_correlated)

    num_iters = len(fnr_indep)
    plt.hist([fnr_indep, fnr_correlated], label=['independent group testing', 'correlated group testing'], alpha=0.5)
    plt.legend(loc='upper right')
    plt.xlabel('False negative rate')
    plt.ylabel('Frequency')
    plt.title('Histogram of FNR values for one-stage group testing \n under prevalence = {}'.format(prevalence))
    plt.savefig('../figs/experiment_2/fnr_prev_{}_pool_size_{}.pdf'.format(prevalence, pool_size))
    plt.close()

    plt.hist(fnr_correlated -  fnr_indep)
    plt.xlabel(r"$FNR_{corr} - FNR_{indep}$")
    plt.ylabel('Frequency')
    plt.title('difference in false negative rate between correlated \n and independent group testing under prevalence {}'.format(prevalence))
    plt.savefig('../figs/experiment_2/fnr_diff_prev_{}_pool_size_{}.pdf'.format(prevalence, pool_size))
    plt.close()
    return


def simulation_variable_household_size(population_size, pool_size, prevalence, household_dist=US_DIST, SAR=0.3741, num_iters=1000):
    fnr_group_testing_with_correlation = np.zeros(num_iters)
    fnr_group_testing_without_correlation = np.zeros(num_iters)
    eff_group_testing_with_correlation = np.zeros(num_iters)
    eff_group_testing_without_correlation = np.zeros(num_iters)

    print('running simulation for population size {} under prevalence {} with {} iterations...'.format(population_size, prevalence, num_iters))
    for i in range(num_iters):
        infections = generate_correlated_infections(population_size, prevalence, household_dist=household_dist, SAR=SAR)

        fnr_group_testing_without_correlation[i] = one_stage_group_testing(infections, pool_size, type="binary", shuffle=True)[0]
        fnr_group_testing_with_correlation[i] = one_stage_group_testing(infections, pool_size, household_size="variable", shuffle=False)[0]

    return fnr_group_testing_without_correlation, fnr_group_testing_with_correlation


# consider nominal;
# sensitivity analyses: for each, plot for fnr_indep and fnr_corr vs one parameter; plot for efficiency indep + corr vs one parameter
# pareto fontier
# add LoD to the parameter usign args?
# save all the simulation results based on param values
