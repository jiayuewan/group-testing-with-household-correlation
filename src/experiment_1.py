import matplotlib.pyplot as plt
import numpy as np
import pickle
from generate_infection_states import generate_correlated_infections_fixed_household_size
from one_stage_hierarchical_group_testing import one_stage_group_testing_fixed_household_size


def plot_hist(fnr_indep, fnr_correlated, prevalence, pool_size):
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
    plt.savefig('../figs/experiment_1/variable_household_fnr_diff_prev_{}_pool_size_{}.pdf'.format(prevalence, pool_size))
    plt.close()
    return


def simulation_fixed_household_size(population_size, household_size, pool_size, prevalence, num_iters=1000):
    fnr_group_testing_with_correlation = np.zeros(num_iters)
    fnr_group_testing_without_correlation = np.zeros(num_iters)

    print('running simulation for fixed household size under prevalence {}, pool size {} with {} iterations...'.format(prevalence, pool_size, num_iters))
    for i in range(num_iters):
        infections = generate_correlated_infections_fixed_household_size(population_size, household_size, prevalence)
        fnr_group_testing_without_correlation[i] = one_stage_group_testing_fixed_household_size(infections, pool_size, shuffle=True)[0]
        fnr_group_testing_with_correlation[i] = one_stage_group_testing_fixed_household_size(infections, pool_size, shuffle=False)[0]

    return fnr_group_testing_without_correlation, fnr_group_testing_with_correlation


def main():
    prevalences = [0.001, 0.005, 0.01, 0.05]
    pop_size = 12000
    household_size = 3
    num_iters = 500
    pool_sizes = [6, 12, 24]

    fnr_indep_results = np.zeros((len(prevalences), len(pool_sizes), num_iters))
    fnr_corr_results = np.zeros((len(prevalences), len(pool_sizes), num_iters))

    for i, p in enumerate(prevalences):
        for j, l in enumerate(pool_sizes):
            fnr_ind, fnr_corr = simulation_fixed_household_size(pop_size, household_size, l, p, num_iters)
            plot_hist(fnr_ind, fnr_corr, p, l)
            fnr_indep_results[i, j, :] = fnr_ind
            fnr_corr_results[i, j, :] = fnr_corr
            print('independent fnr = {}, correlated fnr = {}'.format(np.mean(fnr_ind), np.mean(fnr_corr)))

    with open('../results/experiment_1/fnr_indep.data', 'wb') as f:
        pickle.dump(fnr_indep_results, f)

    with open('../results/experiment_1/fnr_corr.data', 'wb') as f:
        pickle.dump(fnr_corr_results, f)


if __name__ == '__main__':
    main()
