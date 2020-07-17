import numpy as np
import scipy.stats as st
import random
from generate_infection_states import generate_correlated_infections_fixed_household_size, generate_independent_infections
import matplotlib.pyplot as plt


def false_negative_rate(num_positives, type='exp'):
    assert num_positives >= 1
    if type == 'exp':
        return 0.1 ** num_positives


def one_stage_group_testing_fixed_household_size(infections, pool_size, shuffle=False):
    population_size = infections.size
    assert population_size % pool_size == 0

    num_pools = population_size // pool_size

    if shuffle:
        pools = infections.flatten()
        np.random.shuffle(pools)
        pools = pools.reshape((num_pools, -1))
    else:
        pools = infections.reshape((num_pools, -1))

    num_positives_in_pools = np.sum(pools, axis=1)
    results = np.array([st.bernoulli.rvs(1 - false_negative_rate(n)) if n >= 1 else 0 for n in num_positives_in_pools])
    num_false_negatives = np.sum(pools - results.reshape((-1,1)) == 1)
    num_positives = np.sum(pools)
    fnr_group_testing = num_false_negatives / num_positives
    return fnr_group_testing


def simulation_fixed_household_size(population_size, household_size, pool_size, prevalence, num_iters=100):
    fnr_group_testing_with_correlation = np.zeros(num_iters)
    fnr_group_testing_without_correlation = np.zeros(num_iters)

    print('running simulation for fixed household size under prevalence {} with {} iterations...'.format(prevalence, num_iters))
    for i in range(num_iters):
        infections = generate_correlated_infections_fixed_household_size(population_size, household_size, prevalence)
        fnr_group_testing_without_correlation[i] = one_stage_group_testing_fixed_household_size(infections, pool_size, shuffle=True)
        fnr_group_testing_with_correlation[i] = one_stage_group_testing_fixed_household_size(infections, pool_size, shuffle=False)

    return fnr_group_testing_without_correlation, fnr_group_testing_with_correlation


def plot_hist(fnr_indep, fnr_correlated, prevalence):
    assert len(fnr_indep) == len(fnr_correlated)
    num_iters = len(fnr_indep)
    plt.hist([fnr_indep, fnr_correlated], label=['independent group testing', 'correlated group testing'], alpha=0.5)
    plt.legend(loc='upper right')
    plt.xlabel('False negative rate')
    plt.ylabel('Frequency')
    plt.title('Histogram of FNR values for one-stage group testing \n with fixed household size under prevalence = {}'.format(prevalence))
    plt.savefig('../figs/fixed_household_prev_{}.pdf'.format(prevalence))
    plt.close()
    return


if __name__ == '__main__':
    for prevalence in [0.005, 0.01, 0.05, 0.1, 0.2]:
        fnr_indep, fnr_correlated = simulation_fixed_household_size(3000, 3, 30, prevalence, 1000)
        print('independent fnr = {}, correlated fnr = {}'.format(np.mean(fnr_indep), np.mean(fnr_correlated)))
        plot_hist(fnr_indep, fnr_correlated, prevalence)
