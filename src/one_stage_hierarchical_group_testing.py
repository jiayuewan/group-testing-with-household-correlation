import numpy as np
import scipy.stats as st
import random
from generate_infection_states import generate_independent_infections, generate_correlated_infections_fixed_household_size
import matplotlib.pyplot as plt


def false_negative_rate(num_positives, type='exp'):
    assert num_positives >= 1
    if type == 'exp':
        return 0.1 ** num_positives


def one_stage_group_testing_fixed_household_size(infections, pool_size):
    population_size = infections.size
    assert population_size % pool_size == 0

    num_pools = population_size // pool_size

    pools = infections.reshape((num_pools, -1))
    num_positives_in_pools = np.sum(pools, axis=1)
    results = np.array([st.bernoulli.rvs(1 - false_negative_rate(n)) if n >= 1 else 0 for n in num_positives_in_pools])
    num_false_negatives = np.sum(pools - results.reshape((-1,1)) == 1)
    num_positives = np.sum(pools)
    fnr_group_testing = num_false_negatives / num_positives
    return fnr_group_testing


def simulation_fixed_household_size(population_size, household_size, pool_size, prevalence, num_iters=100):
    fnr_group_testing_indep_infections = np.zeros(num_iters)
    fnr_group_testing_correlated_infections = np.zeros(num_iters)

    print('running simulation for fixed household size with {} iterations...'.format(num_iters))
    for i in range(num_iters):
        indep_infections = generate_independent_infections(population_size, prevalence)
        fnr_group_testing_indep_infections[i] = one_stage_group_testing_fixed_household_size(indep_infections, pool_size)

        correlated_infections = generate_correlated_infections_fixed_household_size(population_size, household_size, prevalence)
        fnr_group_testing_correlated_infections[i] = one_stage_group_testing_fixed_household_size(correlated_infections, pool_size)


    return fnr_group_testing_indep_infections, fnr_group_testing_correlated_infections


def plot_hist(fnr_indep, fnr_correlated, prevalence):
    plt.hist([fnr_indep, fnr_correlated], label=['independent infections', 'correlated infections'], alpha=0.5)
    plt.legend(loc='upper right')
    plt.xlabel('False negative rate')
    plt.ylabel('Incidence')
    plt.title('Distribution of FNR values for random and correlated group testing \n (prevalence {})'.format(prevalence))
    plt.show()
    #plt.savefig('../figs/')
    return


if __name__ == '__main__':
    prevalence = 0.01
    fnr_indep, fnr_correlated = simulation_fixed_household_size(3000, 3, 30, prevalence, 1000)
    print(np.mean(fnr_indep), np.mean(fnr_correlated))
    plot_hist(fnr_indep, fnr_correlated, prevalence)
