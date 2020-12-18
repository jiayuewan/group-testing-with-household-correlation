import numpy as np
import scipy.stats as st
import random
from eval_p_index import US_DIST
from generate_infection_states import generate_correlated_infections
import matplotlib.pyplot as plt


def false_negative_rate(num_positives, type='exp'):
    """
    computes the false negative rate (FNR) of a pooled PCR test, based on the number of
    positive samples in the pooled sample

    INPUT:
    num_positives = number of positive samples in the pooled sample
    type = false negative rate function used in the calculation
    """
    assert num_positives >= 1

    if type == 'exp':
        return 0.1 ** num_positives

    elif type == 'reciprocal':
        return 1 / (1 + num_positives)

    else:
        assert type == 'step'
        return 1 - (num_positives >= 3)


# decprecated; use one_stage_group_testing with household_size="fixed" instead
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
    num_ones = np.sum(num_positives_in_pools == 1)
    results = np.array([st.bernoulli.rvs(1 - false_negative_rate(n)) if n >= 1 else 0 for n in num_positives_in_pools])
    num_false_negatives = np.sum(pools - results.reshape((-1,1)) == 1)
    num_positives = np.sum(pools)
    fnr_group_testing = num_false_negatives / num_positives
    return fnr_group_testing, num_positives_in_pools


# two proposals for implementing one-stage hierarchical group testing
# proposal 1: sort the households in random order, put them in second pool if full
# proposal 2: have a collection have a collection of pools not yet filled;
#             look for the first unfinished pool that has enoughspace
# both can be done.
# for independent testing; still use random shuffle
def one_stage_group_testing(infections, pool_size, household_size="variable", shuffle=False):
    """
    perform one-stage hierarchical group testing.

    INPUT:
    infections: infection states of the population, either a 2-d array (when household_size="fixed")
    or a list of lists (when household_size="variable")
    pool_size: Int, number of samples in a pooled test
    household_size: "variable" if household sizes vary in the population,
    "fixed" if household sizes are the same across the population
    shuffle: whether to randomly place individuals into the pools or to assign pools based on households
    """
    population_size = len(sum(infections,[]))
    assert population_size % pool_size == 0

    num_pools = population_size // pool_size

    if household_size == "fixed":
        infections = infections.tolist()

    if shuffle:
        pools = np.array(sum(infections,[]))
        np.random.shuffle(pools)
        pools = pools.reshape((num_pools, -1))
    else:
        pools = np.zeros((num_pools, pool_size))
        capacities = [pool_size] * num_pools
        for household_infections in infections:
            household_size = len(household_infections)
            pool_idx = next((i for i, c in enumerate(capacities) if c >= household_size), -1)
            if pool_idx != -1:
                start = pool_size - capacities[pool_idx]
                pools[pool_idx, start:start + household_size] = household_infections
                capacities[pool_idx] -= household_size
            else:
                last_pool = next(i for i, c in enumerate(np.cumsum(capacities)) if c >= household_size)
                current = 0
                for i in range(last_pool):
                    pools[i, pool_size - capacities[i]:] = household_infections[current:current + capacities[i]]
                    current += capacities[i]
                    capacities[i] = 0

                to_allocate = household_size - current
                start = pool_size - capacities[last_pool]
                pools[last_pool, start: start + to_allocate] = household_infections[current:]
                capacities[last_pool] -= to_allocate

    num_positives_in_pools = np.sum(pools, axis=1)
    results = np.array([st.bernoulli.rvs(1 - false_negative_rate(n)) if n >= 1 else 0 for n in num_positives_in_pools])
    num_false_negatives = np.sum(pools - results.reshape((-1,1)) == 1)
    num_positives = np.sum(pools)
    fnr_group_testing = num_false_negatives / num_positives
    return fnr_group_testing, num_positives_in_pools


def simulation_fixed_household_size(population_size, household_size, pool_size, prevalence, num_iters=100):
    fnr_group_testing_with_correlation = np.zeros(num_iters)
    fnr_group_testing_without_correlation = np.zeros(num_iters)

    print('running simulation for fixed household size under prevalence {} with {} iterations...'.format(prevalence, num_iters))
    for i in range(num_iters):
        infections = generate_correlated_infections_fixed_household_size(population_size, household_size, prevalence)
        fnr_group_testing_without_correlation[i] = one_stage_group_testing_fixed_household_size(infections, pool_size, shuffle=True)[0]
        fnr_group_testing_with_correlation[i] = one_stage_group_testing_fixed_household_size(infections, pool_size, shuffle=False)[0]

    return fnr_group_testing_without_correlation, fnr_group_testing_with_correlation


def simulation_variable_household_size(population_size, pool_size, prevalence, household_dist=US_DIST, SAR=0.3741, num_iters=100):
    fnr_group_testing_with_correlation = np.zeros(num_iters)
    fnr_group_testing_without_correlation = np.zeros(num_iters)

    print('running simulation for population size {} under prevalence {} with {} iterations...'.format(population_size, prevalence, num_iters))
    for i in range(num_iters):
        infections = generate_correlated_infections(population_size, prevalence, household_dist=household_dist, SAR=SAR)

        fnr_group_testing_without_correlation[i] = one_stage_group_testing(infections, pool_size, household_size="variable", shuffle=True)[0]
        fnr_group_testing_with_correlation[i] = one_stage_group_testing(infections, pool_size, household_size="variable", shuffle=False)[0]

    return fnr_group_testing_without_correlation, fnr_group_testing_with_correlation


def plot_hist(fnr_indep, fnr_correlated, prevalence):
    assert len(fnr_indep) == len(fnr_correlated)
    num_iters = len(fnr_indep)
    plt.hist([fnr_indep, fnr_correlated], label=['independent group testing', 'correlated group testing'], alpha=0.5)
    plt.legend(loc='upper right')
    plt.xlabel('False negative rate')
    plt.ylabel('Frequency')
    plt.title('Histogram of FNR values for one-stage group testing \n under prevalence = {}'.format(prevalence))
    plt.savefig('../figs/variable_household_prev_{}.pdf'.format(prevalence))
    plt.close()

    plt.hist(fnr_correlated -  fnr_indep)
    plt.xlabel(r"$FNR_{corr} - FNR_{indep}$")
    plt.ylabel('Frequency')
    plt.title('difference in false negative rate between correlated \n and independent group testing under prevalence {}'.format(prevalence))
    plt.savefig('../figs/variable_household_fnr_diff_prev_{}.pdf'.format(prevalence))
    plt.close()
    return


def main():
    # for prevalence in [0.005, 0.01, 0.05, 0.1, 0.2]:
    #     fnr_indep, fnr_correlated = simulation_fixed_household_size(3000, 3, 30, prevalence, 1000)
    #     print('independent fnr = {}, correlated fnr = {}'.format(np.mean(fnr_indep), np.mean(fnr_correlated)))
    #     plot_hist(fnr_indep, fnr_correlated, prevalence)
    # fnr_indep, fnr_correlated = simulation_fixed_household_size(3000, 2, 10, 0.5, 1500)
    # print('independent fnr = {}, correlated fnr = {}'.format(np.mean(fnr_indep), np.mean(fnr_correlated)))
    # plot_hist(fnr_indep, fnr_correlated, 0.5)
    fnr_indep, fnr_correlated = simulation_variable_household_size(10000, 10, 0.01, num_iters=100)
    print('independent fnr = {}, correlated fnr = {}'.format(np.mean(fnr_indep), np.mean(fnr_correlated)))
    plot_hist(fnr_indep, fnr_correlated, 0.01)

if __name__ == '__main__':
    main()
