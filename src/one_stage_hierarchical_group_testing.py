import numpy as np
import scipy.stats as st
import random
from household_dist import US_DIST
from generate_infection_states import generate_correlated_infections, generate_correlated_infections_fixed_household_size
from pcr_test import false_negative_rate_binary, pooled_PCR_test


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
    results = np.array([st.bernoulli.rvs(1 - false_negative_rate_binary(n)) if n >= 1 else 0 for n in num_positives_in_pools])
    num_false_negatives = np.sum(pools - results.reshape((-1,1)) == 1)
    num_positives = np.sum(pools)
    fnr_group_testing = num_false_negatives / num_positives
    return fnr_group_testing, num_positives_in_pools


# two proposals for implementing one-stage hierarchical group testing
# proposal 1: sort the households in random order, put them in second pool if full
# proposal 2: have a collection have a collection of pools not yet filled;
#             look for the first unfinished pool that has enough space
# both can be done.
# for independent testing; still use random shuffle
def one_stage_group_testing(infections, pool_size, type='binary', shuffle=False):
    """
    perform one-stage hierarchical group testing.

    INPUT:
    infections: infection states of the population, a list of lists
    pool_size: Int, number of samples in a pooled test
    test: 'binary' if the infection state is binary, 'real' if the infection state is the individual's Log10 viral load
    shuffle: whether to randomly place individuals into the pools or to assign pools based on households
    """
    population_size = len(sum(infections,[]))
    assert population_size % pool_size == 0

    num_pools = population_size // pool_size

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

    if type == "binary":
        num_positives_in_pools = np.sum(pools, axis=1)
        results = np.array([st.bernoulli.rvs(1 - false_negative_rate_binary(n)) if n >= 1 else 0 for n in num_positives_in_pools])
        num_false_negatives = np.sum(pools - results.reshape((-1,1)) == 1)
        num_positives = np.sum(pools)
        fnr_group_testing = num_false_negatives / num_positives
    else:
        assert type == "real"
        convert_log10_viral_load = lambda x: int(10**x) if x > 0 else 0
        convert_log10_viral_load = np.vectorize(convert_log10_viral_load)
        viral_loads =  convert_log10_viral_load(pools)
        has_positives_in_pools = np.sum(pools, axis=1) > 0
        results = np.array([pooled_PCR_test(viral_loads[i, :]) if has_positives_in_pools[i] else 0 for i in range(num_pools)])
        expected_outcomes = pools > 0
        num_positives_in_pools = np.sum(expected_outcomes, axis=1)
        #print(expected_outcomes)
        print(num_positives_in_pools)
        num_false_negatives = np.sum(expected_outcomes - results.reshape((-1,1)) == 1)
        num_positives = np.sum(expected_outcomes)
        fnr_group_testing = num_false_negatives / num_positives

    return fnr_group_testing, num_positives_in_pools


def main():
    # pop_size = 12000
    # household_size = 4
    # prevalence = 0.05

    # print("testing one-stage group testing for fixed household size...")
    # infections = generate_correlated_infections_fixed_household_size(pop_size, household_size, prevalence)
    # fnr_indep = one_stage_group_testing_fixed_household_size(infections, pool_size=12, shuffle=True)[0]
    # fnr_correlated = one_stage_group_testing_fixed_household_size(infections, pool_size=12, shuffle=False)[0]
    # print('independent fnr = {}, correlated fnr = {}'.format(fnr_indep, fnr_correlated))
    #
    # print("testing one-stage group testing for variable household size...")
    # infections = generate_correlated_infections(pop_size, prevalence)
    # fnr_indep = one_stage_group_testing(infections, pool_size=10, shuffle=True)[0]
    # fnr_correlated = one_stage_group_testing(infections, pool_size=10, shuffle=False)[0]
    # print('independent fnr = {}, correlated fnr = {}'.format(fnr_indep, fnr_correlated))

    print("testing one-stage group testing for variable household size with VL data...")
    infections = generate_correlated_infections(10000, 0.01, type='real')
    #print(infections)
    print("test")
    fnr_indep = one_stage_group_testing(infections, pool_size=20, type="real", shuffle=True)[0]
    fnr_correlated = one_stage_group_testing(infections, pool_size=20, type="real", shuffle=False)[0]
    print('independent fnr = {}, correlated fnr = {}'.format(fnr_indep, fnr_correlated))



if __name__ == '__main__':
    main()
