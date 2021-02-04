import numpy as np
import scipy.stats as st
import random
from generate_infection_states import generate_correlated_infections, generate_correlated_infections_fixed_household_size
from PCR_test import false_negative_rate_binary, pooled_PCR_test


def one_stage_group_testing_fixed_household_size(infections, pool_size, shuffle=False):
    """
    perform one-stage hierarchical group testing on population with fixed househould size and binary infection state

    INPUT:
    infections: 2-d array of shape (m x k), the binary infection states for individuals
    where m is the number of households, k is the household size
    pool_size: Int, number of samples in a pooled test
    shuffle: whether to randomly place individuals into the pools or to assign pools based on households
    """
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
    results = np.array([st.bernoulli.rvs(1 - false_negative_rate_binary(n)) if n >= 1 else 0 for n in num_positives_in_pools])
    num_positive_pools = np.sum(results)
    num_false_negatives = np.sum(pools - results.reshape((-1,1)) == 1)
    num_positives = np.sum(pools)
    fnr_group_testing = num_false_negatives / num_positives

    num_indiv_tests = num_positive_pools * pool_size
    total_num_tests = num_pools + num_positive_pools * pool_size
    efficiency = population_size / total_num_tests
    num_positives_found = num_positives - num_false_negatives
    num_indiv_tests_per_positive_found = num_indiv_tests / num_positives_found

    return fnr_group_testing, efficiency, num_indiv_tests_per_positive_found


def one_stage_group_testing(infections, pool_size, type='binary', LoD=None, shuffle=False):
    """
    perform one-stage hierarchical group testing.

    INPUT:
    infections: list of lists, the infection states of the population
    pool_size: Int, number of samples in a pooled test
    type: 'binary' if the infection state is binary,
        'real' if the infection state is the individual's Log10 viral load
    LoD: limit of detection of the PCR test. Should only be specified if type == 'real'
    shuffle: boolean, whether to randomly place individuals into the pools or to assign pools based on households
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
        capacities = [pool_size] * num_pools # remaining capacity in each pool
        for household_infections in infections:
            household_size = len(household_infections)
            pool_idx = next((i for i, c in enumerate(capacities) if c >= household_size), -1)
            if pool_idx != -1:
                loc = pool_size - capacities[pool_idx]
                pools[pool_idx, loc:loc + household_size] = household_infections
                capacities[pool_idx] -= household_size
            else:
                last_pool = next(i for i, c in enumerate(np.cumsum(capacities)) if c >= household_size) # last pool for the household to be placed into
                allocated = 0
                for i in range(last_pool):
                    pools[i, pool_size - capacities[i]:] = household_infections[allocated:allocated + capacities[i]]
                    allocated += capacities[i]
                    capacities[i] = 0

                to_allocate_in_last_pool = household_size - allocated
                loc = pool_size - capacities[last_pool]
                pools[last_pool, loc: loc + to_allocate_in_last_pool] = household_infections[allocated:]
                capacities[last_pool] -= to_allocate_in_last_pool

    if type == "binary":
        num_positives_in_pools = np.sum(pools, axis=1)
        results = np.array([st.bernoulli.rvs(1 - false_negative_rate_binary(n)) if n >= 1 else 0 for n in num_positives_in_pools])
        num_false_negatives = np.sum(pools - results.reshape((-1,1)) == 1)
        num_positives = np.sum(pools)
        fnr_group_testing = num_false_negatives / num_positives
        return fnr_group_testing, num_positives_in_pools

    else:
        assert type == "real"
        convert_log10_viral_load = lambda x: int(10 ** x) if x > 0 else 0
        convert_log10_viral_load = np.vectorize(convert_log10_viral_load)
        viral_loads =  convert_log10_viral_load(pools)

        if LoD:
            group_testing_results = pooled_PCR_test(viral_loads, LoD=LoD)
        else:
            group_testing_results = pooled_PCR_test(viral_loads)
        num_positive_pools = np.sum(group_testing_results)
        group_testing_results = np.repeat(group_testing_results.reshape((-1,1)), pool_size, axis=1)

        if LoD:
            individual_testing_results = pooled_PCR_test(viral_loads, individual=True, LoD=LoD)
        else:
            individual_testing_results = pooled_PCR_test(viral_loads, individual=True)

        final_results = (group_testing_results  * individual_testing_results).astype(int)
        expected_outcomes = pools > 0
        num_false_negatives = np.sum(expected_outcomes - final_results == 1)
        num_positives = np.sum(expected_outcomes)
        fnr_group_testing = num_false_negatives / num_positives
        total_num_tests = num_pools + num_positive_pools * pool_size
        num_positives_in_pools = np.sum(pools > 0, axis=1)
        efficiency = population_size / total_num_tests
        return fnr_group_testing, efficiency, num_positives_in_pools


def main():
    print("testing one-stage group testing for fixed household size...")
    infections = generate_correlated_infections_fixed_household_size(10000, 4, 0.1)
    fnr_indep, eff_indep = one_stage_group_testing_fixed_household_size(infections, 20, shuffle=True)[:2]
    fnr_corr, eff_corr = one_stage_group_testing_fixed_household_size(infections, 20, shuffle=False)[:2]
    print('indep fnr = {}, corr fnr = {}, indep eff = {}, corr eff = {}'.format(fnr_indep, fnr_corr, eff_indep, eff_corr))

    # print("testing one-stage group testing for US household distribution with VL data...")
    # infections = generate_correlated_infections(12000, 0.01, type='real', SAR=0.188)
    # fnr_indep, num_tests_indep = one_stage_group_testing(infections, pool_size=6, LoD=174, type="real", shuffle=True)[:2]
    # fnr_correlated, num_tests_correlated = one_stage_group_testing(infections, pool_size=6, LoD=174, type="real", shuffle=False)[:2]
    # print('independent fnr = {}, correlated fnr = {}, num tests indep = {}, num tests corr = {}'.format(fnr_indep, fnr_correlated, num_tests_indep, num_tests_correlated))
    # return

if __name__ == '__main__':
    main()
