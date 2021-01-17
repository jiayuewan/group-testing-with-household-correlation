import numpy as np
import scipy.stats as st
import random
from collections import Counter
from eval_p_index import match_prevalence, eval_p_index, compute_household_infection_prob
from household_dist import HOUSEHOLD_DIST
from viral_load_distribution import sample_log10_viral_loads


def generate_correlated_infections_fixed_household_size(population_size, household_size, prevalence, SAR=0.3741):
    assert population_size % household_size == 0
    assert household_size <= 7

    household_dist = [0] * 7
    household_dist[household_size - 1] = 1
    p_index = eval_p_index(match_prevalence, prevalence, household_dist, SAR)
    num_households = population_size // household_size
    infections = st.bernoulli.rvs(p_index, size=(num_households, household_size))
    infected_households = np.sum(infections, axis=1)
    for i in range(num_households):
        for j in range(household_size):
            if infections[i, j] == 0 and infected_households[i] >= 1:
                infections[i, j] = st.bernoulli.rvs(SAR)
    return infections


def generate_correlated_infections(population_size, prevalence, type='binary', household_dist='US', SAR=0.3741):
    """
    generate a list of lists that describes the infection status of individual based on prevalence,
    household size distribution and second attack rate

    INPUT:
    population_size
    prevalence = population level prevalence
    household_dist = array-like, probability distribution of household sizes 1, 2, 3, ...
    SAR = household secondary attack rate
    """
    household_dist = HOUSEHOLD_DIST[household_dist]
    max_household_size = len(household_dist)
    p_household = compute_household_infection_prob(prevalence, household_dist, SAR)
    remaining_population_size = population_size
    households = []

    while remaining_population_size > 0:
        sampled_household_size = np.random.choice(max_household_size, p=household_dist) + 1
        sampled_household_size = min(sampled_household_size, remaining_population_size)
        infected = st.bernoulli.rvs(p_household)
        if infected:
            primary_idx = np.random.choice(sampled_household_size)
            sampled_infections = [st.bernoulli.rvs(SAR) if i != primary_idx else 1 for i in range(sampled_household_size)]
            if type == 'real':
                #sampled_infections = [sample_log10_viral_loads(n_samples=1)[0] if x == 1 else 0 for x in sampled_infections]
                sampled_infections = [a * b for a , b in zip(sampled_infections, sample_log10_viral_loads(n_samples=sampled_household_size))]
        else:
            sampled_infections = [0] * sampled_household_size

        households.append(sampled_infections)
        remaining_population_size -= sampled_household_size

    return households


def main():
    population_size = 100000
    prevalence = 0.1
    print("testing fixed household size...")
    sampled_households = generate_correlated_infections_fixed_household_size(population_size, 4, prevalence)
    total_num_infections = np.sum(sampled_households)
    print("total number of sampled infections among {} population under prevalence {} is {}".format(population_size,
        prevalence, total_num_infections))

    print("testing variable household size...")
    sampled_households = generate_correlated_infections(population_size, prevalence)
    total_num_infections = sum(sum(x) for x in sampled_households)
    household_sizes = Counter([len(x) for x in sampled_households])
    print("total number of sampled infections among {} population under prevalence {} is {}".format(population_size,
        prevalence, total_num_infections))
    print("sampled household size distribution is: " + str([(i, household_sizes[i] / len(sampled_households)) for i in household_sizes]))

    population_size = 20
    prevalence = 0.5

    print("testing viral loads for variable household size...")
    sampled_households = generate_correlated_infections(population_size, prevalence, type='real')
    print(sampled_households)


if __name__ == '__main__':
    main()
