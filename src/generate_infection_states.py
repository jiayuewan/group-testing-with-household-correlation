import numpy as np
import scipy.stats as st
from eval_p_index import match_prevalence, eval_p_index, compute_household_infection_prob, US_DIST
import random
import matplotlib.pyplot as plt
from collections import Counter


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


# generate list of households, may or maynot be efficient
# proposal 1: maintain one list of infection status; maintain a second list indicating the first index of each house hold
# proposal 2 (yes): maintain a list of lists, each inner list represents a household
def generate_correlated_infections(population_size, prevalence, household_dist=US_DIST, SAR=0.3741):
    """
    generate a list of binary lists that describes the infection status of individual based on prevalence,
    household size distribution and second attack rate

    INPUT:
    population_size
    prevalence = population level prevalence
    household_dist = array-like, probability distribution of household sizes 1, 2, 3, ...
    SAR = household secondary attack rate
    """
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
        else:
            sampled_infections = [0] * sampled_household_size

        households.append(sampled_infections)
        remaining_population_size -= sampled_household_size

    return households


if __name__ == '__main__':
    # population_size = 30000
    # household_size = 3
    # prevalences = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    # for prevalence in prevalences:
    #     infections = generate_correlated_infections_fixed_household_size(population_size, household_size, prevalence)
    #     print('expected prevlance = {}, simulated prevalence = {}'.format(prevalence, np.sum(infections) / population_size))
    #
    # independent_infections = np.zeros(1000)
    # correlated_infections = np.zeros(1000)
    # for i in range(1000):
    #     independent_infections[i] = np.sum(generate_independent_infections(3000, 0.01))
    #     correlated_infections[i] = np.sum(generate_correlated_infections_fixed_household_size(3000, 3, 0.01))
    #
    # print('independent infections: ', np.mean(independent_infections), np.std(independent_infections))
    # print('correlated infections: ', np.mean(correlated_infections), np.std(correlated_infections))
    # plt.hist([independent_infections, correlated_infections], label=['independent', 'correlated'], alpha=0.5)
    # plt.legend(loc='upper right')
    # plt.show()
    population_size = 10000
    sampled_households = generate_correlated_infections(population_size, 0.5, household_dist=US_DIST)
    total_num_infections = sum(sum(x) for x in sampled_households)
    household_sizes = Counter([len(x) for x in sampled_households])
    print("total number of sampled infections among {} population is {}".format(population_size, total_num_infections))
    print("sampled household size distribution is: " + str([(i, household_sizes[i] / len(sampled_households)) for i in household_sizes]))
