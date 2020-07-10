import numpy as np
import scipy.stats as st
from eval_p_index import match_prevalence, eval_p_index


def generate_independent_infections(population_size, prevalence):
    return st.bernoulli.rvs(prevalence, size=population_size)


def generate_correlated_infections_fixed_household_size(population_size, household_size, prevalence, SAR=0.3741):
    assert population_size % household_size == 0
    assert household_size <= 7

    household_dist = [0] * 7
    household_dist[household_size] = 1
    p_index = eval_p_index(match_prevalence, 0.01, household_dist, SAR)
    num_households = population_size // household_size
    infections = st.bernoulli.rvs(prevalence, size=(num_households, household_size))
    infected_households = np.sum(infections, axis=1)
    for i in range(num_households):
        for j in range(household_size):
            if infections[i, j] == 0 and infected_households[i] >= 1:
                infections[i, j] = st.bernoulli.rvs(p_index)

    return infections


if __name__ == '__main__':
    pass
