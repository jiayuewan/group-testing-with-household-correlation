import numpy as np
import scipy.stats as st
from eval_p_index import match_prevalence, eval_p_index
import matplotlib.pyplot as plt


def generate_independent_infections(population_size, prevalence):
    return st.bernoulli.rvs(prevalence, size=population_size)


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

# def generate_correlated_infections(population_size, prevalence, household_dist=US_DIST, SAR=0.3741):


if __name__ == '__main__':
    population_size = 30000
    household_size = 3
    prevalences = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    for prevalence in prevalences:
        infections = generate_correlated_infections_fixed_household_size(population_size, household_size, prevalence)
        print('expected prevlance = {}, simulated prevalence = {}'.format(prevalence, np.sum(infections) / population_size))

    independent_infections = np.zeros(1000)
    correlated_infections = np.zeros(1000)
    for i in range(1000):
        independent_infections[i] = np.sum(generate_independent_infections(3000, 0.01))
        correlated_infections[i] = np.sum(generate_correlated_infections_fixed_household_size(3000, 3, 0.01))

    print('independent infections: ', np.mean(independent_infections), np.std(independent_infections))
    print('correlated infections: ', np.mean(correlated_infections), np.std(correlated_infections))
    plt.hist([independent_infections, correlated_infections], label=['independent', 'correlated'], alpha=0.5)
    plt.legend(loc='upper right')
    plt.show()
