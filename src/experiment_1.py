import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import pickle
from generate_infection_states import generate_correlated_infections_fixed_household_size
from one_stage_hierarchical_group_testing import one_stage_group_testing_fixed_household_size
from plotting_helpers import plot_hist_exp_1


def simulation_fixed_household_size(population_size, household_size, pool_size, prevalence, num_iters=1000):
    fnr_group_testing_without_correlation = np.zeros(num_iters)
    fnr_group_testing_with_correlation = np.zeros(num_iters)

    results = np.zeros((num_iters, 6))

    print('running simulation for fixed household size = {} under prevalence {}, pool size {} with {} iterations...'.format(household_size, prevalence, pool_size, num_iters))
    for i in range(num_iters):
        infections = generate_correlated_infections_fixed_household_size(population_size, household_size, prevalence, SAR=0.188)
        fnr_indep, eff_indep, num_indiv_tests_per_positive_found_indep = one_stage_group_testing_fixed_household_size(infections, pool_size=pool_size, shuffle=True)[:3]
        fnr_corr, eff_corr, num_indiv_tests_per_positive_found_corr = one_stage_group_testing_fixed_household_size(infections, pool_size=pool_size, shuffle=False)[:3]
        results[i, :] = [fnr_indep, fnr_corr, eff_indep, eff_corr, num_indiv_tests_per_positive_found_indep, num_indiv_tests_per_positive_found_corr]
    return results


def main():
    prevalences = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.1]
    pop_size = 12000
    household_sizes = [3, 6]
    num_iters = 500
    #pool_sizes = [6, 12, 24]
    pool_sizes = [6]

    fnr_indep_results = np.zeros((len(prevalences), len(pool_sizes), num_iters))
    fnr_corr_results = np.zeros((len(prevalences), len(pool_sizes), num_iters))

    for p in prevalences:
        for m in pool_sizes:
            for k in household_sizes:
                results = simulation_fixed_household_size(population_size=pop_size, household_size=k, pool_size=m, prevalence=p, num_iters=num_iters)
                plot_hist_exp_1(results, household_size=k, pool_size=m, prevalence=p)
                avgs = np.mean(results, axis=0)
                print('indep fnr = {}, corr fnr = {}, indep eff = {}, corr eff = {}'.format(avgs[0], avgs[1], avgs[2], avgs[3]))

                with open('../results/experiment_1/results_pool-size={}_household-size={}_prevalence={}.data'.format(m, k, p), 'wb') as f:
                    np.savetxt(f, results)

    # with open('../results/experiment_1/fnr_corr.data', 'wb') as f:
    #     pickle.dump(fnr_corr_results, f)


if __name__ == '__main__':
    #main()

    filedir = "../results/experiment_1/results_pool-size=6_household-size=3_prevalence=0.01.data"
    with open(filedir) as f:
        results = np.loadtxt(f)

    print('results for FNR...')
    FNR = lambda n: 10 ** (-n)
    q = 0.188
    n = 6
    k = 3
    pFNR0 = FNR(1)
    pFNR1 = sum([FNR(1 + i) * st.binom.pmf(i, k-1, q) for i in range(k)])
    print('theoretically randomized pooling FNR = {}, correlated pooling FNR = {}, difference in FNR = {}'.format(pFNR0, pFNR1, pFNR0 - pFNR1))
    print('in experiments, the observed FNRs are {}, and the difference is {}'.format(np.mean(results[:, :2], axis=0), np.mean(np.subtract(results[:,0], results[:, 1]))))

    print('resutls for followup tests consumption...')
    print('theoretically the improvement in consumption of followup tests per positive identified is {0} or {1:.2%}'.format(n  - n / (1 + q * (k-1)),1 - 1/(1 + q * (k-1))))
    print('in reality, the aboslute saving is:', np.mean(np.subtract(results[:,4], results[:, 5])))
    print('in reality, the relative saving is:', 1 - np.mean(np.divide(results[:,5], results[:, 4])))
