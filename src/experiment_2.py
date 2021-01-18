import matplotlib.pyplot as plt
import numpy as np
import pickle
from household_dist import HOUSEHOLD_DIST
from generate_infection_states import generate_correlated_infections
from one_stage_hierarchical_group_testing import one_stage_group_testing
from plotting_helpers import plot_hist_exp_2, generate_sensitivity_plots


NOMINAL_PARAMS = {
"pool size": 6,
"prevalence": 0.01,
"household dist": 'US',
"SAR": 0.188,
"FNR": 0.05
}


def simulation_variable_household_size(population_size, params=NOMINAL_PARAMS, num_iters=500):
    # simulation results: fnr_indep, fnr_corr, efficiency_indep, efficiency_corr
    pool_size = params['pool size']
    prevalence = params['prevalence']
    household_dist = params['household dist']
    SAR = params['SAR']
    FNR = params['FNR']

    if not FNR in [0.025, 0.05, 0.1]:
        raise ValueError("supported FNR values do not include {}".format(FNR))
    else:
        LoD = 108 if FNR == 0.025 else 174 if FNR == 0.05 else 345

    print('running simulation under r = {}, n = {}, FNR = {}, SAR = {}, dist = {} with {} iterations...'.format(prevalence, pool_size, FNR, SAR, household_dist, num_iters))
    results = np.zeros((num_iters, 4))
    for i in range(num_iters):
        infections = generate_correlated_infections(population_size, prevalence, type='real', household_dist=household_dist, SAR=SAR)

        fnr_indep, eff_indep = one_stage_group_testing(infections, pool_size=pool_size, LoD=LoD, type="real", shuffle=True)[:2]
        fnr_correlated, eff_corr = one_stage_group_testing(infections, pool_size=pool_size, LoD=LoD, type="real", shuffle=False)[:2]
        results[i, :] = [fnr_indep, fnr_correlated, eff_indep, eff_corr]

    return results


def run_simulations_for_sensitivity_analysis():
    pop_size = 12000
    num_iters = 500
    houshold_dists = list(HOUSEHOLD_DIST.keys())
    houshold_dists.remove('US')

    configs = {
    'prevalence' :[0.001, 0.005, 0.05, 0.1],
    'SAR' : [0.039, 0.154, 0.222, 0.446],
    'pool size': [3, 12, 24],
    'FNR': [0.025, 0.1],
    'household dist': houshold_dists
    }

    results = simulation_variable_household_size(pop_size, params=NOMINAL_PARAMS, num_iters=num_iters)
    plot_hist_exp_2(results, "nominal")
    avgs = np.mean(results, axis=0)
    print('indep fnr = {}, corr fnr = {}, indep eff = {}, corr eff = {}'.format(avgs[0], avgs[1], avgs[2], avgs[3]))
    nominal_filedir = '../results/experiment_2/sensitivity_analysis/results_prevalence={}_SAR={}_pool size={}_FNR={}_household dist={}.data'.format(\
        NOMINAL_PARAMS['prevalence'], NOMINAL_PARAMS['SAR'], NOMINAL_PARAMS['pool size'], NOMINAL_PARAMS['FNR'], NOMINAL_PARAMS['household dist'])
    with open(nominal_filedir, 'wb') as f:
        np.savetxt(f, results)

    for param, values in configs.items():
        for val in values:
            params = NOMINAL_PARAMS.copy()
            params[param] = val

            results = simulation_variable_household_size(pop_size, params=params, num_iters=num_iters)
            plot_hist_exp_2(results, param, val)
            avgs = np.mean(results, axis=0)
            print('indep fnr = {}, corr fnr = {}, indep eff = {}, corr eff = {}'.format(avgs[0], avgs[1], avgs[2], avgs[3]))

            with open('../results/experiment_2/sensitivity_analysis/results_{}={}.data'.format(param, val), 'wb') as f:
                np.savetxt(f, results)
    return


def run_simulations_for_pareto_fontier():
    pop_size = 12000
    num_iters = 500

    prevalences = [0.001, 0.005, 0.01, 0.05, 0.1]
    pool_sizes = [3,4,6,8,10,12,15,20,24,30,40]

    for p in prevalences:
        for n in pool_sizes:
            params = NOMINAL_PARAMS.copy()
            params['prevalence'] = p
            params['pool size'] = n

            results = simulation_variable_household_size(pop_size, params=params, num_iters=num_iters)
            avgs = np.mean(results, axis=0)
            print('indep fnr = {}, corr fnr = {}, indep eff = {}, corr eff = {}'.format(avgs[0], avgs[1], avgs[2], avgs[3]))

            with open('../results/experiment_2/pareto_analysis/results_prevalence={}_pool-size={}.data'.format(p, n), 'wb') as f:
                np.savetxt(f, results)
    return



if __name__ == '__main__':
    #run_simulations_for_sensitivity_analysis()
    run_simulations_for_pareto_fontier()
    # for param in ['prevalence', 'pool size', 'SAR', 'FNR', 'household dist']:
    #     generate_sensitivity_plots(param)
