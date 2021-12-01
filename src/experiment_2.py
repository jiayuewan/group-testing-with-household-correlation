import matplotlib.pyplot as plt
import numpy as np
from household_dist import HOUSEHOLD_DIST
from generate_infection_states import generate_correlated_infections
from one_stage_hierarchical_group_testing import one_stage_group_testing
from plotting_helpers import plot_hist_exp_2, generate_sensitivity_plots
from multiprocessing import Process
import os 
import sys


NOMINAL_PARAMS = {
"pool size": 6,
"prevalence": 0.01,
"household dist": 'US',
"SAR": 0.166,
"FNR": 0.05
}


def simulation_variable_household_size(file_dir, population_size, params=NOMINAL_PARAMS, num_iters=500):
    # simulation results: fnr_indep, fnr_corr, efficiency_indep, efficiency_corr
    pool_size = params['pool size']
    prevalence = params['prevalence']
    household_dist = params['household dist']
    SAR = params['SAR']
    FNR = params['FNR']

    if not FNR in [0.025, 0.05, 0.1, 0.2]:
        raise ValueError("supported FNR values do not include {}".format(FNR))
    else:
        LoD = 108 if FNR == 0.025 else 174 if FNR == 0.05 else 342 if FNR == 0.1 else 1240

    print(f"Process {os.getpid()}: Running simulations for param config: {params} with {num_iters} iterations...")
    sys.stdout.flush()
    
    results = np.zeros((num_iters, 4))
    for i in range(num_iters):
        infections = generate_correlated_infections(population_size, prevalence, type='real', household_dist=household_dist, SAR=SAR)

        fnr_indep, eff_indep = one_stage_group_testing(infections, pool_size=pool_size, LoD=LoD, type="real", shuffle=True)[:2]
        fnr_correlated, eff_corr = one_stage_group_testing(infections, pool_size=pool_size, LoD=LoD, type="real", shuffle=False)[:2]
        results[i, :] = [fnr_indep, fnr_correlated, eff_indep, eff_corr]
    
    with open(file_dir, 'wb') as f:
            np.savetxt(f, results)
    
    return 


def run_simulations_for_sensitivity_analysis(num_iters=2000):
    pop_size = 12000
    houshold_dists = list(HOUSEHOLD_DIST.keys())
    houshold_dists.remove('US')

    configs = {
    'prevalence' :[0.001, 0.005, 0.05, 0.1],
    'SAR' : [0.005, 0.140, 0.193, 0.446],
    'pool size': [3, 12, 24],
    'FNR': [0.025, 0.1, 0.2],
    'household dist': houshold_dists
    }

    output_dir = "../results/experiment_2/sensitivity_analysis_{}/".format(num_iters)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    jobs = []

    # run simulations for the baseline scenario
    filename = 'results_prevalence={}_SAR={}_pool size={}_FNR={}_household dist={}.data'.format(\
        NOMINAL_PARAMS['prevalence'], NOMINAL_PARAMS['SAR'], NOMINAL_PARAMS['pool size'], NOMINAL_PARAMS['FNR'], NOMINAL_PARAMS['household dist'])
    file_dir = os.path.join(output_dir, filename)
    fn_args = (file_dir, pop_size, NOMINAL_PARAMS, num_iters)
    proc = Process(target=simulation_variable_household_size, args=fn_args)
    jobs.append(proc)
    proc.start()

    # run simulations for the sensitivity analysis
    for param, values in configs.items():
        for val in values:
            params = NOMINAL_PARAMS.copy()
            params[param] = val
            
            filename = "results_{}={}.data".format(param, val)
            file_dir = os.path.join(output_dir, filename)
            fn_args = (file_dir, pop_size, params, num_iters)
            proc = Process(target=simulation_variable_household_size, args=fn_args)
            jobs.append(proc)
            proc.start()
            
    map(lambda p: p.join(), jobs)
    
    return


def run_simulations_for_pareto_fontier(num_iters=2000):
    pop_size = 12000
    prevalences = [0.001, 0.005, 0.01, 0.05, 0.1]
    pool_sizes = [3,4,6,8,10,12,15,20,24,30,40]

    output_dir = "../results/experiment_2/pareto_analysis_{}/".format(num_iters)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    jobs = []
    
    # run simulations for the pareto analysis (sensitivity vs efficiency) for different prevalence and pool size
    for p in prevalences:
        for n in pool_sizes:
            params = NOMINAL_PARAMS.copy()
            params['prevalence'] = p
            params['pool size'] = n
            
            filename = "results_prevalence={}_pool-size={}.data".format(p, n)
            file_dir = os.path.join(output_dir, filename)
            fn_args = (file_dir, pop_size, params, num_iters)
            proc = Process(target=simulation_variable_household_size, args=fn_args)
            jobs.append(proc)
            proc.start()

    for p in jobs:
        p.join()

    return


if __name__ == '__main__':
    num_iters = int(sys.argv[1])
    run_simulations_for_sensitivity_analysis(num_iters=num_iters)
    run_simulations_for_pareto_fontier(num_iters=num_iters)

