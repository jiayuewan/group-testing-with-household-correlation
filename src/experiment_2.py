import matplotlib.pyplot as plt
import numpy as np
import pickle
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
"SAR": 0.188,
"FNR": 0.05
}


def simulation_variable_household_size(file_dir, population_size, params=NOMINAL_PARAMS, num_iters=500):
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

    print(f"Process {os.getpid()}: Running simulations for param config: {params} with {num_iters} iterations...")
    sys.stdout.flush()
    # print('Process {}: running simulation under r = {}, n = {}, FNR = {}, SAR = {}, dist = {} with {} iterations...'.format(os.getpid(), prevalence, pool_size, FNR, SAR, household_dist, num_iters))
    
    results = np.zeros((num_iters, 4))
    for i in range(num_iters):
        infections = generate_correlated_infections(population_size, prevalence, type='real', household_dist=household_dist, SAR=SAR)

        fnr_indep, eff_indep = one_stage_group_testing(infections, pool_size=pool_size, LoD=LoD, type="real", shuffle=True)[:2]
        fnr_correlated, eff_corr = one_stage_group_testing(infections, pool_size=pool_size, LoD=LoD, type="real", shuffle=False)[:2]
        results[i, :] = [fnr_indep, fnr_correlated, eff_indep, eff_corr]
    
    # avgs = np.mean(results, axis=0)
    # print('indep fnr = {}, corr fnr = {}, indep eff = {}, corr eff = {}'.format(avgs[0], avgs[1], avgs[2], avgs[3]))

    # param_val = {x: params[x] for x in params if NOMINAL_PARAMS[x] != params[x]}
    with open(file_dir, 'wb') as f:
            np.savetxt(f, results)
    
    # if param_val == {}: # nominal
    #     if not os.path.exists('../results/experiment_2/sensitivity_analysis_{}/'.format(num_iters)):
    #         os.makedirs('../results/experiment_2/sensitivity_analysis_{}/'.format(num_iters))
        
    #     filedir = '../results/experiment_2/sensitivity_analysis_{}/results_prevalence={}_SAR={}_pool size={}_FNR={}_household dist={}.data'.format(\
    #     num_iters, params['prevalence'], params['SAR'], params['pool size'], params['FNR'], params['household dist'])
    #     with open(filedir, 'wb') as f:
    #         np.savetxt(f, results)

    #     p = params['prevalence']
    #     n = params['pool size']
    #     if not os.path.exists('../results/experiment_2/pareto_analysis_{}/'.format(num_iters)):
    #         os.makedirs('../results/experiment_2/pareto_analysis_{}/'.format(num_iters))
    #     with open('../results/experiment_2/pareto_analysis_{}/results_prevalence={}_pool-size={}.data'.format(num_iters, p, n), 'wb') as f:
    #         np.savetxt(f, results)

    # elif len(param_val) == 1: 
    #     (param, val), = param_val.items()
    #     if not os.path.exists('../results/experiment_2/sensitivity_analysis_{}/'.format(num_iters)):
    #         os.makedirs('../results/experiment_2/sensitivity_analysis_{}/'.format(num_iters))
    #     with open('../results/experiment_2/sensitivity_analysis_{}/results_{}={}.data'.format(num_iters, param, val), 'wb') as f:
    #         np.savetxt(f, results)

    # else:
    #     assert len(param_val) == 2
    #     p = params['prevalence']
    #     n = params['pool size']        
    #     if not os.path.exists('../results/experiment_2/pareto_analysis_{}/'.format(num_iters)):
    #         os.makedirs('../results/experiment_2/pareto_analysis_{}/'.format(num_iters))
    #     with open('../results/experiment_2/pareto_analysis_{}/results_prevalence={}_pool-size={}.data'.format(num_iters, p, n), 'wb') as f:
    #         np.savetxt(f, results)

    print('Process {} is done...'.format(os.getpid()))
    sys.stdout.flush()
    return 


def run_simulations_for_sensitivity_analysis(num_iters=2000):
    # print("sensitivity")
    pop_size = 12000
    houshold_dists = list(HOUSEHOLD_DIST.keys())
    houshold_dists.remove('US')

    configs = {
    'prevalence' :[0.001, 0.005, 0.05, 0.1],
    'SAR' : [0.039, 0.154, 0.222, 0.446],
    'pool size': [3, 12, 24],
    'FNR': [0.025, 0.1],
    'household dist': houshold_dists
    }

    output_dir = "../results/experiment_2/sensitivity_analysis_{}/".format(num_iters)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # results = simulation_variable_household_size(pop_size, params=NOMINAL_PARAMS, num_iters=num_iters)
    # plot_hist_exp_2(results, "nominal")
    # avgs = np.mean(results, axis=0)
    
    # print('indep fnr = {}, corr fnr = {}, indep eff = {}, corr eff = {}'.format(avgs[0], avgs[1], avgs[2], avgs[3]))
    # nominal_filedir = '../results/experiment_2/sensitivity_analysis/results_prevalence={}_SAR={}_pool size={}_FNR={}_household dist={}_niters={}.data'.format(\
    #     NOMINAL_PARAMS['prevalence'], NOMINAL_PARAMS['SAR'], NOMINAL_PARAMS['pool size'], NOMINAL_PARAMS['FNR'], NOMINAL_PARAMS['household dist'], num_iters)
    # with open(nominal_filedir, 'wb') as f:
    #     np.savetxt(f, results)
    output_dir = "../results/experiment_2/sensitivity_analysis_{}/".format(num_iters)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    jobs = []

    filename = 'results_prevalence={}_SAR={}_pool size={}_FNR={}_household dist={}.data'.format(\
        NOMINAL_PARAMS['prevalence'], NOMINAL_PARAMS['SAR'], NOMINAL_PARAMS['pool size'], NOMINAL_PARAMS['FNR'], NOMINAL_PARAMS['household dist'])
    file_dir = os.path.join(output_dir, filename)
    fn_args = (file_dir, pop_size, NOMINAL_PARAMS, num_iters)
    proc = Process(target=simulation_variable_household_size, args=fn_args)
    jobs.append(proc)
    proc.start()

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
    
    # print("Simulation for sensitivity analysis is done.")
    # results = simulation_variable_household_size(pop_size, params=params, num_iters=num_iters)
    # plot_hist_exp_2(results, param, val)
    # avgs = np.mean(results, axis=0)
    # print('indep fnr = {}, corr fnr = {}, indep eff = {}, corr eff = {}'.format(avgs[0], avgs[1], avgs[2], avgs[3]))

    # with open('../results/experiment_2/sensitivity_analysis/results_{}={}.data'.format(param, val), 'wb') as f:
    #     np.savetxt(f, results)

    # for param, values in configs.items():
    #     for val in values:
    #         params = NOMINAL_PARAMS.copy()
    #         params[param] = val

    #         results = simulation_variable_household_size(pop_size, params=params, num_iters=num_iters)
    #         plot_hist_exp_2(results, param, val)
    #         avgs = np.mean(results, axis=0)
    #         print('indep fnr = {}, corr fnr = {}, indep eff = {}, corr eff = {}'.format(avgs[0], avgs[1], avgs[2], avgs[3]))

    #         with open('../results/experiment_2/sensitivity_analysis/results_{}={}.data'.format(param, val), 'wb') as f:
    #             np.savetxt(f, results)
    return


def run_simulations_for_pareto_fontier(num_iters=2000):
    # print('pareto')
    pop_size = 12000

    prevalences = [0.001, 0.005, 0.01, 0.05, 0.1]
    pool_sizes = [3,4,6,8,10,12,15,20,24,30,40]

    output_dir = "../results/experiment_2/pareto_analysis_{}/".format(num_iters)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    jobs = []
    
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
            # print(f"Running simulations for param config: {params} with {num_iters} iterations...")

    # map(lambda p: p.join(), jobs)

    for p in jobs:
        p.join()
            # results = simulation_variable_household_size(pop_size, params=params, num_iters=num_iters)
            # avgs = np.mean(results, axis=0)
            # print('indep fnr = {}, corr fnr = {}, indep eff = {}, corr eff = {}'.format(avgs[0], avgs[1], avgs[2], avgs[3]))

    print("Simulation for pareto analysis is done.")

    return



if __name__ == '__main__':
    num_iters = int(sys.argv[1])
    run_simulations_for_sensitivity_analysis(num_iters=num_iters)
    run_simulations_for_pareto_fontier(num_iters=num_iters)

    # for param in ['prevalence', 'pool size', 'SAR', 'FNR', 'household dist']:
    #     generate_sensitivity_plots(param)
