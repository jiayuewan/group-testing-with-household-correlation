import numpy as np
from functools import partial
from viral_load_distribution import sample_log10_viral_loads
import matplotlib.pyplot as plt 
# from scipy.optimize import fsolve
from multiprocessing import Pool, cpu_count
import csv

PCR_PARAMS = {'V_sample': 1, 'c_1': 1/10, 'xi': 1/2, 'c_2': 1}  # c_1 = V_subsample / V_sample
PCR_LOD = 100
# LoD -- FNR table
# 108 -- 0.025
# 174 -- 0.05
# 342 -- 0.10


def false_negative_rate_binary(num_positives, type='exp'):
    """
    computes the false negative rate (FNR) of a pooled PCR test, based on the number of
    positive samples in the pooled sample

    INPUT:
    num_positives = number of positive samples in the pooled sample
    type = false negative rate function used in the calculation
    """
    assert num_positives >= 0

    if type == 'exp':
        return 0.1 ** num_positives

    elif type == 'reciprocal':
        return 1 / (1 + num_positives)

    else:
        assert type == 'step'
        return 1 - (num_positives >= 3)


def pooled_PCR_test(mu, individual=False, params=PCR_PARAMS, LoD=PCR_LOD):
    """
    Perform one pooled pcr test and output the test result

    INPUT:
    mu = 1d or 2d array of viral loads (copies/mL) in the samples. If individual = False, samples in the same row are put in the same pool
    individual= boolean, True if performing an individual PCR test, False if performing an pooled sample PCR test
    params = PCR test parameters for subsampling and dilution

    OUTPUT: 1 (True) for test positive; 0 (False) for test negative
    """
    if individual:
        V_sample = params['V_sample']
        c_1 = params['c_1']
        xi = params['xi']
        c_2 = params['c_2']

        # copies of RNA in a subsample that is c_1 the original volume
        N_subsamples = np.random.binomial(V_sample * mu, c_1)
        N_pre_extraction = N_subsamples

        # copies of extracted RNA in the subsample
        N_extracted = np.random.binomial(N_pre_extraction, xi)

        # copies of RNA in PCR template that is c_2 of the previous step's eluted quantity
        N_templates = np.random.binomial(N_extracted, c_2)
        return N_templates >= LoD

    else:
        V_sample = params['V_sample']
        c_1 = params['c_1']
        xi = params['xi']
        c_2 = params['c_2']
        pool_size = len(mu) if mu.ndim == 1 else mu.shape[1]

        # copies of RNA in a subsample that is c_1 the original volume
        N_subsamples = np.random.binomial(V_sample * mu, c_1 / pool_size)
        N_pre_extraction = np.sum(N_subsamples) if mu.ndim == 1 else np.sum(N_subsamples, axis=1)

        # copies of extracted RNA in the subsample
        N_extracted = np.random.binomial(N_pre_extraction, xi)

        # copies of RNA in PCR template that is c_2 of the previous step's eluted quantity
        N_templates = np.random.binomial(N_extracted, c_2)
        return N_templates >= LoD


def pooled_PCR_test_alternative(mu, individual=False, params=PCR_PARAMS, LoD=PCR_LOD):
    """
    An alternative method of sampling the PCR test outcomes but requires less binomial samples
    same input and output format as pooled_PCR_test()
    """
    if individual:
        V_sample = params['V_sample']
        c_1 = params['c_1']
        xi = params['xi']
        c_2 = params['c_2']

        N_templates = np.random.binomial(V_sample * mu, c_1 * xi * c_2)

    else:
        V_sample = params['V_sample']
        c_1 = params['c_1']
        xi = params['xi']
        c_2 = params['c_2']
        pool_size = len(mu) if mu.ndim == 1 else mu.shape[1]
    
        X = np.sum(mu) if mu.ndim == 1 else np.sum(mu, axis=1)
        N_templates = np.random.binomial(V_sample * X, c_1 / pool_size * xi * c_2)

    return N_templates >= LoD


# deprecated
def eval_FNR(mu, params=PCR_PARAMS, n_iters=1000):
    """Input: viral load concentration, parameters
    Output: expected false negative rate"""

    detected = 0
    for j in range(n_iters):
        detected += pooled_PCR_test(mu, params)

    return 1 - detected / n_iters



def calculate_FNR(LoD, sample_size=100000):
    log10_mu_list = sample_log10_viral_loads(sample_size)
    mu_list = (10 ** np.array(log10_mu_list)).astype(int)
    FNs = sample_size - sum(pooled_PCR_test_alternative(mu_list, individual=True, LoD=LoD))
    return FNs / sample_size


def generate_LoD_to_FNR_table(min_LoD, max_LoD):
    LoDs = np.arange(min_LoD, max_LoD + 1)
    num_processes = cpu_count()
    print(LoDs, num_processes)
    with Pool(num_processes) as p:
        FNRs = p.map(calculate_FNR, LoDs)
    
    results = zip(LoDs, FNRs)
    with open('../results/PCR_tests/LoD_to_FNR_altenative.csv','w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['LoD','FNR'])
        for row in results:
            csv_out.writerow(row)
    return



def calculate_FNR_for_fixed_VL(LoD, mu, sample_size=1000000):
    mu_list = np.ones(sample_size, dtype=int) * int(10 ** mu)
    FNs = sample_size - sum(pooled_PCR_test(mu_list, individual=True, LoD=LoD))
    return FNs / sample_size
    # LoD = 174
    # mu = 3.5 -> 89.7% (VL = 3162)
    # mu = 3.55 -> 38.5%
    # mu = 3.594 -> 4.6%
    # mu = 3.65 -> 0.02% (VL = 4467)


def generate_indiv_test_sensitivity_curve():
    mus = np.linspace(3, 4, 100)
    Sns = [1 - calculate_FNR_for_fixed_VL(174, mu) for mu in mus]

    #plt.figure(figsize=(6,4))
    plt.plot(mus, Sns)
    plt.xlabel(r"$\log_{10} VL$ (copies/mL)")
    plt.ylabel("sensitivity")
    plt.title(r"Individual test sensitvity vs. $\log_{10}$ viral load under $\tau=174$")
    plt.savefig('../figs/individual_test_sensitivity.pdf')
    plt.close()


def compute_bound_in_theorem_2(pool_size=2, LoD=174, n_iters=100000):
    count_Y_1_SD_0_given_S = np.zeros(pool_size)
    count_Y_1_SD_1_given_S_1 = 0

    for i in range(pool_size):
        for _ in range(n_iters):
            viral_loads = np.array([10 ** x for x in sample_log10_viral_loads(n_samples=i + 1)] + [0] * (pool_size - i - 1)).astype(int)
            S_D = sum(pooled_PCR_test_alternative(viral_loads, individual=True, LoD=LoD))
            Y = pooled_PCR_test_alternative(viral_loads, individual=False, LoD=LoD)
            if Y == 1 and S_D == 0:
                count_Y_1_SD_0_given_S[i] += 1


    for _ in range(n_iters):
        viral_loads = np.array([10 ** x for x in sample_log10_viral_loads(n_samples=1)] + [0] * (pool_size - 1)).astype(int)
        S_D = sum(pooled_PCR_test_alternative(viral_loads, individual=True, LoD=LoD))
        Y = pooled_PCR_test_alternative(viral_loads, individual=False, LoD=LoD)
        if Y == 1 and S_D > 0:
            count_Y_1_SD_1_given_S_1 += 1

    return (pool_size, LoD, n_iters, count_Y_1_SD_0_given_S, count_Y_1_SD_1_given_S_1, count_Y_1_SD_0_given_S/count_Y_1_SD_1_given_S_1)


def compute_bounds_in_theorem_2(n_iters=100000):
    result_list = []
    def log_result(result):
        result_list.append(result)

    p = Pool()
    n_iters = n_iters
    for n in [2, 4, 6, 12]:
        for LoD in [108, 174, 342, 1240]:
            print(f"computing bound for n = {n} and LoD = {LoD}...") # to multiply # iterations by 100
            p.apply_async(compute_bound_in_theorem_2, args = (n, LoD, n_iters, ), callback = log_result)

    p.close()
    p.join()

    result_list = sorted(result_list, key=lambda x: (x[0], x[1]))
    with open('../results/PCR_tests/bounds_in_theorem_2_updated_{}.csv'.format(n_iters),'w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['pool size', 'LoD', 'niters', 'numerator', 'denominator', 'bound'])
        for row in result_list:
            csv_out.writerow(row)
    return

if __name__ == '__main__':
    # print(calculate_FNR(140))
    # print(calculate_FNR_for_fixed_VL(174, 3.45))
    # print(pooled_PCR_test(np.array([0, 0, 0])))
    # print(pooled_PCR_test(np.array([100, 100, 1000])))
    # print(compute_bound_in_theorem_2(1, 2))
    # for n in [2, 6, 12]:
    #     #print(n, compute_bound_in_theorem_2(n, n))
    #     print(compute_bound_in_theorem_2(n, n)[0] / compute_bound_in_theorem_2(1, n)[0])
        #ratios = np.zeros(n - 1)
    
        #for k in range(1, n):
        #    ratios[k-1] = compute_bound_in_theorem_2(k, n)[2]
    
        #print("for pool size = {0}, the ratios for k from 1 to {0} are {1}".format(n, ratios))
    # plt.rcParams["font.family"] = 'serif'
    # generate_indiv_test_sensitivity_curve()
    # generate_LoD_to_FNR_table(10, 1500)
    compute_bounds_in_theorem_2(n_iters=1000)
