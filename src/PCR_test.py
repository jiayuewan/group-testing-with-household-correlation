import numpy as np
from viral_load_distribution import sample_log10_viral_loads


PCR_PARAMS = {'V_sample': 1, 'c_1': 1/10, 'gamma': 1/2, 'c_2': 1}
PCR_LOD = 100
# LoD -- FNR table
# 108 -- 0.025
# 174 -- 0.05
# 345 -- 0.10


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
        gamma = params['gamma']
        c_2 = params['c_2']

        # copies of RNA in a subsample that is c_1 the original volume
        N_subsamples = np.random.binomial(V_sample * mu, c_1)
        N_pre_extraction = N_subsamples

        # copies of extracted RNA in the subsample
        N_extracted = np.random.binomial(N_pre_extraction, gamma)

        # copies of RNA in PCR template that is c_2 of the previous step's eluted quantity
        N_templates = np.random.binomial(N_extracted, c_2)
        return N_templates >= LoD

    else:
        V_sample = params['V_sample']
        c_1 = params['c_1']
        gamma = params['gamma']
        c_2 = params['c_2']
        pool_size = len(mu) if mu.ndim == 1 else mu.shape[1]

        # copies of RNA in a subsample that is c_1 the original volume
        N_subsamples = np.random.binomial(V_sample * mu, c_1 / pool_size)
        N_pre_extraction = np.sum(N_subsamples) if mu.ndim == 1 else np.sum(N_subsamples, axis=1)

        # copies of extracted RNA in the subsample
        N_extracted = np.random.binomial(N_pre_extraction, gamma)

        # copies of RNA in PCR template that is c_2 of the previous step's eluted quantity
        N_templates = np.random.binomial(N_extracted, c_2)
        return N_templates >= LoD


# deprecated
def eval_FNR(mu, params=PCR_PARAMS, n_iters=1000):
    """Input: viral load concentration, parameters
    Output: expected false negative rate"""

    detected = 0
    for j in range(n_iter):
        detected += pooled_PCR_test(mu, params)

    return 1 - detected / n_iter


# typically takes ~ 1 minute to run
def calculate_FNR(LoD, sample_size=1000000):
    log10_mu_list = sample_log10_viral_loads(sample_size)
    mu_list = (10 ** np.array(log10_mu_list)).astype(int)
    FNs = sample_size - sum(pooled_PCR_test(mu_list, individual=True, LoD=LoD))
    return FNs / sample_size


def calculate_FNR_for_fixed_VL(LoD, mu, sample_size=1000000):
    mu_list = np.ones(sample_size, dtype=int) * int(10 ** mu)
    FNs = sample_size - sum(pooled_PCR_test(mu_list, individual=True, LoD=LoD))
    return FNs / sample_size
    # LoD = 174
    # mu = 3.5 -> 89.7% (VL = 3162)
    # mu = 3.55 -> 38.5%
    # mu = 3.594 -> 4.6%
    # mu = 3.65 -> 0.02% (VL = 4467)


def compute_bound_in_theorem_2(num_positives=2, pool_size=2, n_iters=100000):
    count_Y_1_SD_0 = 0
    count_Y_1_SD_1 = 0

    for _ in range(n_iters):
        viral_loads = np.array([10 ** x for x in sample_log10_viral_loads(n_samples=num_positives)] + [0] * (pool_size - num_positives)).astype(int)
        S_D = sum(pooled_PCR_test(viral_loads, individual=True))
        Y = pooled_PCR_test(viral_loads, individual=False)
        if Y == 1 and S_D == 0:
            count_Y_1_SD_0 += 1

        elif Y == 1 and S_D > 0:
            count_Y_1_SD_1 += 1

    return (count_Y_1_SD_0, count_Y_1_SD_1, count_Y_1_SD_0/count_Y_1_SD_1)


if __name__ == '__main__':
    # print(calculate_FNR_for_fixed_VL(174, 3.45))
    # print(pooled_PCR_test(np.array([0, 0, 0])))
    # print(pooled_PCR_test(np.array([100, 100, 1000])))
    print(compute_bound_in_theorem_2(1, 2))
    # for n in [2, 6, 12]:
    #     print(n, compute_bound_in_theorem_2(n, n))
    #     ratios = np.zeros(n - 1)
    #
    #     for k in range(1, n):
    #         ratios[k-1] = compute_bound_in_theorem_2(k, n)[2]
    #
    #     print("for pool size = {0}, the ratios for k from 1 to {0} are {1}".format(n, ratios))
