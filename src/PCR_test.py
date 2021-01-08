import numpy as np


PCR_PARAMS = {'V_sample': 1, 'c_1': 1/10, 'gamma': 1/2, 'c_2': 1, 'LoD': 10}


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


def pooled_PCR_test(mu, params=PCR_PARAMS):
    """
    Perform one pooled pcr test and output the test result

    INPUT:
    mu = array of viral loads (copies/mL) in the sample to be put in the same pool;
    params = PCR test parameters for subsampling and dilution

    OUTPUT: 1 for test positive; 0 for test negative
    """
    V_sample = params['V_sample']
    c_1 = params['c_1']
    gamma = params['gamma']
    c_2 = params['c_2']
    LoD = params['LoD']
    pool_size = len(mu)

    # copies of RNA in a subsample that is c_1 the original volume
    N_subsamples = np.random.binomial(V_sample * mu, c_1 / pool_size)
    N_pre_extraction = np.sum(N_subsamples)

    # copies of extracted RNA in the subsample
    N_extracted = np.random.binomial(N_pre_extraction, gamma)

    # copies of RNA in PCR template that is c_2 of the previous step's eluted quantity
    N_template = np.random.binomial(N_extracted, c_2)

    return int(N_template >= LoD)

# deprecated
def eval_FNR(mu, params=PCR_PARAMS, n_iter=1000):
    """Input: viral load concentration, parameters
    Output: expected false negative rate"""

    detected = 0
    for j in range(n_iter):
        detected += one_PCR_test(mu, params)

    return 1 - detected / n_iter


if __name__ == '__main__':
    print('')

    # n_iter = 100
    # mu_list = 10 ** np.linspace(-2, 4, 61)
    # FNRs = np.zeros(len(mu_list))
    # for i in range(len(mu_list)):
    #     FNRs[i] = eval_FNR(mu_list[i])
    # print(FNRs)
