import numpy as np

base_params = {'V_sample': 1000, 'c_1': 1/20, 'gamma': 1/2, 'c_2': 1}

def one_PCR_test(mu, params=base_params):
    """Input: mu = viral load concentration; params = PCR test parameters for subsampling and dilution
    Output: 1 for test positive; 0 for test negative"""

    V_sample = params['V_sample']
    c_1 = params['c_1']
    gamma = params['gamma']
    c_2 = params['c_2']

    # copies of RNA in a subsample that is c_1 the original volume
    N_subsample = np.random.binomial(V_sample * mu, c_1)

    # copies of extracted RNA in the subsample
    N_extracted = np.random.binomial(N_subsample, gamma)

    # copies of RNA in PCR template that is c_2 of the previous step's eluted quantity
    N_template = np.random.binomial(N_extracted, c_2)

    return (N_template>=5)

if __name__ == '__main__':
    n_iter = 100
    mu_list = 10 ** np.linspace(-2, 4, 61)
    fraction_detected = np.zeros(len(mu_list))
    for i in range(len(mu_list)):
        for j in range(n_iter):
            fraction_detected[i] += one_PCR_test(mu_list[i])
        fraction_detected[i] = fraction_detected[i] / n_iter
    print(fraction_detected)
