import viral_load_distribution as VL_dist
import PCR_test
import numpy as np
import scipy.stats as st
import scipy.optimize as opt
import matplotlib.pyplot as plt

sample = 10 ** VL_dist.sample_log10_VL(10000)

base_params = {'V_sample': 1, 'c_1': 1/20, 'gamma': 1/2, 'c_2': 1, 'LoD':20}

def tune_LoD(sample, LoD, params=base_params):
    params['LoD'] = LoD
    FNRs = np.zeros(len(sample))
    for i in range(len(sample)):
        FNRs[i] = PCR_test.eval_FNR(sample[i], params, 200)

    return np.mean(FNRs)


if __name__ == '__main__':
    #mean_FNR = tune_LoD(sample,20)
    #print(mean_FNR)
    mu_list = 10 ** np.linspace(3, 4, 21)
    FNRs = np.zeros(len(mu_list))
    for i in range(len(mu_list)):
        FNRs[i] = PCR_test.eval_FNR(mu_list[i], base_params, 500)
    print(FNRs)
