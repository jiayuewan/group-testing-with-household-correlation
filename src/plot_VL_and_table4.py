import numpy as np
import matplotlib.pyplot as plt

def plot_VL(n_samples = 1000000):
    mixture_weights = [0.33, 0.54, 0.13]
    mixture_params = [[8.09, 1.06], [5.35, 0.89], [3.75, 0.39]]
    
    samples = []

    for _ in range(n_samples):
        component = np.random.choice(3, p = mixture_weights)
        mean, sd = mixture_params[component]
        samples.append(np.random.normal(loc = mean, scale = sd))

        plt.rcParams["font.family"] = 'serif'
        plt.hist(samples, bins = 100, density = True)
        plt.title('log10 viral load distribution')
        plt.xlabel('log10 viral load')
        plt.ylabel('Density')

        # plt.savefig('log10_VL_density.pdf')


def plot_table_4():
    prevs = ['0.1%', '0.5%', '1%', '5%', '10%']
    np_sizes = [40, 15, 12, 6, 4]
    cp_sizes = [40, 20, 12, 6, 4]
    np_sens_times_eff = [13.52, 6.29, 4.56, 2.17, 1.59]
    cp_sens_times_eff = [15.86, 7.26, 5.23, 2.44, 1.72]

    plt.rcParams["font.family"] = 'serif'

    plt.plot(np.arange(5), [1/np_sens_times_eff[i] for i in range(5)], 
                color = 'mediumpurple', marker = '^', linestyle = '--', label = 'naive')
    plt.plot(np.arange(5), [1/cp_sens_times_eff[i] for i in range(5)], 
                color = 'mediumaquamarine', marker = 'o', linestyle = '-', label = 'correlated')
    plt.xticks(np.arange(5), prevs)

    for i in range(5):
        plt.annotate(np_sizes[i], (i-0.2, 1/np_sens_times_eff[i]))
        plt.annotate(cp_sizes[i], (i+0.11, 1/cp_sens_times_eff[i]-0.01))

    plt.xlabel('Prevalence')
    plt.ylabel(r'$(Sensitivity * Efficiency)^{-1}$')
    plt.legend()
    plt.title(r'$(Sensitivity * Efficiency)^{-1}$ for naive and correlated pooling')

    # plt.savefig('sens_eff.pdf')