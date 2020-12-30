import numpy as np
import scipy.stats as st
import scipy.optimize as opt
import matplotlib.pyplot as plt


# mix_gauss_params = [[0.33, 20.13, 3.6], [0.54, 29.41, 3.02], [0.13, 34.81, 1.31]]
GMM_PARAMS = [[0.33, 0.54, 0.13], [8.09, 5.35, 3.75], [1.06, 0.89, 0.39]]


def sample_log10_viral_loads(n_samples=1, params=GMM_PARAMS):
    pis = params[0]
    mus = params[1]
    sigmas = params[2]

    clusters = np.random.choice(3, size=n_samples, p=pis)
    viral_loads = [st.norm.rvs(loc=mus[cluster], scale=sigmas[cluster]) for cluster in clusters]
    return viral_loads


def plot_log10_VL_pdf(params=GMM_PARAMS):
    pis = params[0]
    mus = params[1]
    sigmas = params[2]
    n_clusters = len(pis)
    def compute_log_10_viral_load_pdf(x):
        pdf = np.sum([pis[i] * st.norm.pdf(x, loc=mus[i], scale=sigmas[i]) for i in range(n_clusters)])
        return pdf

    xs = np.linspace(-5, 15, 1000)
    plt.plot(xs, [compute_log_10_viral_load_pdf(x) for x in xs])
    plt.title('Distribution of log10 viral load')
    plt.xlabel('log10 viral load')
    plt.ylabel('Probability density')
    plt.savefig('../figs/log_10_viral_load_dist.pdf')
    return


if __name__ == '__main__':
    #print([log10_viral_load_pdf(i) for i in range(20)])
    #print([log10_viral_load_cdf(i) for i in range(20)])
    #print(sample_log10_VL(10))
    print(sample_log10_viral_loads(10))
    plot_log10_VL_pdf()
