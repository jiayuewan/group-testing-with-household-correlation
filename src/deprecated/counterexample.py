import numpy as np
import matplotlib
import matplotlib.cm
cmap = matplotlib.cm.get_cmap().copy()
cmap.set_bad('white',np.nan)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


theta1_list = np.linspace(0.01, 1, 100)
theta2_list = np.linspace(0.01, 1, 100)
num_grid = len(theta1_list)
n=2
alpha=0.01

num_grid = len(theta1_list)


def compute_np_beta(alpha, theta1, theta2):

    return (1 - alpha * (7/12*theta2**2 + 7/12*5/12*theta2 + 5**2/12**2) \
            - (1-alpha) * (7/12*theta1*theta2+5/12*theta1))

def compute_np_eta(alpha, theta1, theta2, np_beta, n=2):

    denom = n * alpha * (1-np_beta)
    num = n * (theta1 * 2 * alpha * (1-alpha)\
                + theta2 * alpha**2 * ((7/12)**2 + 2 * 7/12 * 5/12)\
                + alpha**2 * (5/12)**2)
  
    return num/denom

def compute_np_efficiency(n, alpha, eta, beta):
    inv_eff = 1/n + alpha * eta * (1-beta)
    return 1/inv_eff

def compute_cp_efficiency(n, alpha, theta1, theta2):
    cp_inv_eff = 1/n + alpha * n * (8*theta1+2*theta2)/(5*theta1+3*theta1*theta2+4*theta2**2)*(theta1*theta2/4 + theta2**2/3 + 5*theta1/12)

    return 1/cp_inv_eff


def plot_diff_efficiency():

    diff_eff_vals = np.empty((num_grid, num_grid))
    diff_eff_vals[:] = np.nan

    for i in range(num_grid):
        for j in range(i, num_grid):

            np_beta = compute_np_beta(alpha, theta1_list[i], theta2_list[j])
            np_eta = compute_np_eta(alpha, theta1_list[i], theta2_list[j], np_beta)
            np_eff = compute_np_efficiency(n, alpha, np_eta, np_beta)
            cp_eff = compute_cp_efficiency(n, alpha, theta1_list[i], theta2_list[j])

            diff_eff_vals[i,j] = cp_eff - np_eff

    data = diff_eff_vals.transpose()

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap = cmap, extent = [0, 100, 100, 0])
    cbar = ax.figure.colorbar(im, ax=ax)
        
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_title("Difference in efficiency \n between correlated and naive pooling")
    fig.tight_layout()
    #plt.show()
    plt.contour(data, levels = [0], colors = 'r', linewidth=5)
    ax.hlines(y=100, xmin=0, xmax=49, color='r', linestyle='-', linewidth=5)
    ax.vlines(x=0.1, ymin=2, ymax=100, color='r', linestyle='-', linewidth=5)


    plt.savefig('/home/yz685/group-testing-with-household-correlation/figs/counterexample.pdf', format='pdf', dpi=600, bbox_inches='tight')




if __name__ == '__main__':
    plt.rcParams["font.family"] = 'serif'
    
    plot_diff_efficiency()