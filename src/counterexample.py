import numpy as np
import matplotlib
import matplotlib.cm
cmap = matplotlib.cm.get_cmap().copy()
cmap.set_bad('white',np.nan)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


beta1_list = np.linspace(0.01, 1, 100)
beta2_list = np.linspace(0.01, 1, 100)
n=2
alpha=0.01

num_grid = len(beta1_list)


def diff_efficiency(beta1, beta2, n=n, alpha=alpha):
  np_val_inv = 1/n + alpha * n * 12/(7*beta2+5)*(7*beta1*beta2/12 + 5*beta1/12)
  cp_val_inv = 1/n + alpha * n * (8*beta1+2*beta2)/(5*beta1+3*beta1*beta2+4*beta2**2)*(beta1*beta2/4 + beta2**2/3 + 5*beta1/12)

  return 1/cp_val_inv - 1/np_val_inv # positive means cp is better


def plot_diff_efficiency():

    diff_eff_vals = np.empty((num_grid, num_grid))
    diff_eff_vals[:] = np.nan

    for i in range(num_grid):
        for j in range(i, num_grid):
            diff_eff_vals[i,j] = diff_efficiency(beta1_list[i], beta2_list[j], n, alpha)    

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