import numpy as np
import scipy.stats as st

mix_gauss_params = [[0.33, 20.13, 3.6], [0.54, 29.41, 3.02], [0.13, 34.81, 1.31]]

def log10_viral_load(c, params = mix_gauss_params):
    p=0
    const1 = np.log(10)/0.745
    const2 = 14+np.log10(8)
    for i in range(len(mix_gauss_params)):
        loc_transform = 2*params[i][1] + const1 * (c-const2)
        p += params[i][0] * const1 * st.norm.pdf(loc_transform, params[i][1], params[i][2])

    return p

print([log10_viral_load(i) for i in range(10)])