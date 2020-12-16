from scipy.optimize import fsolve
import numpy as np

US_DIST = [0.2837, 0.3451, 0.1507, 0.1276, 0.0578, 0.0226, 0.0125]

# copied from Massey
def match_prevalence(p_index, target_prevalence, household_dist, SAR):
    # computes probability of a primary case given population level prevalence, household size distribution,
    # and household secondary attack rate

    # INPUT:
    # p_index = probability of a primary case in the household
    # target_prevalence = population level prevalence
    # household_dist = probability distribution of household sizes 1,2,3,...
    # SAR = household secondary attack rate

    assert(np.absolute(np.sum(household_dist) - 1) < 1e-6)

    exp_household_size = 0
    for i in range(len(household_dist)):
        exp_household_size += (i + 1) * household_dist[i]

    frac_tot_infected = 0
    for i in range(len(household_dist)):
        frac_tot_infected += (i + 1) * (p_index + SAR * (1 - p_index) - SAR * (1 - p_index) ** (i + 1)) * household_dist[
            i] / exp_household_size

    return frac_tot_infected - target_prevalence


def eval_p_index(match_prevalence, target_prevalence, household_dist=US_DIST, SAR=0.3741):
    return fsolve(match_prevalence, 0.005, args=(target_prevalence, household_dist, SAR))


if __name__ == '__main__':
    print("find p_index for US household distribution, target prevalence = 0.005: " + str(eval_p_index(match_prevalence, 0.005)))
    print("find p_index for US household distribution, target prevalence = 0.01: " + str(eval_p_index(match_prevalence, 0.01)))
