from scipy.optimize import fsolve
import numpy as np
from household_dist import HOUSEHOLD_DIST


def compute_household_infection_prob(prevalence, household_dist, SAR=0.3741):
    """
    computes the probability that a household is infected given population level prevalence,
    household size distribution and household secondary attack rate

    INPUT:
    prevalence = population level prevalence
    household_dist = array-like, probability distribution of household sizes 1, 2, 3, ...
    SAR = household secondary attack rate
    """
    assert(np.absolute(np.sum(household_dist) - 1) < 1e-6)

    exp_household_size = 0
    exp_household_infection_multiplier = 0
    for i in range(len(household_dist)):
        exp_household_size += (i + 1) * household_dist[i]
        exp_household_infection_multiplier += (1 + (i + 1 - 1) * SAR) * household_dist[i]

    p = prevalence * exp_household_size / exp_household_infection_multiplier
    return p


# deprecated, modified from Massey's groupt testing code
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


# deprecated, modified from Massey's groupt testing code
def eval_p_index(match_prevalence, target_prevalence, household_dist, SAR=0.3741):
    return fsolve(match_prevalence, 0.005, args=(target_prevalence, household_dist, SAR))


if __name__ == '__main__':
    household_dist = HOUSEHOLD_DIST['US_DIST']
    print("household infection probability (US population): " + str(compute_household_infection_prob(0.01, household_dist)))
    print("household infection probability (household size = 3): " + str(compute_household_infection_prob(0.01, household_dist=[0,0,1])))
