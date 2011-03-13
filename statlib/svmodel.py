"""
Univariate stochastic volatility (SV) model
"""

import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt

import statlib.plotting as plotting

class SVModel(object):
    """
    Univariate stochastic volatility (SV) model using normal approximation

    Parameters
    ----------


    Notes
    -----
    r_t ~ N(0, \sigma_t^2)
    \sigma_t = exp(\mu + x_t)
    x_t = \phi x_{t-1} + \epsilon_t
    \epsilon_t ~ N(0, v)

    y_t = log(r_t^2) / 2
    y_t = \mu + x_t + \nu_t, \nu_t = \log(\kappa_t) / 2
    \kappa_t ~ \chi^2_1

    Approximate p(\nu_t) \approx \sum_{j=1}^J q_j N(b_j, w_j)
    """
    def __init__(self, data):
        pass

# TODO arbitrary mixture...e.g. betas or gammas

class NormalMixture(object):
    """
    Simple class for storing mixture of some distributions

    Parameters
    ----------
    """
    def __init__(self, means, variances, weights):
        if not (len(means) == len(variances) == len(weights)):
            raise ValueError('inputs must be all same length sequences')

        self.means = np.asarray(means)
        self.variances = np.asarray(variances)
        self.weights = np.asarray(weights)

        self.dists = [stats.norm(m, np.sqrt(v))
                      for m, v in zip(self.means, self.variances)]

    def pdf(self, x):
        """
        Evaluate mixture probability density function
        """
        density = 0
        for w, dist in zip(self.weights, self.dists):
            density += w * dist.pdf(x)

        return density

    def plot(self, style='k', ax=None, **plot_kwds):
        max_sd = np.sqrt(self.variances.max())

        # overkill perhaps
        lo = self.means.min() - 4 * max_sd
        hi = self.means.max() + 4 * max_sd

        if ax is None:
            ax = plt.gca()

        plotting.plot_support(self.pdf, lo, hi, style=style,
                              **plot_kwds)

def compare_mix_with_logchi2(draws=5000):
    # make a graph showing the mixture approx

    plt.figure()

    mixture = get_log_chi2_normal_mix()

    rv = stats.chi2(1)
    draws = np.log(rv.rvs(5000)) / 2

    plotting.density_plot(draws, style='k--', label=r'$\log(\chi^2_1) / 2$')
    mixture.plot(style='k', label='Mixture')

    plt.legend(loc='best')

def get_log_chi2_normal_mix():
    # Kim, Shephard, Chib (1998) approx of log \chi^2_1
    weights = [0.0073, 0.0000, 0.1056, 0.2575, 0.3400, 0.2457, 0.0440]
    means = [-5.7002, -4.9186, -2.6216, -1.1793, -0.3255, 0.2624, 0.7537]
    variances = [1.4490, 1.2949, 0.6534, 0.3157, 0.1600, 0.0851, 0.0418]
    return NormalMixture(means, variances, weights)

if __name__ == '__main__':
    import statlib.datasets as ds

    data = ds.fx_gbpusd()

    mixture = get_log_chi2_normal_mix()
