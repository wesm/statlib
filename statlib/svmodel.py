"""
Univariate stochastic volatility (SV) model
"""
from __future__ import division

from numpy.random import rand
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt

import statlib.distributions as dist
import statlib.ffbs as ffbs
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

    reparameterize
    y_t = z_t + \nu_t
    z_t = \mu + \phi(z_{t-1} - \mu) + \epsilon_t

    Approximate p(\nu_t) \approx \sum_{j=1}^J q_j N(b_j, w_j)
    Defaults to Kim, Shephard, Chib mixture of 7 components
    """
    def __init__(self, data, nu_approx=None):
        if nu_approx is None:
            nu_approx = get_log_chi2_normal_mix()

        # HACK
        self.phi_prior = 0.5, 0.1

        self.nobs = len(data)
        self.data = data
        self.nu_approx = nu_approx

    def sample(self, niter=1000, nburn=0, thin=1):
        self._setup_storage(niter, nburn, thin)

        # mixture component selected
        gamma = np.zeros(self.nobs)
        mu = 0
        phi = 0.9
        v = 1
        z = np.zeros(self.nobs + 1) # volatility sequence

        for i in range(-nburn, niter):
            if i % 50 == 0:
                print i

            gamma = self._sample_gamma(z)
            phi = self._sample_phi(phi, z, mu, v)
            mu = self._sample_mu()
            v = self._sample_v()
            z = self._sample_z()

            if i % thin == 0 and i >= 0:
                k = i // thin
                self.gamma[k] = gamma
                self.phi[k] = phi
                self.mu[k] = mu
                self.v[k] = v
                self.z[k] = z

    def _setup_storage(self, niter, nburn, thin):
        nresults = niter // thin

        self.z = np.zeros((nresults, self.nobs + 1))
        self.gamma = np.zeros((nresults, self.nobs + 1))

        self.mu = np.zeros(nresults)
        self.phi = np.zeros(nresults)
        self.v = np.zeros(self.nobs + 1)

    def _sample_gamma(self, z):
        mix = self.nu_approx

        probs = np.empty((len(self.mix.weights), self.nobs))

        i = 0
        for q, m, w in zip(mix.weights, mix.means, mix.variances):
            probs[i] = q * np.exp(-0.5*(self.y-m-z[1:])**2 / w) / np.sqrt(w)
            i += 1

        # bleh, super slow
        probs /= probs.sum(0)
        gamma = np.empty(self.nobs)
        for i, p in enumerate(probs.T):
            gamma[i] = sample_measure(p)
        return gamma

    def _sample_phi(self, phi, z, mu, v):
        """
        Sample phi from truncated normal with Metropolis-Hastings acceptance
        step
        """
        # MH step
        prior_m, prior_v = self.phi_prior

        r = z - mu
        sumsq = (r[:-1] ** 2).sum()
        A = v * sumsq

        prop_mean = ((r[1:] * r[:-1]).sum() / sumsq) / A + prior_m / prior_v
        prop_var = 1 / A + 1 / prior_v
        proposal = dist.rtrunc_norm(prop_mean, np.sqrt(prop_var), 0, 1)

        a_prop = 0
        a_phi = 0

        if rand() < a_prop / a_phi:
            return proposal
        else:
            return phi

    def _sample_mu(self, gamma, phi, v):
        # Invoke Cython code
        mu = ffbs.univ_sv_ffbs(self.y, phi,
                               self.nu_approx.means,
                               self.nu_approx.variances,
                               gamma, v)

        return mu

    def _sample_v(self):
        pass

    def _sample_z(self):
        # FFBS
        pass

def sample_measure(weights):
    return weights.cumsum().searchsorted(rand())

def rmvnorm(mu, cov, size=1):
    """
    Compute multivariate normal draws from possibly singular covariance matrix
    using singular value decomposition (SVD)
    """
    # TODO: find reference
    U, s, Vh = np.linalg.svd(cov)
    cov_sqrt = np.dot(U * np.sqrt(s), Vh)

    # N(0, 1) draws
    draws = np.random.randn(size, len(cov))
    return np.dot(draws, cov_sqrt) + mu

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
