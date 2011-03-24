"""
Notes
-----
References
Ishwaran & James (2001) Gibbs Sampling Methods for Stick-Breaking Priors
"""

import scipy.stats as stats

import numpy as np
import pymc
import pymc.flib as flib
import gpustats

import statlib.ffbs as ffbs

class DPNormalMixture(object):
    """
    Truncated Dirichlet Process Mixture of Normals

    Parameters
    ----------

    Notes
    -----

    Returns
    -------
    **Attributes**
    """
    # alpha hyperparameters
    e = f = 1

    def __init__(self, data, ncomp=256):
        self.data = np.asarray(data)
        self.nobs, self.ndim = self.data.shape
        self.ncomp = ncomp

        # TODO hyperparameters
        self.mu_prior_mean = None # prior mean for component means
        self.nu = None # prior degrees of freedom
        self.Phi = None # prior location for Sigma_j's
        self.gamma = None

    def sample(self, niter=1000, nburn=0, thin=1):
        self._setup_storage(niter)

        for i in range(-nburn, niter):
            if i % 50 == 0:
                print i

            labels = self._update_labels(mu, Sigma, weights)

            component_mask = _get_mask(labels, self.ncomp)
            counts = component_mask.sum(1)

            stick_weights = self._update_stick_weights(counts, alpha)
            weights = stick_break(stick_weights)
            alpha = self._update_alpha(V)

            mu, Sigma = self._update_mu_Sigma(labels, Sigma, counts)

    def _setup_storage(self, niter=1000, thin=1):
        self.weights = np.ones((niter, self.ncomp))
        self.mu = np.ones((niter, self.ncomp, self.ndim))
        self.Sigma = np.ones((niter, self.ncomp, self.ndim, self.ndim))
        self.alpha = np.ones(niter)

        self.stick_weights = np.ones((niter, self.ncomp - 1))

    def _update_labels(self, mu, Sigma, weights):
        # GPU business happens?
        densities = gpustats.mvnpdf_multi(self.data, mu, Sigma)
        densities = (densities.T / densities.sum(1)).T

        # convert this to run in the GPU
        return ffbs.sample_discrete(densities)

    def _update_stick_weights(self, counts, alpha):
        reverse_cumsum = counts[::-1].cumsum()[::-1]
        dist = stats.beta(1 + counts[:-1], alpha + reverse_cumsum[1:])
        return dist.rvs(self.nobs - 1)

    def _update_alpha(self, V):
        a = self.ncomp + self.e - 1
        b = self.f - np.log(1 - V).sum()
        return np.random.gamma(a, scale=1 / b)

    def _update_mu_Sigma(self, labels, Sigma, component_mask):
        mu_output = np.zeros((self.ncomp, self.ndim))
        Sigma_output = np.zeros((self.ncomp, self.ndim, self.ndim))

        for j in xrange(self.ncomp):
            mask = component_mask[j]
            Xj = self.data[mask]
            nj = len(Xj)

            # TODO: sample from prior if nj == 0
            sumxj = Xj.sum(0)

            gam = self.gamma[j]
            mu_hyper = self.mu_prior_mean[j]

            post_mean = (mu_hyper / gam + sumxj) / (1 / gam + nj)
            post_cov = 1 / (1 / gam + nj) * Sigma
            new_mu = pymc.rmv_normal_cov(post_mean, post_cov)

            mu_SS = np.outer(new_mu - mu_hyper, new_mu - mu_hyper) / gam
            Xj_demeaned = Xj - new_mu
            data_SS = (np.outer(Xj_demeaned, Xj_demeaned)
            post_Phi = data_SS + mu_SS + self.nu[j] * self.Phi[j])
            flib.symmetrize(post_Phi)

            # P(Sigma) ~ IW(nu + 2, nu * Phi)
            post_nu = nj + self.nu + 3
            new_Sigma = pymc.rinverse_wishart(post_nu, post_Phi)

            mu_output[j] = new_mu
            Sigma_output[j] = new_Sigma

        return mu_output, Sigma_output

def stick_break(V):
    pi = (1 - V).cumprod()
    pi = np.empty(len(V) + 1)
    pi[0] = V[0]

    v_cumprod = (1 - V).cumprod()
    pi[1:-1] = V[1:] * v_cumprod[:-1]
    pi[-1] = v_cumprod[-1]

    return pi

def _get_mask(labels, ncomp):
    return np.equal.outer(np.arange(ncomp), labels)



if __name__ == '__main__':
    pass
