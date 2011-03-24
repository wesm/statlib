"""
Notes
-----
References
Ishwaran & James (2001) Gibbs Sampling Methods for Stick-Breaking Priors
"""

import scipy.stats as stats

import numpy as np
import numpy.random as npr

import pymc as pm
import gpustats

import statlib.ffbs as ffbs

class DPNormalMixture(object):
    """
    Truncated Dirichlet Process Mixture of Normals

    Parameters
    ----------

    Notes
    -----

    \alpha ~ Ga(a, b)

    Returns
    -------
    **Attributes**
    """
    # alpha hyperparameters
    e = f = 1
    weights = None
    mu = None
    Sigma = None
    alpha = None
    stick_weights = None

    def __init__(self, data, ncomp=256, alpha0=1, nu0=None, Phi0=None,
                 mu0=None, Sigma0=None, weights0=None, alpha_a0=1,
                 alpha_b0=1):
        self.data = np.asarray(data)
        self.nobs, self.ndim = self.data.shape
        self.ncomp = ncomp

        # TODO hyperparameters
        self.mu_prior_mean = None # prior mean for component means

        # starting values, are these sensible?
        if mu0 is None:
            mu0 = np.zeros((self.ncomp, self.ndim))
        if Sigma0 is None:
            Sigma0 = np.zeros((self.ncomp, self.ndim, self.ndim))
        if Phi0 is None:
            Phi0 = np.empty((self.ncomp, self.ndim, self.ndim))
            Phi0[:] = np.eye(self.ndim)
        if nu0 is None:
            nu0 = np.ones(self.ncomp)

        self._alpha0 = alpha0
        self._alpha_a0 = alpha_a0
        self._alpha_b0 = alpha_b0

        self._weights0 = weights0
        self._mu0 = mu0
        self._Sigma0 = Sigma0
        self._nu0 = nu0 # prior degrees of freedom
        self._Phi0 = Phi0 # prior location for Sigma_j's

        self.gamma = np.ones(ncomp)

    def sample(self, niter=1000, nburn=0, thin=1):
        self._setup_storage(niter)

        alpha = self._alpha0
        weights = self._weights0
        mu = self._mu0
        Sigma = self._Sigma0

        for i in range(-nburn, niter + 1):
            if i % 50 == 0:
                print i

            labels = self._update_labels(mu, Sigma, weights)

            component_mask = _get_mask(labels, self.ncomp)
            counts = component_mask.sum(1)

            stick_weights = self._update_stick_weights(counts, alpha)
            weights = _stick_break(stick_weights)
            alpha = self._update_alpha(stick_weights)
            mu, Sigma = self._update_mu_Sigma(Sigma, counts)

            if i <= 0:
                continue

            self.stick_weights[i] = stick_weights
            self.weights[i] = weights
            self.alpha[i] = alpha
            self.mu[i] = mu
            self.Sigma[i] = Sigma

    def _setup_storage(self, niter=1000, thin=1):
        nresults = niter // thin
        self.weights = np.ones((nresults, self.ncomp))
        self.mu = np.ones((nresults, self.ncomp, self.ndim))
        self.Sigma = np.ones((nresults, self.ncomp, self.ndim, self.ndim))
        self.alpha = np.ones(nresults)
        self.stick_weights = np.ones((nresults, self.ncomp - 1))

    def _update_labels(self, mu, Sigma, weights):
        # GPU business happens?
        densities = gpustats.mvnpdf_multi(self.data, mu, Sigma, weights=weights)
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
        return npr.gamma(a, scale=1 / b)

    def _update_mu_Sigma(self, Sigma, component_mask):
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
            new_mu = pm.rmv_normal_cov(post_mean, post_cov)


            Xj_demeaned = Xj - new_mu

            mu_SS = np.outer(new_mu - mu_hyper, new_mu - mu_hyper) / gam
            data_SS = np.outer(Xj_demeaned, Xj_demeaned)
            post_Phi = data_SS + mu_SS + self._nu0[j] * self._Phi0[j]

            # symmetrize
            post_Phi = (post_Phi + post_Phi.T) / 2

            # P(Sigma) ~ IW(nu + 2, nu * Phi)
            # P(Sigma | theta, Y) ~
            post_nu = nj + self.ncomp + self._nu0 + 3
            new_Sigma = pm.rinverse_wishart(post_nu, post_Phi)

            mu_output[j] = new_mu
            Sigma_output[j] = new_Sigma

        return mu_output, Sigma_output

def _stick_break(V):
    pi = (1 - V).cumprod()
    pi = np.empty(len(V) + 1)
    pi[0] = V[0]

    v_cumprod = (1 - V).cumprod()
    pi[1:-1] = V[1:] * v_cumprod[:-1]
    pi[-1] = v_cumprod[-1]

    return pi

def _get_mask(labels, ncomp):
    return np.equal.outer(np.arange(ncomp), labels)


import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# Generate MV normal mixture

gen_mean = {
    0 : [0, 5],
    1 : [-10, 0],
    2 : [-10, 10]
}

gen_sd = {
    0 : [0.5, 0.5],
    1 : [.5, 1],
    2 : [1, .25]
}

gen_corr = {
    0 : 0.5,
    1 : -0.5,
    2 : 0
}

group_weights = [0.6, 0.3, 0.1]

def generate_data(n=1e5, k=2, ncomps=3, seed=1):
    npr.seed(seed)
    data_concat = []
    labels_concat = []

    for j in range(ncomps):
        mean = gen_mean[j]
        sd = gen_sd[j]
        corr = gen_corr[j]

        cov = np.empty((k, k))
        cov.fill(corr)
        cov[np.diag_indices(k)] = 1
        cov *= np.outer(sd, sd)

        num = int(n * group_weights[j])
        rvs = pm.rmv_normal_cov(mean, cov, size=num)

        data_concat.append(rvs)
        labels_concat.append(np.repeat(j, num))

    return (np.concatenate(labels_concat),
            np.concatenate(data_concat, axis=0))

def plot_2d_mixture(data, labels):
    plt.figure(figsize=(10, 10))
    colors = 'bgr'
    for j in np.unique(labels):
        x, y = data[labels == j].T
        plt.plot(x, y, '%s.' % colors[j], ms=2)


def plot_thetas(sampler):
    plot_2d_mixture(data, true_labels)

    def plot_theta(i):
        x, y = sampler.trace('theta_%d' % i)[:].T
        plt.plot(x, y, 'k.')

    for i in range(3):
        plot_theta(i)

if __name__ == '__main__':
    N = int(1e5) # n data points per component
    K = 2 # ndim
    ncomps = 3 # n mixture components
    true_labels, data = generate_data(n=N, k=K, ncomps=ncomps)

