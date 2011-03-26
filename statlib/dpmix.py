"""
Notes
-----
References
Ishwaran & James (2001) Gibbs Sampling Methods for Stick-Breaking Priors
"""
from __future__ import division

import matplotlib.pyplot as plt

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
    data : ndarray (nobs x ndim)
    ncomp : int
        Number of mixture components

    Notes
    -----
    y ~ \sum_{j=1}^J \pi_j {\cal N}(\mu_j, \Sigma_j)
    \alpha ~ Ga(e, f)
    \Sigma_j ~ IW(nu0 + 2, nu0 * \Phi_j)

    Returns
    -------
    **Attributes**
    """

    def __init__(self, data, ncomp=256, alpha0=1, nu0=None, Phi0=None,
                 mu0=None, Sigma0=None, weights0=None, alpha_a0=1,
                 alpha_b0=1):
        self.data = np.asarray(data)
        self.nobs, self.ndim = self.data.shape
        self.ncomp = ncomp

        # TODO hyperparameters
        # prior mean for component means
        self.mu_prior_mean = np.zeros(self.ndim)
        self.gamma = np.ones(ncomp)

        self._set_initial_values(alpha0, nu0, Phi0, mu0, Sigma0,
                                 weights0, alpha_a0, alpha_b0)

    def _set_initial_values(self, alpha0, nu0, Phi0, mu0, Sigma0, weights0,
                            alpha_a0, alpha_b0):
        if nu0 is None:
            nu0 = 3

        if Phi0 is None:
            Phi0 = np.empty((self.ncomp, self.ndim, self.ndim))
            Phi0[:] = np.eye(self.ndim) * (nu0 - 1)

        if Sigma0 is None:
            # draw from prior
            Sigma0 = np.empty((self.ncomp, self.ndim, self.ndim))
            for j in xrange(self.ncomp):
                Sigma0[j] = pm.rinverse_wishart_prec(nu0 + 2 + self.ncomp, Phi0[j])

        # starting values, are these sensible?
        if mu0 is None:
            mu0 = np.empty((self.ncomp, self.ndim))
            for j in xrange(self.ncomp):
                mu0[j] = pm.rmv_normal_cov(self.mu_prior_mean,
                                           self.gamma[j] * Sigma0[j])

        if weights0 is None:
            _, weights0 = stick_break_proc(1, 1, size=self.ncomp - 1)

        self._alpha0 = alpha0
        self._alpha_a0 = alpha_a0
        self._alpha_b0 = alpha_b0

        self._weights0 = weights0
        self._mu0 = mu0
        self._Sigma0 = Sigma0
        self._nu0 = nu0 # prior degrees of freedom
        self._Phi0 = Phi0 # prior location for Sigma_j's

    def sample(self, niter=1000, nburn=0, thin=1):
        self._setup_storage(niter)

        alpha = self._alpha0
        weights = self._weights0
        mu = self._mu0
        Sigma = self._Sigma0

        ax = plt.gca()

        for i in range(-nburn, niter):
            labels = self._update_labels(mu, Sigma, weights)

            component_mask = _get_mask(labels, self.ncomp)
            counts = component_mask.sum(1)

            stick_weights, weights = self._update_stick_weights(counts, alpha)

            alpha = self._update_alpha(stick_weights)
            mu, Sigma = self._update_mu_Sigma(Sigma, component_mask)

            if i % 50 == 0:
                print i, counts
                print np.c_[mu, weights]

            '''
                for j in xrange(self.ncomp):
                    ax.plot(self.weights[:i, j])
                plt.show()
                plt.draw_if_interactive()
            '''

            if i < 0:
                continue

            self.stick_weights[i] = stick_weights
            self.weights[i] = weights
            self.alpha[i] = alpha
            self.mu[i] = mu
            self.Sigma[i] = Sigma

    # so pylint won't complain so much
    # alpha hyperparameters
    e = f = 1
    weights = None
    mu = None
    Sigma = None
    alpha = None
    stick_weights = None

    def _setup_storage(self, niter=1000, thin=1):
        nresults = niter // thin
        self.weights = np.zeros((nresults, self.ncomp))
        self.mu = np.zeros((nresults, self.ncomp, self.ndim))
        self.Sigma = np.zeros((nresults, self.ncomp, self.ndim, self.ndim))
        self.alpha = np.zeros(nresults)
        self.stick_weights = np.zeros((nresults, self.ncomp - 1))

    def _update_labels(self, mu, Sigma, weights):
        # GPU business happens?
        densities = gpustats.mvnpdf_multi(self.data, mu, Sigma, weights=weights)
        densities = (densities.T / densities.sum(1)).T

        # convert this to run in the GPU
        return ffbs.f32_sample_discrete(densities)

    def _update_stick_weights(self, counts, alpha):
        """

        """
        reverse_cumsum = counts[::-1].cumsum()[::-1]

        a = 1 + counts[:-1]
        b = alpha + reverse_cumsum[1:]

        stick_weights, mixture_weights = stick_break_proc(a, b)
        return stick_weights, mixture_weights

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
            mu_hyper = self.mu_prior_mean

            post_mean = (mu_hyper / gam + sumxj) / (1 / gam + nj)
            post_cov = 1 / (1 / gam + nj) * Sigma[j]

            new_mu = pm.rmv_normal_cov(post_mean, post_cov)

            Xj_demeaned = Xj - new_mu

            mu_SS = np.outer(new_mu - mu_hyper, new_mu - mu_hyper) / gam
            data_SS = np.dot(Xj_demeaned.T, Xj_demeaned)
            post_Phi = data_SS + mu_SS + self._nu0 * self._Phi0[j]

            # symmetrize
            post_Phi = (post_Phi + post_Phi.T) / 2

            # P(Sigma) ~ IW(nu + 2, nu * Phi)
            # P(Sigma | theta, Y) ~
            post_nu = nj + self.ncomp + self._nu0 + 3

            # pymc rinverse_wishart takes
            new_Sigma = pm.rinverse_wishart_prec(post_nu, post_Phi)

            mu_output[j] = new_mu
            Sigma_output[j] = new_Sigma

        return mu_output, Sigma_output

def stick_break_proc(beta_a, beta_b, size=None):
    """
    Kernel stick breaking procedure for truncated Dirichlet Process

    Parameters
    ----------
    beta_a : scalar or array-like
    beta_b : scalar or array-like
    size : int, default None
        If array-like a, b, leave as None

    Notes
    -----


    Returns
    -------
    (stick_weights, mixture_weights) : (1d ndarray, 1d ndarray)
    """
    if not np.isscalar(beta_a):
        size = len(beta_a)
        assert(size == len(beta_b))
    else:
        assert(size is not None)

    dist = stats.beta(beta_a, beta_b)
    V = stick_weights = dist.rvs(size)
    pi = mixture_weights = np.empty(len(V) + 1)

    pi[0] = V[0]
    prod = (1 - V[0])
    for k in xrange(2, len(V)):
        pi[k] = prod * V[k]
        prod *= 1 - V[k]
    pi[-1] = prod

    '''
    v_cumprod = (1 - V).cumprod()
    pi[1:-1] = V[1:] * v_cumprod[:-1]
    pi[-1] = v_cumprod[-1]
    '''
    return stick_weights, mixture_weights

def _get_mask(labels, ncomp):
    return np.equal.outer(np.arange(ncomp), labels)


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
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    colors = 'bgr'
    for j in np.unique(labels):
        x, y = data[labels == j].T
        plt.plot(x, y, '%s.' % colors[j], ms=2)

if __name__ == '__main__':
    N = int(1e4) # n data points per component
    K = 2 # ndim
    ncomps = 3 # n mixture components
    npr.seed(1)
    true_labels, data = generate_data(n=N, k=K, ncomps=ncomps)

    model = DPNormalMixture(data, ncomp=256)
    model.sample(2000)
