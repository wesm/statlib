from __future__ import division

from numpy.linalg import inv
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

import scipy.stats as stats

from scikits.statsmodels.tools.decorators import cache_readonly
from scikits.statsmodels.tools.tools import chain_dot as cdot

import statlib.plotting as plotting

from pandas.util.testing import set_trace as st

class BayesLM(object):
    """
    Bayesian linear model with independent errors

    Normal-InverseGamma prior

    Parameters
    ----------
    Y : ndarray (length n)
        response variable
    X : ndarray (n x p)
        independent variables
    beta_prior : (mean, cov)
        Prior for coefficients: Normal(mean, cov)
    var_prior : (df, df * phi_0)
        df degrees of freedom
        phi_0 = 1 / var_0^2 (prior precision)
        Prior for error variance: InverseGamma(df /2, df * phi_0 / 2)

    Notes
    -----
    y = X'b + e
    e ~ N(0, v * I_n)
    b | v ~ N(m_0, v * C_0)
    v ~ IG(n0 / 2, d0 / 2)
    """
    def __init__(self, Y, X, beta_prior, var_prior):
        self.y = Y
        self.x = X

        self.nobs = len(Y)

        self.m0, self.c0 = beta_prior
        self.n0, self.d0 = var_prior

    @cache_readonly
    def beta_post_params(self):
        X = self.x

        Q = self.Q
        Qinv = inv(Q)

        A = cdot(self.c0, X.T, Qinv)

        m = self.m0 + np.dot(A, self.prior_resid)
        C = self.c0 - cdot(A, Q, A.T)

        return m, C

    @property
    def sigma2(self):
        """
        mean posterior estimate of error variance
        """
        n, d = self.var_post_params
        return d / n

    @property
    def params(self):
        """
        Mode of posterior t distribution for params
        """
        return self.beta_post_params[0]

    def beta_ci(self, alpha=0.05):
        df, _ = self.var_post_params
        mode, scale = self.beta_post_params

        std = np.sqrt(self.sigma2 * np.diag(scale))

        q = stats.t(df).ppf(1 - alpha / 2)

        ci_lower = mode - q * std
        ci_upper = mode + q * std

        return ci_lower, ci_upper

    def prec_ci(self, alpha=0.05, hpd=True, hpd_draws=100000):
        """
        Computes HPD or regular frequentist interval
        """
        from statlib.tools import quantile

        if hpd:
            n, d = self.var_post_params
            prec_post = stats.gamma(n / 2., scale=2./d)
            prec_draws = prec_post.rvs(hpd_draws)
            qs = quantile(prec_draws, [alpha / 2, 0.5, 1 - alpha / 2])
            hpd_lower, median, hpd_upper = qs

            return hpd_lower, median, hpd_upper
        else:
            raise NotImplementedError

    @cache_readonly
    def Q(self):
        return cdot(self.x, self.c0, self.x.T) + np.eye(self.nobs)

    @cache_readonly
    def var_post_params(self):
        post_n = self.nobs + self.n0

        resid = self.prior_resid
        post_d = np.dot(resid, LA.solve(self.Q, resid)) + self.d0

        return post_n, post_d

    @property
    def precision_post_dist(self):
        post_n, post_d = self.var_post_params
        return stats.gamma(post_n, scale=2. / post_d)

    @property
    def var_post_dist(self):
        post_n, post_d = self.var_post_params
        return stats.invgamma(post_n, scale=post_d / 2.)

    @cache_readonly
    def prior_resid(self):
        return self.y - np.dot(self.x, self.m0)

    def sample(self, samples=1000):
        """
        Generate Monte Carlo samples of posterior distribution of parameters

        Samples v, then b | v, with standard normal-inverse-gamma full
        conditional distributions.

        Parameters
        ----------
        samples : int
        """
        precision_samples = self.precision_post_dist.rvs(samples)
        var_samples = 1 / precision_samples

        # sample beta condition on v
        beta_samples = []

        m, C = self.beta_post_params

        for v in var_samples:
            rv = np.random.multivariate_normal(m, v * C)
            beta_samples.append(rv)

        return var_samples, np.array(beta_samples)

    def plot_forecast(self, x, ax=None, label=None, **kwds):
        """
        Plot posterior predictive distribution for input predictor values

        y ~ N(Xb, s^2)

        Parameters
        ----------
        x : 1d
        """
        if ax is None:
            ax = plt.gca()

        mode, scale = self.beta_post_params

        s2 = self.sigma2
        # forecast variance
        rQ = np.sqrt(s2 * (np.dot(x, np.dot(scale, x)) + 1))
        m = np.dot(x, self.params)

        dist = stats.norm(m, scale=rQ)

        plotting.plot_support(dist.pdf, m - 5 * rQ, m + 5 * rQ, ax=ax)

        if label is not None:
            y = dist.pdf(m)
            x = m
            ax.text(x, y, label, **kwds)

    def posterior_pred_dist(self, x):
        mode, scale = self.beta_post_params

        s2 = self.sigma2
        # forecast variance
        rQ = np.sqrt(s2 * (np.dot(x, np.dot(scale, x)) + 1))
        m = np.dot(x, self.params)

        return stats.norm(m, scale=rQ)

