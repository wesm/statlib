"""
Implementing random things from West & Harrison
"""

from datetime import datetime
import re

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from pandas.util.testing import debug, set_trace as st
from statlib.tools import chain_dot

class DLM(object):
    """
    Simple univariate Bayesian Dynamic Linear Model with discount factor

    Parameters
    ----------
    y : ndarray n x 1
        Response variable
    F : ndarray n x k
        Regressor matrix
    G : ndarray k x k
        State transition matrix
    mean_prior : tuple (mean, variance)
        mean: length k
        variance: k x k
        Normal prior for mean response
    var_prior : tuple (a, b)
        Inverse-gamma prior for observation variance
    discount : float
        Wt = Ct * (1 - d) / d
    """

    def __init__(self, y, F, G=None, mean_prior=None, var_prior=None,
                 discount=0.9):
        self.y = np.array(y)
        self.nobs = len(y)

        if self.y.ndim == 1:
            pass
        else:
            raise Exception

        F = np.array(F)
        if F.ndim == 1:
            F = F.reshape((len(F), 1))
        self.F = F

        self.ndim = self.F.shape[1]

        if G is None:
            G = np.eye(self.ndim)
        self.G = G

        self.mean_prior = mean_prior
        self.var_prior = var_prior
        self.disc = discount

        self._compute_parameters()

    def _compute_parameters(self):
        results = linear_dlm(self.y, self.F, self.G, self.mean_prior,
                             self.var_prior, disc=self.disc)

        self.df = results['df']
        self.var_est = results['var_est']
        self.mu_mode = results['mu_mode']
        self.mu_scale = results['mu_scale']

    def plot_forc(self, alpha=0.10, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.subplot(111)

        rng = np.arange(self.nobs)

        ci_lower, ci_upper = self.forc_ci()

        ax.plot(rng, self.y, 'k.')
        ax.plot(rng, self.forecast, 'k--')

        ax.plot(rng, ci_lower, 'k-.')
        ax.plot(rng, ci_upper, 'k-.')

        ptp = np.ptp(self.y)
        ylim = (self.y.min() - 1 * ptp, self.y.max() + 1 * ptp)
        ax.set_ylim(ylim)

    def plot_mu(self, alpha=0.10, prior=True, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.subplot(111)

        level, (ci_lower, ci_upper) = self.mu_ci(prior=prior, alpha=alpha)

        rng = np.arange(self.nobs)
        ax.plot(rng, level, 'k--')
        ax.plot(rng, ci_lower, 'k-.')
        ax.plot(rng, ci_upper, 'k-.')

        ptp = np.ptp(level)
        ylim = (level.min() - 1 * ptp, level.max() + 1 * ptp)
        ax.set_ylim(ylim)

    @property
    def forc_dist(self):
        return stats.t(self.df[:-1],
                       loc=self.mu_mode[:-1],
                       scale=self.mu_scale[:-1] / self.disc)

    @property
    def mupost_dist(self):
        pass

    @property
    def forecast(self):
        return (self.F * self.mu_mode[:-1]).sum(1)

    @property
    def forc_std(self):
        Rt = self.mu_scale[:-1] / self.disc
        Qt = self.F ** 2 * Rt + self.var_est[:-1]
        return np.sqrt(Qt)

    @property
    def rmse(self):
        forc_err = self.y - self.forecast
        return np.sqrt((forc_err ** 2).sum() / self.nobs)

    @property
    def mad(self):
        forc_err = self.y - self.forecast
        return np.abs(forc_err).sum() / self.nobs

    @property
    def pred_like(self):
        return stats.t.pdf(self.y, self.df[:-1], loc=self.forecast,
                           scale=self.forc_std)

    @property
    def pred_loglike(self):
        return np.log(self.pred_like).sum()

    def mu_ci(self, alpha=0.10, prior=True):
        """
        If prior is False, compute posterior
        """
        if prior:
            df = self.df[:-1]
            level = self.mu_mode[:-1]
            scale = np.sqrt(self.mu_scale[:-1] / self.disc)
        else:
            df = self.df[1:]
            level = self.mu_mode[1:]
            scale = np.sqrt(self.mu_scale[1:])

        return level, make_t_ci(df, level, scale, alpha=alpha)

    def forc_ci(self, alpha=0.05):
        return make_t_ci(self.df[:-1], self.forecast, self.forc_std,
                         alpha=alpha)

    def fit(self, discount=0.9):
        pass


import unittest

class TestDLM(unittest.TestCase):

    def setUp(self):
        pass

def make_t_ci(df, level, scale, alpha=0.10):
    sigma = stats.t(df).ppf(1 - alpha / 2)
    upper = level + sigma * scale
    lower = level - sigma * scale
    return lower, upper

def _result_array(*shape):
    arr = np.empty(shape, dtype=float)
    arr.fill(np.NaN)

    return arr

def linear_dlm(y, F, G, mean_prior, var_prior, disc=0.9):
    """

    Parameters
    ----------

    Returns
    -------

    """

    nobs = len(y)
    k = F.shape[1]

    mode = _result_array(nobs + 1, k)
    C = _result_array(nobs + 1, k, k)
    df = _result_array(nobs + 1)
    S = _result_array(nobs + 1)

    mode[0], C[0] = mean_prior
    n, d = var_prior
    df[0] = n
    S[0] = d / n

    for i, obs in enumerate(y):
        # column vector, for W&H notational consistency
        Ft = F[i, :].T

        # advance index: y_1 through y_nobs, 0 is prior
        t = i + 1

        # derive innovation variance from discount factor

        Rt = chain_dot(G, C[t - 1], G.T) / disc
        Qt = chain_dot(Ft.T, Rt, Ft) + S[t-1]

        # forecast theta as time t
        a_t = np.dot(G, mode[t - 1])
        f_t = np.dot(Ft.T, a_t)

        forc_err = obs - f_t

        At = np.dot(Rt, Ft) / Qt

        # update mean parameters
        mode[t] = a_t + np.dot(At, forc_err)
        S[t] = S[t-1] + (S[t-1] / n) * ((forc_err ** 2) / Qt - 1)
        C[t] = (S[t] / S[t-1]) * (Rt - np.dot(At, At.T) * Qt)
        df[t] = df[t - 1] + 1

    return {
        'df' : np.array(df),
        'var_est' : S,
        'mu_mode' : mode,
        'mu_scale' : C
    }

class DLMResults(object):

    def __init__(self):
        pass

if __name__ == '__main__':
    pass
