"""
Implementing random things from West & Harrison
"""

from datetime import datetime
import re

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pandas.util.testing import debug, set_trace as st

class DLM(object):
    """
    Simple univariate Bayesian Dynamic Linear Model with discount factor
    """

    def __init__(self, y, x, mean_prior=None, var_prior=None, discount=0.9):
        self.y, self.x = map(np.array, np.atleast_1d(y, x))
        self.nobs = len(y)

        self.mean_prior = mean_prior
        self.var_prior = var_prior
        self.disc = discount

        self._compute_parameters()

    def _compute_parameters(self):
        results = linear_dlm(self.y, self.x, self.mean_prior, self.var_prior,
                             disc=self.disc)

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
        return self.mu_mode[:-1] * self.x

    @property
    def forc_std(self):
        Rt = self.mu_scale[:-1] / self.disc
        Qt = self.x ** 2 * Rt + self.var_est[:-1]
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


def make_t_ci(df, level, scale, alpha=0.10):
    sigma = stats.t(df).ppf(1 - alpha / 2)
    upper = level + sigma * scale
    lower = level - sigma * scale
    return lower, upper

!
def linear_dlm(y, x, mean_prior, var_prior, disc=0.9):
    """

    Parameters
    ----------

    Returns
    -------

    """
    mprior, Cprior = mean_prior
    n, d = var_prior
    Sprior = d / n

    mu_mode = [mprior]
    mu_scale = [Cprior]
    df = [n]
    var_est = [d / n]

    for i, obs in enumerate(y):
        # derive innovation variance from discount factor

        Ft = x[i]

        Rt = Cprior / disc
        Qt = Ft ** 2 * Rt + Sprior
        At = Rt * Ft / Qt

        forc_err = y[i] - Ft * mprior

        # update obs error parameters
        n += 1
        St = Sprior = Sprior + (Sprior / n) * ((forc_err ** 2) / Qt - 1)

        # update mean parameters
        mprior = mprior + At * forc_err

        Cprior = Rt * St / Qt

        df.append(n)
        var_est.append(Sprior)
        mu_mode.append(mprior)
        mu_scale.append(Cprior)

    return {
        'df' : np.array(df),
        'var_est' : np.array(var_est),
        'mu_mode' : np.array(mu_mode),
        'mu_scale' : np.array(mu_scale)
    }

class DLMResults(object):

    def __init__(self):
        pass
