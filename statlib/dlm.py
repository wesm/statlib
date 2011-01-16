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

    def __init__(self, y, mean_prior=None, var_prior=None, discount=0.9):
        self.y = np.asarray(y)
        self.nobs = len(y)

        self.mean_prior = mean_prior
        self.var_prior = var_prior
        self.disc = discount

        self._compute_parameters()

    def _compute_parameters(self):
        mprior, Cprior = self.mean_prior
        n, d = self.var_prior
        Sprior = d / n

        mu_mode = [mprior]
        mu_scale = [Cprior]
        var_n = [n]
        var_d = [d]

        self.qt = []
        self.st = []

        for i, obs in enumerate(self.y):
            # derive innovation variance from discount factor
            Rt = Cprior / self.disc
            Qt = Rt + Sprior
            At = Rt / Qt

            forc_err = self.y[i] - mprior

            # update obs error parameters
            d = d + Sprior * (forc_err ** 2) / Qt
            n += 1

            # St = Sprior = d / n
            St = Sprior = Sprior + (Sprior / n) * ((forc_err ** 2) / Qt - 1)

            self.qt.append(Qt)
            self.st.append(St)

            # update mean parameters
            mprior = mprior + At * forc_err

            Cprior = At * St

            var_n.append(n)
            var_d.append(d)
            mu_mode.append(mprior)
            mu_scale.append(Cprior)

        self.var_n = np.array(var_n)
        self.var_d = np.array(var_d)
        self.mu_mode = np.array(mu_mode)
        self.mu_scale = np.array(mu_scale)

    def plot_forc(self, alpha=0.05):
        fig = plt.figure()
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

    @property
    def forc_dist(self):
        return stats.t(self.var_n[:-1],
                       loc=self.mu_mode[:-1],
                       scale=self.mu_scale[:-1] / self.disc)

    @property
    def mupost_dist(self):
        pass

    @property
    def forecast(self):
        return self.mu_mode[:-1]

    @property
    def forc_std(self):
        Rt = self.mu_scale[:-1] / self.disc
        Qt = Rt + (self.var_d / self.var_n)[:-1]
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
        return stats.t.pdf(self.y, self.var_n[:-1], loc=self.forecast,
                           scale=self.forc_std)

    @property
    def pred_loglike(self):
        return np.log(self.pred_like).sum()

    def forc_ci(self, alpha=0.05):
        sigma = stats.t(self.var_n[:-1]).ppf(1 - alpha / 2)
        scale = self.forc_std
        upper = self.forecast + sigma * scale
        lower = self.forecast - sigma * scale
        return lower, upper

    def fit(self, discount=0.9):
        pass


class DLMResults(object):

    def __init__(self):
        pass
