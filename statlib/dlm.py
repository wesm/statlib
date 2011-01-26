"""
Implementing random things from West & Harrison
"""
from __future__ import division


from datetime import datetime
import re

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.special as special
from scipy.special import gammaln as gamln

from pandas.util.testing import debug, set_trace as st
from statlib.tools import chain_dot

def nct_pdf(x, df, nc):
    from numpy import sqrt, log, exp

    import pdb
    pdb.set_trace()

    n = df*1.0
    nc = nc*1.0
    x2 = x*x
    ncx2 = nc*nc*x2
    fac1 = n + x2
    trm1 = n/2.*log(n) + gamln(n+1)
    trm1 -= n*log(2)+nc*nc/2.+(n/2.)*log(fac1)+gamln(n/2.)
    Px = exp(trm1)
    valF = ncx2 / (2*fac1)
    trm1 = sqrt(2)*nc*x*special.hyp1f1(n/2+1,1.5,valF)
    trm1 /= arr(fac1*special.gamma((n+1)/2))
    trm2 = special.hyp1f1((n+1)/2,0.5,valF)
    trm2 /= arr(sqrt(fac1)*special.gamma(n/2+1))
    Px *= trm1+trm2
    return Px

class DLM(object):
    """
    Bayesian Gaussian Dynamic Linear Model (DLM) with discount factor

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

        self.mu_mode = _result_array(self.nobs + 1, self.ndim)
        self.mu_scale = _result_array(self.nobs + 1, self.ndim, self.ndim)
        self.df = _result_array(self.nobs + 1)
        self.var_est = _result_array(self.nobs + 1)
        self.forc_var = _result_array(self.nobs)
        self.R = _result_array(self.nobs + 1, self.ndim, self.ndim)
        self.ncp = _result_array(self.nobs)

        self.mu_mode[0], self.mu_scale[0] = mean_prior
        n, d = var_prior
        self.df[0] = n
        self.var_est[0] = d / n

        self._compute_parameters()

    def _compute_parameters(self):
        """
        Compute parameter estimates for Gaussian Univariate DLM

        Parameters
        ----------

        Notes
        -----
        West & Harrison pp. 111-112

        Returns
        -------

        """
        # allocate result arrays
        mode = self.mu_mode
        C = self.mu_scale
        df = self.df
        S = self.var_est
        F = self.F
        G = self.G

        for i, obs in enumerate(self.y):
            # column vector, for W&H notational consistency
            Ft = F[i:i+1].T

            # advance index: y_1 through y_nobs, 0 is prior
            t = i + 1

            # derive innovation variance from discount factor

            Rt = chain_dot(G, C[t - 1], G.T) / self.disc
            Qt = chain_dot(Ft.T, Rt, Ft) + S[t-1]

            # forecast theta as time t
            a_t = np.dot(G, mode[t - 1])
            f_t = np.dot(Ft.T, a_t)
            forc_err = obs - f_t
            At = np.dot(Rt, Ft) / Qt

            # update mean parameters
            df[t] = df[t - 1] + 1
            mode[t] = a_t + np.dot(At, forc_err)
            S[t] = S[t-1] + (S[t-1] / df[t]) * ((forc_err ** 2) / Qt - 1)
            C[t] = (S[t] / S[t-1]) * (Rt - np.dot(At, At.T) * Qt)

            n = df[t-1]
            ncp = (f_t * np.sqrt(2 / n) *
                   special.gamma(n/2) / special.gamma((n - 1) / 2))

            self.ncp[t-1] = ncp
            self.forc_var[t-1] = Qt
            self.R[t] = Rt

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

    def plot_mu(self, alpha=0.10, prior=True, index=None, ax=None):
        if ax is None:
            fig, axes = plt.subplots(nrows=self.ndim, sharex=True, squeeze=False)

        level, ci_lower, ci_upper = self.mu_ci(prior=prior, alpha=alpha)
        rng = np.arange(self.nobs)

        if index is None:
            indices = range(self.ndim)
        else:
            indices = [index]

        for i in indices:
            if ax is None:
                ax = axes[i][0]

            lev = level[:, i]
            ax.plot(rng, lev, 'k--')
            ax.plot(rng, ci_lower[:, i], 'k-.')
            ax.plot(rng, ci_upper[:, i], 'k-.')

            ptp = np.ptp(lev)
            ylim = (lev.min() - 1 * ptp, lev.max() + 1 * ptp)
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
        return np.sqrt(self.forc_var)

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
        pdfs = []
        for a, b, c in zip(self.y, self.df[:-1], self.ncp):
            pdfs.append(nct_pdf(a, b, c))

        return np.array(pdfs)

    @property
    def pred_loglike(self):
        return np.log(self.pred_like).sum()

    def mu_ci(self, alpha=0.10, prior=True):
        """
        If prior is False, compute posterior

        TODO: need multivariate T distribution
        """
        _x, _y = np.diag_indices(self.ndim)
        diags = self.mu_scale[:, _x, _y]

        adj = np.where(self.df > 2, np.sqrt(self.df / (self.df - 2)), 1)

        if prior:
            sd = np.sqrt(diags.T * adj / self.disc).T[:-1]
        else:
            sd = np.sqrt(diags.T * adj).T[1:]

        if prior:
            df = self.df[:-1]
            level = self.mu_mode[:-1]
            scale = np.sqrt(self.mu_scale[:-1] / self.disc)
        else:
            df = self.df[1:]
            level = self.mu_mode[1:]
            scale = np.sqrt(self.mu_scale[1:])

        ci_lower = level - 2 * sd
        ci_upper = level + 2 * sd

        return level, ci_lower, ci_upper # make_t_ci(df, level, scale, alpha=alpha)

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

class DLMResults(object):

    def __init__(self):
        pass

if __name__ == '__main__':
    pass
