"""
Implementing random things from West & Harrison

Notes on notation:

Y_t = F_t' Th_t + nu_t
Th_t = G_t th_{t-1} + \omega_t

\nu_t \sim \cal{N}(0, V_t)
\omega_t \sim \cal{N}(0, W_t)
"""
from __future__ import division

from numpy import log
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from statlib.tools import chain_dot, nan_array
from statlib.plotting import plotf
import statlib.tools as tools
import statlib.distributions as distm

def nct_pdf(x, df, nc):
    from rpy2.robjects import r
    dt = r.dt
    return dt(x, df, nc)

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
        self.dates = y.index
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

        self.mu_mode = nan_array(self.nobs + 1, self.ndim)
        self.mu_forc_mode = nan_array(self.nobs + 1, self.ndim)
        self.mu_scale = nan_array(self.nobs + 1, self.ndim, self.ndim)
        self.var_est = nan_array(self.nobs + 1)
        self.forc_var = nan_array(self.nobs)
        self.R = nan_array(self.nobs + 1, self.ndim, self.ndim)

        self.mu_mode[0], self.mu_scale[0] = mean_prior
        n, d = var_prior
        self.df = n + np.arange(self.nobs + 1) # precompute
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
        G = self.G

        for i, obs in enumerate(self.y):
            # column vector, for W&H notational consistency
            Ft = self._get_Ft(i)

            # advance index: y_1 through y_nobs, 0 is prior
            t = i + 1

            # derive innovation variance from discount factor
            if t > 1:
                # only discount after first time step! hmm
                at = np.dot(G, mode[t - 1])
                Rt = chain_dot(G, C[t - 1], G.T) / self.disc
            else:
                at = mode[0]
                Rt = C[0]

            Qt = chain_dot(Ft.T, Rt, Ft) + S[t-1]
            At = np.dot(Rt, Ft) / Qt

            # forecast theta as time t
            ft = np.dot(Ft.T, at)
            err = obs - ft

            # update mean parameters
            mode[t] = at + np.dot(At, err)
            S[t] = S[t-1] + (S[t-1] / df[t]) * ((err ** 2) / Qt - 1)
            C[t] = (S[t] / S[t-1]) * (Rt - np.dot(At, At.T) * Qt)

            self.mu_forc_mode[t] = at
            self.forc_var[t-1] = Qt
            self.R[t] = Rt

    def _get_Ft(self, t):
        return self.F[t:t+1].T

    def plot_forc(self, alpha=0.10, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.subplot(111)

        # rng = np.arange(self.nobs)
        rng = self.dates # np.arange(self.nobs)

        ci_lower, ci_upper = self.forc_ci(alpha=alpha)

        ax.plot(rng, self.y, 'k.')
        ax.plot(rng, self.forecast, 'k--')

        ax.plot(rng, ci_lower, 'k-.')
        ax.plot(rng, ci_upper, 'k-.')

        ptp = np.ptp(self.y)
        ylim = (self.y.min() - 1 * ptp, self.y.max() + 1 * ptp)
        ax.set_ylim(ylim)

    def plot_forc_err(self, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.subplot(111)

        err = self.y - self.forecast
        ax.plot(err)
        ax.axhline(0)

    def plot_mu(self, alpha=0.10, prior=True, index=None, ax=None):
        if ax is None:
            _, axes = plt.subplots(nrows=self.ndim, sharex=True, squeeze=False,
                                   figsize=(12, 8))

        level, ci_lower, ci_upper = self.mu_ci(prior=prior, alpha=alpha)
        rng = np.arange(self.nobs)

        if index is None:
            indices = range(self.ndim)
        else:
            indices = [index]

        for i in indices:
            if ax is None:
                this_ax = axes[i][0]
            else:
                this_ax = ax

            lev = level[:, i]
            this_ax.plot(rng, lev, 'k--')
            this_ax.plot(rng, ci_lower[:, i], 'k-.')
            this_ax.plot(rng, ci_upper[:, i], 'k-.')

            ptp = lev.ptp()
            ylim = (lev.min() - 0.5 * ptp, lev.max() + 0.5 * ptp)
            this_ax.set_ylim(ylim)

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
        return (self.F * self.mu_forc_mode[1:]).sum(1)

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
        return stats.t.pdf(self.y, self.df[:-1],
                           loc=self.forecast,
                           scale=np.sqrt(self.forc_var))

    @property
    def pred_loglike(self):
        return log(self.pred_like).sum()

    def mu_ci(self, alpha=0.10, prior=False):
        """
        Compute marginal confidence intervals around each parameter \theta_{ti}
        If prior is False, compute posterior
        """
        _x, _y = np.diag_indices(self.ndim)
        diags = self.mu_scale[:, _x, _y]

        # Only care about marginal scale
        delta = self.disc
        if isinstance(delta, np.ndarray):
            delta = np.diag(delta)

        if prior:
            df = self.df[:-1]
            mode = self.mu_mode[:-1]
            scale = np.sqrt(diags[:-1] / self.disc)
        else:
            df = self.df[1:]
            mode = self.mu_mode[1:]
            scale = np.sqrt(diags[1:])

        q = stats.t(df).ppf(1 - alpha / 2)
        band = (scale.T * q).T
        ci_lower = mode - band
        ci_upper = mode + band

        return mode, ci_lower, ci_upper

    def forc_ci(self, alpha=0.10):
        return distm.make_t_ci(self.df[:-1], self.forecast,
                               np.sqrt(self.forc_var), alpha=alpha)

    def fit(self, discount=0.9):
        pass

class ConstantDLM(DLM):
    """

    """

    def _get_Ft(self, t):
        return self.F[0:1].T

