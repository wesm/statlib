"""
Implementing random things from West & Harrison

Notes on notation:

Y_t = F_t' Th_t + nu_t
Th_t = G_t th_{t-1} + \omega_t

\nu_t \sim \cal{N}(0, V_t)
\omega_t \sim \cal{N}(0, W_t)

TODO
----
handling of missing data
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
from pandas.util.testing import set_trace as st

m_ = np.array

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
    F = None
    mu_mode = None # mt
    mu_scale = None # Ct
    mu_forc_mode = None # at
    var_est = None # St
    forc_var = None # Qt
    R = None # Rt

    def __init__(self, y, F, G=None, mean_prior=None, var_prior=None,
                 discount=0.9):
        self.dates = y.index
        self.y = m_(y)
        self.nobs = len(y)

        if self.y.ndim == 1:
            pass
        else:
            raise Exception

        F = m_(F)
        if F.ndim == 1:
            self.ndim = len(F)
        else:
            self.ndim = F.shape[1]

        # constant DLM handling
        self._set_F(F)

        if G is None:
            G = np.eye(self.ndim)

        self.G = G

        self.mean_prior = mean_prior
        self.var_prior = var_prior
        self.disc = discount

        df0, df0v0 = var_prior
        self.df = df0 + np.arange(self.nobs + 1) # precompute

        self.m0, self.C0 = mean_prior
        self.df0 = df0
        self.v0 = df0v0 / df0

        self._compute_parameters()

    def _set_F(self, F):
        if F.ndim == 1:
            self.F = np.ones((self.nobs, self.ndim)) * F
        else:
            if len(F) == 1 and self.nobs > 1:
                F = np.ones((self.nobs, self.ndim)) * F[0]
            self.F = F

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
        (self.mu_mode,
         self.mu_forc_mode,
         self.mu_scale,
         self.var_est,
         self.forc_var,
         self.R) = _filter_python(self.y, self.F, self.G, self.disc,
                                  self.df0, self.v0, self.m0, self.C0)

    def backward_sample(self, steps=1):
        """
        Generate state sequence using distributions:

        .. math:: p(\theta_{t-k} | D_t)
        """
        if steps != 1:
            raise Exception('only do one step backward sampling for now...')

        # Backward sample
        mu = np.zeros((self.nobs + 1, self.ndim))

        # initial values for smoothed dist'n
        for t in xrange(T, -1, -1):
            if t == T:
                # sample from posterior
                fm = mode[-1]
                fR = C[-1]
            else:
                # B_{t} = C_t G_t+1' R_t+1^-1
                B = np.dot(C[t] * phi, la.inv(R[t+1:t+2]))

                # smoothed mean
                fm = mode[t] + np.dot(B, mode[t+1] - a[t+1])
                fR = C[t] + chain_dot(B, C[t+1] - R[t+1], B.T)

            mu[t] = dist.rmvnorm(fm, np.atleast_2d(fR))

        return mu.squeeze()

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

class DLM2(DLM):

    def _compute_parameters(self):
        df0, df0v0 = self.var_prior
        v0 = float(df0v0) / df0

        m0, C0 = self.mean_prior

        (mode, a, Q, C, S) = _filter_cython(self.y, self.F, self.G,
                                            self.disc, df0, v0, m0, C0)

        self.forc_var = Q
        self.mu_forc_mode = a
        self.mu_mode = mode
        self.mu_scale = C
        self.var_est = S

class MVDLM(object):
    """
    Matrix-variate DLM with matrix normal inverse Wishart prior

    Parameters
    ----------
    y : ndarray (n x k)
    F : ndarray (n x k x r)
    G : ndarray (k x k)
        system matrix, defaults to identity matrix

    Notes
    -----

    .. math::

        Y = Ft' \theta_t + \nu_t
        \theta_t = G \theta_{t-1} + \omega_t

    (Theta_0, Sigma) ~ NIW(M, c, n, D)
    Theta_0 | Sigma ~ N(M, c * Sigma)
    Sigma ~ IW(n, D)
    """

    def __init__(self, y, F, G=None, mean_prior=None, var_prior=None,
                 discount=0.9):
        self.dates = y.index
        self.y = m_(y)

        self.nobs, self.ndim = self.y.shape
        self.nparam = F.shape[1]

        if self.y.ndim == 1:
            pass
        else:
            raise Exception

        F = m_(F)
        self._set_F(F)

        if G is None:
            G = np.eye(self.ndim)

        self.G = G

        self.mean_prior = mean_prior
        self.var_prior = var_prior
        self.disc = discount

        df0, df0v0 = var_prior
        self.df = df0 + np.arange(self.nobs + 1) # precompute

        self.m0, self.C0 = mean_prior
        self.df0 = df0
        self.v0 = df0v0 / df0

        # self._compute_parameters()

    def _set_F(self, F):
        self.F = F

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
        (self.mu_mode, self.mu_forc_mode,
         self.mu_scale, self.var_est,
         self.forc_var, self.R) = _filter_python(self.y, self.F, self.G,
                                                 self.disc, self.df0, self.v0,
                                                 self.m0, self.C0)


def _mvfilter_python(Y, F, G, V, delta, df0, v0, m0, C0):
    """
    Matrix-variate DLM update equations

    V : V_t sequence
    """

    nobs, ndim = Y.shape
    nparam = len(G)

    mode = nan_array(nobs + 1, nparam)
    a = nan_array(nobs + 1, nparam)
    C = nan_array(nobs + 1, nparam, nparam)
    R = nan_array(nobs + 1, nparam, nparam)

    # variance multiplier term
    Q = nan_array(nobs)

    # scale matrix
    D = nan_array(nobs + 1, ndim, ndim)

    # covariance estimate D / n
    S = nan_array(nobs + 1, ndim, ndim)

    mode[0] = m0
    C[0] = C0

    df = df0 + np.arange(nobs + 1) # precompute
    S[0] = v0

    # allocate result arrays
    for t in xrange(1, nobs + 1):
        obs = Y[t - 1]
        Ft = F[t - 1]

        # derive innovation variance from discount factor

        at = mode[t - 1]
        Rt = C[t - 1]
        if t > 1 and G is not None:
            # only discount after first time step!
            at = np.dot(G, mode[t - 1])
            Rt = chain_dot(G, C[t - 1], G.T) / delta

        Qt = chain_dot(Ft.T, Rt, Ft) + V[t-1]

        At = np.dot(Rt, Ft) / Qt

        # forecast theta as time t
        ft = np.dot(at, Ft)
        err = obs - ft

        # update mean parameters
        mode[t] = at + np.dot(At, err.T)

        D[t] = D[t-1] + np.outer(err, err) / Qt

        S[t] = D[t] / df[t]
        C[t] = Rt - np.dot(At, At.T) * Qt
        a[t] = at
        Q[t-1] = Qt
        R[t] = Rt

    return mode, a, C, S, Q, R

def _filter_python(Y, F, G, delta, df0, v0, m0, C0):
    """
    Univariate DLM update equations
    """

    nobs = len(Y)
    ndim = len(G)

    mode = nan_array(nobs + 1, ndim)
    a = nan_array(nobs + 1, ndim)
    C = nan_array(nobs + 1, ndim, ndim)
    S = nan_array(nobs + 1)
    Q = nan_array(nobs)
    R = nan_array(nobs + 1, ndim, ndim)

    mode[0] = m0
    C[0] = C0

    df = df0 + np.arange(nobs + 1) # precompute
    S[0] = v0

    # allocate result arrays
    for i, obs in enumerate(Y):
        # column vector, for W&H notational consistency
        Ft = F[i]

        # advance index: y_1 through y_nobs, 0 is prior
        t = i + 1

        # derive innovation variance from discount factor
        at = mode[t - 1]
        Rt = C[t - 1]
        if t > 1 and G is not None:
            # only discount after first time step!
            at = np.dot(G, mode[t - 1])
            Rt = chain_dot(G, C[t - 1], G.T) / delta

        Qt = chain_dot(Ft, Rt, Ft) + S[t-1]
        At = np.dot(Rt, Ft) / Qt

        # forecast theta as time t
        ft = np.dot(Ft, at)
        err = obs - ft

        # update mean parameters
        mode[t] = at + np.dot(At, err)
        S[t] = S[t-1] + (S[t-1] / df[t]) * ((err ** 2) / Qt - 1)
        C[t] = (S[t] / S[t-1]) * (Rt - np.dot(At, At.T) * Qt)

        a[t] = at
        Q[t-1] = Qt
        R[t] = Rt

    return mode, a, C, S, Q, R

def _filter_cython(Y, F, G, delta, df0, v0, m0, C0):
    import statlib.ffbs as ffbs

    (mode, a, Q, C, S) = ffbs.udlm_filter_disc(Y, np.asarray(F, order='C'),
                                               G, delta,
                                               df0, v0,
                                               np.asarray(m0, dtype=float),
                                               np.asarray(C0, dtype=float))

    return mode, a, Q, C, S

if __name__ == '__main__':
    import statlib.datasets as ds

    returns = ds.fx_rates_returns()
    F = np.ones((1, len(returns.columns)))

    model = MVDLM(returns, F)
