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
import statlib.distributions as distm

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
    mu_forc_scale = None # Rt
    var_est = None # St
    forc_var = None # Qt

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

        f_shape = self.nobs, self.ndim
        # constant DLM handling
        if F.ndim == 1:
            self.F = np.ones(f_shape) * F
        else:
            if len(F) == 1 and self.nobs > 1:
                F = np.ones(f_shape) * F[0]
            self.F = F

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
         self.mu_forc_scale) = _filter_python(self.y, self.F, self.G, self.disc,
                                              self.df0, self.v0, self.m0,
                                              self.C0)

    def backward_sample(self, steps=1):
        """
        Generate state sequence using distributions:

        .. math:: p(\theta_{t-k} | D_t)
        """
        from statlib.distributions import rmvnorm

        if steps != 1:
            raise Exception('only do one step backward sampling for now...')

        T = self.nobs
        # Backward sample
        mu_draws = np.zeros((T + 1, self.ndim))

        m = self.mu_mode
        C = self.mu_scale
        a = self.mu_forc_mode
        R = self.mu_forc_scale

        mu_draws[T] = rmvnorm(m[T], C[T])

        # initial values for smoothed dist'n
        for t in xrange(T-1, -1, -1):
            # B_{t} = C_t G_t+1' R_t+1^-1
            B = chain_dot(C[t], self.G, R[t+1])

            # smoothed mean
            ht = m[t] + np.dot(B, mu_draws[t+1] - a[t+1])
            Ht = C[t] - chain_dot(B, R[t+1], B.T)

            mu_draws[t] = rmvnorm(ht, np.atleast_2d(Ht))

        return mu_draws.squeeze()

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
    Matrix-variate DLM with matrix normal inverse Wishart prior and discount
    factor on covariance

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

    def __init__(self, y, F, G=None, V=1, mean_prior=None, var_prior=None,
                 state_discount=0.9, cov_discount=0.95):
        self.y = m_(y)
        self.dates = y.index
        self.names = y.columns

        self.nobs, self.ndim = self.y.shape

        assert(self.y.ndim == 2)

        F = m_(F)
        if F.ndim == 1:
            self.nparam = len(F)
            shape = (self.nobs, self.nparam)
            F = np.ones(shape) * F
        else:
            self.nparam = F.shape[1]
            shape = (self.nobs, self.nparam)
            if len(F) == 1 and self.nobs > 1:
                F = np.ones(shape) * F[0]

        self.F = F
        self.G = G
        self.V = V

        self.mean_prior = mean_prior
        self.var_prior = var_prior

        self.beta = cov_discount
        self.delta = state_discount

        self.n0, self.D0 = var_prior
        self.m0, self.C0 = mean_prior

        self.df = self.n0 + np.arange(self.nobs + 1) # precompute
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
        (self.mu_mode,
         self.mu_forc_mode,
         self.mu_scale,
         self.var_est,
         self.forc_var) = _mvfilter_python(self.y, self.F, self.G, self.V,
                                           self.delta, self.beta, self.n0,
                                           self.D0, self.m0, self.C0)


from pandas.util.testing import set_trace as st

def _mvfilter_python(Y, F, G, V, delta, beta, df0, v0, m0, C0):
    """
    Matrix-variate DLM update equations

    V : V_t sequence
    """
    nobs, k = Y.shape
    p = F.shape[1]
    mode = nan_array(nobs + 1, p, k) # theta posterior mode
    C = nan_array(nobs + 1, p, p) # theta posterior scale
    a = nan_array(nobs + 1, p, k) # theta forecast mode
    Q = nan_array(nobs) # variance multiplier term
    D = nan_array(nobs + 1, k, k) # scale matrix
    S = nan_array(nobs + 1, k, k) # covariance estimate D / n
    df = nan_array(nobs + 1)

    mode[0] = m0
    C[0] = C0
    df[0] = df0
    S[0] = v0

    Mt = m0
    Ct = C0
    n = df0
    d = n + k - 1
    Dt = D0
    St = Dt / n

    # allocate result arrays
    for t in xrange(1, nobs + 1):
        obs = Y[t - 1:t].T
        Ft = F[t - 1:t].T

        # derive innovation variance from discount factor
        # only discount after first time step?
        if G is not None:
            at = np.dot(G, Mt)
            Rt = chain_dot(G, Ct, G.T) / delta
        else:
            at = Mt
            Rt = Ct / delta

        et = obs - np.dot(at.T, Ft)
        qt = chain_dot(Ft.T, Rt, Ft) + V
        At = np.dot(Rt, Ft) / qt

        # update mean parameters
        n = beta * n
        b = (n + k - 1) / d
        n = n + 1
        d = n + k - 1

        Dt = b * Dt + np.dot(et, et.T) / qt
        St = Dt / n
        Mt = at + np.dot(At, et.T)
        Ct = Rt - np.dot(At, At.T) * qt

        C[t] = Ct; df[t] = n; S[t] = (St + St.T) / 2; mode[t] = Mt
        D[t] = Dt; a[t] = at; Q[t-1] = qt

    return mode, a, C, S, Q

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
        if t > 1:
            # only discount after first time step!
            if G is not None:
                at = np.dot(G, mode[t - 1])
                Rt = chain_dot(G, C[t - 1], G.T) / delta
            else:
                Rt = C[t - 1] / delta

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

def rmatnorm(M, U, V):
    """
    Generate matrix-normal random variates

    Parameters
    ----------
    M : ndarray (r x q)
        mean matrix
    U : ndarray (r x r)
        row covariance matrix
    V : ndarray (q x q)
        column covariance matrix

    Notes
    -----
    Y ~ MN(M, U, V) if vec(Y) ~ N(vec(M), kron(V, U))
    """
    mean = M.ravel('F')
    cov = np.kron(V, U)
    draw = np.random.multivariate_normal(mean, cov)
    return draw.reshape(M.shape, order='F')

def _filter_cython(Y, F, G, delta, df0, v0, m0, C0):
    import statlib.ffbs as ffbs

    (mode, a, Q, C, S) = ffbs.udlm_filter_disc(Y, np.asarray(F, order='C'),
                                               G, delta,
                                               df0, v0,
                                               np.asarray(m0, dtype=float),
                                               np.asarray(C0, dtype=float))

    return mode, a, Q, C, S

if __name__ == '__main__':
    import statlib.components as comp
    reload(comp)
    from statlib.components import VectorAR
    import statlib.datasets as ds

    returns = ds.fx_rates_returns()
    spot = ds.fx_rates_spot()
    k = len(returns.columns)

    spot = np.log(spot)

    # F = np.array([1])
    lags = 2
    VAR_comp = VectorAR(spot, lags=lags)
    F = VAR_comp.F

    Y = spot[lags:]
    Yv = Y.values

    n0 = 2
    d0 = n0 + k - 1
    D0 = n0 * np.eye(k) * np.diff(Yv, axis=0).var(0, ddof=1).mean() / 100
    delta = 0.99995
    beta = 0.95

    p = F.shape[1]
    m0 = np.zeros((p, k))
    m0[1:k+1] = np.eye(k)

    mean_prior = m0, 1000 * np.eye(p)
    var_prior = 2, D0

    model = MVDLM(Y, F,
                  state_discount=delta, cov_discount=beta,
                  mean_prior=mean_prior, var_prior=var_prior)


