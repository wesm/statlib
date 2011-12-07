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

# pylint: disable=W0201

from __future__ import division

from numpy import log
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import scipy.stats as stats

from statlib.tools import chain_dot, nan_array
import statlib.distributions as distm
import statlib.filter as filt

from pandas.util.testing import set_trace as st

m_ = np.array

def nct_pdf(x, df, nc):
    from rpy2.robjects import r
    dt = r.dt
    return dt(x, df, nc)

class DLM(object):
    r"""
    Bayesian Gaussian Dynamic Linear Model (DLM) with discount factors

    Parameters
    ----------
    y : ndarray n x 1
        Response variable
    F : ndarray (n x k or 1d for constant Ft = F)
        Regressor matrix
    G : ndarray k x k
        State transition matrix
    m0 : ndarray k
        Prior mean for state vector
    C0 : ndarray k x k
        Prior covariance for state vector
    n0 : int
        Prior degrees of freedom for observation variance
    s0 : float
        Prior point estimate of observation variance
    state_discount : float
        Discount factor for determining state evolution variance
        Wt = Ct * (1 - d) / d
    var_discount : float
        Discount factor for observational variance

    Notes
    -----
    Normal dynamic linear model (DLM) in state space form

    .. math::

        y = F_t' \theta_t + \nu_t \\
        \theta_t = G_t \theta_{t-1} + \omega_t \\
        \nu_t \sim N(0, V_t) \\
        \omega_t \sim N(0, W_t) \\
        W_t = C_t (1 - \delta) / \delta \\
        IG(df0 / 2, df0 * v0 / 2)

    Priors

    .. math::

        {\sf Normal}(m_0, C_0)\\
        {\sf InverseGamma}(n_0 / 2, n_0 s_0 / 2)
    """
    F = None
    mu_mode = None # mt
    mu_scale = None # Ct
    mu_forc_mode = None # at
    mu_forc_scale = None # Rt
    var_est = None # St
    forc_var = None # Qt

    def __init__(self, y, F, G=None, m0=None, C0=None, n0=None, s0=None,
                 mean_prior=None, var_prior=None, discount=None,
                 state_discount=0.9, var_discount=1.):
        try:
            self.dates = y.index
        except AttributeError:
            self.dates = None

        y = m_(y)
        if y.ndim == 1:
            pass
        else:
            raise Exception

        self.y = y
        self.nobs = len(y)

        constant = False

        F = m_(F)
        if F.ndim == 1:
            if len(F) == len(y):
                self.ndim = 1
            else:
                constant = True
                self.ndim = len(F)
        else:
            self.ndim = F.shape[1]

        f_shape = self.nobs, self.ndim
        # HACK ?
        # constant DLM handling
        if F.ndim == 1:
            if constant:
                F = np.ones(f_shape) * F
            else:
                F = F.reshape(f_shape)
        else:
            if len(F) == 1 and self.nobs > 1:
                F = np.ones(f_shape) * F[0]

        self.F = F

        if G is None:
            G = np.eye(self.ndim)

        self.G = G

        # HACK for transition
        if discount is not None:
            state_discount = discount

        self.state_discount = state_discount
        self.var_discount = var_discount

        if mean_prior is not None:
            m0, C0 = mean_prior

        if var_prior is not None:
            n0, s0 = var_prior

        if np.isscalar(m0):
            m0 = np.array([m0])
        if np.isscalar(C0):
            C0 = np.array([[C0]])

        self.m0, self.C0 = m0, C0
        self.df0, self.s0 = n0, s0

        self._forward_filter()

    def _forward_filter(self):
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
         self.df,
         self.var_est,
         self.forc_var,
         self.mu_forc_scale) = filter_python(self.y, self.F,
                                                  self.G,
                                                  self.state_discount,
                                                  self.var_discount,
                                                  self.df0, self.s0,
                                                  self.m0, self.C0)

    def backward_smooth(self):
        """
        Compute posterior estimates of state vector given full data set,
        i.e. p(\theta_t | D_T)

        cf. W&H sections
            4.7 / 4.8: regular smoothing recurrences (Theorem 4.4)
            10.8.4 adjustments for variance discounting

        Notes
        -----
        \theta_{t-k} | D_t ~ T_{n_t} [a_t(-k), (S_t / S_{t-k}) R_t(-k)]

        Returns
        -------
        (filtered state mode,
         filtered state cov,
         filtered degrees of freedom)
        """
        beta = self.var_discount

        T = self.nobs
        a = self.mu_forc_mode
        R = self.mu_forc_scale

        fdf = self.df.copy()
        fS = self.var_est.copy()
        fm = self.mu_mode.copy()
        fC = self.mu_scale.copy()

        C = fC[T]
        for t in xrange(T - 1, -1, -1):
            B = chain_dot(fC[t], self.G.T, LA.inv(R[t+1]))

            # W&H p. 364
            fdf[t] = (1 - beta) * fdf[t] + beta * fdf[t + 1]
            fS[t] = 1 / ((1 - beta) / fS[t] + beta / fS[t+1])

            # W&H p. 113
            fm[t] = fm[t] + np.dot(B, fm[t+1] - a[t+1])
            C = fC[t] + chain_dot(B, C - R[t+1], B.T)
            fC[t] = C * fS[T] / fS[t]

        return fm, fC, fdf

    def backward_sample(self, steps=1):
        """
        Generate state sequence using distributions:

        .. math:: p(\theta_{t} | \theta_{t + k} D_t)
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
            B = chain_dot(C[t], self.G.T, LA.inv(R[t+1]))

            # smoothed mean
            ht = m[t] + np.dot(B, mu_draws[t+1] - a[t+1])
            Ht = C[t] - chain_dot(B, R[t+1], B.T)

            mu_draws[t] = rmvnorm(ht, np.atleast_2d(Ht))

        return mu_draws.squeeze()

    def plot_forc(self, alpha=0.10, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.subplot(111)

        rng = self._get_x_range()

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
        rng = self._get_x_range()

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


    def _get_x_range(self):
        if self.dates is None:
            return np.arange(self.nobs)
        else:
            return self.dates

    @property
    def forc_dist(self):
        df = self.var_discount * self.df[:-1]
        return stats.t(self.y, df, loc=self.forecast,
                       scale=np.sqrt(self.forc_var))

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
        # WH Table 10.4 re: variance discounting
        df = self.var_discount * self.df[:-1]
        return stats.t.pdf(self.y, df,
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
        delta = self.state_discount
        if isinstance(delta, np.ndarray):
            delta = np.diag(delta)

        if prior:
            df = self.df[:-1]
            mode = self.mu_mode[:-1]
            scale = np.sqrt(diags[:-1] / delta)
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

class VarDiscountDLM(DLM):
    pass

class DLM2(DLM):

    def _forward_filter(self):
        (mode, a, Q, C, S) = filt.filter_cython(self.y, self.F, self.G,
                                                self.state_discount,
                                                self.df0, self.s0,
                                                self.m0, self.C0)

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
        self._forward_filter()

    def _forward_filter(self):
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

from line_profiler import LineProfiler
prof = LineProfiler()

class DLMFilter(object):
    # should go into Cython
    pass

# def filter_python(ndarray[double_t, ndim=1] Y,
#                   ndarray F, ndarray G,
#                   double_t delta, double_t beta,
#                   double_t df0, double_t v0,
#                   ndarray m0, ndarray C0):
@prof
def filter_python(Y, F, G, delta, beta, df0, v0, m0, C0):
    """
    Univariate DLM update equations with unknown observation variance

    delta : state discount
    beta : variance discount
    """
    # cdef:
    #     Py_ssize_t i, t, nobs, ndim
    #     ndarray[double_t, ndim=1] df, Q, S
    #     ndarray a, C, R, mode

    #     ndarray at, mt, Ft, At, Ct, Rt
    #     double_t obs, ft, e, nt, dt, St, Qt

    nobs = len(Y)
    ndim = len(G)

    mode = nan_array(nobs + 1, ndim)
    a = nan_array(nobs + 1, ndim)
    C = nan_array(nobs + 1, ndim, ndim)
    R = nan_array(nobs + 1, ndim, ndim)

    S = nan_array(nobs + 1)
    Q = nan_array(nobs)
    df = nan_array(nobs + 1)

    mode[0] = mt = m0
    C[0] = Ct = C0
    df[0] = nt = df0
    S[0] = St = v0

    dt = df0 * v0

    # allocate result arrays
    for i in range(nobs):
        obs = Y[i]

        # column vector, for W&H notational consistency
        # Ft = F[i]
        Ft = F[i:i+1].T

        # advance index: y_1 through y_nobs, 0 is prior
        t = i + 1

        # derive innovation variance from discount factor
        at = mt
        Rt = Ct
        if t > 1:
            # only discount after first time step?
            if G is not None:
                at = np.dot(G, mt)
                Rt = chain_dot(G, Ct, G.T) / delta
            else:
                Rt = Ct / delta

        # Qt = chain_dot(Ft.T, Rt, Ft) + St
        Qt = chain_dot(Ft.T, Rt, Ft) + St
        At = np.dot(Rt, Ft) / Qt

        # forecast theta as time t
        ft = np.dot(Ft.T, at)
        e = obs - ft

        # update mean parameters
        mode[t] = mt = at + np.dot(At, e)
        dt = beta * dt + St * e * e / Qt
        nt = beta * nt + 1
        St = dt / nt

        S[t] = St
        Ct = (S[t] / S[t-1]) * (Rt - np.dot(At, At.T) * Qt)
        Ct = (Ct + Ct.T) / 2 # symmetrize

        df[t] = nt
        Q[t-1] = Qt

        C[t] = Ct
        a[t] = at
        R[t] = Rt

    return mode, a, C, df, S, Q, R

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

    C0 = 1000 * np.eye(p)

    n0 = 2

    mean_prior = m0, C0
    var_prior = n0, D0

    model = MVDLM(Y, F,
                  state_discount=delta, cov_discount=beta,
                  mean_prior=mean_prior, var_prior=var_prior)


