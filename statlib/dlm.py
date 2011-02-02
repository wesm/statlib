"""
Implementing random things from West & Harrison
"""
from __future__ import division


from datetime import datetime
import re

from numpy import log
import numpy as np
import numpy.linalg as npl
import scipy.linalg as L
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.special as special
from scipy.special import gammaln as gamln

from pandas.util.testing import debug, set_trace as st
from statlib.tools import chain_dot
from statlib.plotting import plotf
import statlib.distributions as dist
reload(dist)

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
        G = self.G

        for i, obs in enumerate(self.y):
            # column vector, for W&H notational consistency
            Ft = self._get_Ft(i)

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

    def _get_Ft(self, t):
        return self.F[t:t+1].T

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
        # pdfs = []
        # for a, b, c in zip(self.y, self.df[:-1], self.ncp):
        #     pdfs.append(nct_pdf(a, b, c))

        # return np.array(pdfs)
        return stats.t.pdf(self.y, self.df[:-1], loc=self.forecast,
                          scale=self.forc_std)
        # return t_pdf(self.y, self.df[:-1], mode=self.forecast,
        #              scale=self.forc_std)

    @property
    def pred_loglike(self):
        return log(self.pred_like).sum()

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


class ConstantDLM(DLM):

    def _get_Ft(self, t):
        return self.F[0:1].T

def t_pdf(x, df, mode=0., scale=1.):
    """
    Three-parameter t distribution
    """
    n = df * 1.0

    part1 = gamln(0.5 * (n + 1)) - gamln(0.5 * n)
    part2 = - 0.5 * log(np.pi * scale * n)
    part3 = - 0.5 * (n + 1) * log(1 + (x - mode) ** 2 / (n * scale))
    return np.exp(part1 + part2 + part3)


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

class Component(object):
    """
    Constant DLM component, can be combined with other components via
    superposition
    """
    def __init__(self, F, G, W=None, discount=1.):
        if F.ndim == 1:
            F = np.atleast_2d(F)

        self.F = F
        self.G = G
        self.discount = discount

    def __add__(self, other):
        if not isinstance(other, Component):
            raise Exception('Can only add other DLM components!')

        return Superposition(self, other)

    def __radd__(self, other):
        return Superposition(other, self)

class Regression(object):

    def __init__(self, F, G, discount=1.):
        pass

class Polynomial(Component):
    """
    nth order Polynomial DLM using Jordan form system matrix

    Parameters
    ----------
    order : int
    lam : float, default 1.
    """
    def __init__(self, order, lam=1., discount=1.):
        self.order = order

        F = _e_vector(order)
        G = jordan_form(order, lam)
        Component.__init__(self, F, G, discount=discount)

    def __repr__(self):
        return 'Polynomial(%d)' % self.order

class Superposition(object):
    """

    """

    def __init__(self, *comps):
        self.comps = list(comps)

    @property
    def F(self):
        return np.concatenate([c.F for c in self.comps], axis=1)

    @property
    def G(self):
        return L.block_diag(*[c.G for c in self.comps])

    @property
    def discount(self):
        pass

    def __repr__(self):
        reprs = ', '.join(repr(c) for c in self.comps)
        return 'Superposition: [%s]' % reprs

    def __add__(self, other):
        if isinstance(other, Component):
            new_comps = self.comps + [other]
        elif isinstance(other, Superposition):
            new_comps = self.comps + other.comps

        return Superposition(*new_comps)

    def __radd__(self):
        pass

class FormFreeSeasonal(Component):

    def __init__(self, period, discount=1.):
        F = _e_vector(period)
        P = perm_matrix(period)
        self.period = period
        Component.__init__(self, F, P, discount=discount)

    def __repr__(self):
        return 'SeasonalFree(period=%d)' % self.period

class FourierForm(Component):
    """

    """
    def __init__(self, theta=None, discount=1.):
        self.theta = theta
        F = _e_vector(2)
        G = fourier_matrix(theta)
        Component.__init__(self, F, G, discount=discount)

    def __repr__(self):
        return 'FourierForm(%.4f)' % self.theta

class FullEffectsFourier(Component):
    """
    Full effects Fourier form DLM rep'n

    Parameters
    ----------
    period : int
    harmonics : sequence, default None
        Optionally specify a subset of harmonics to use
    discount : float

    Notes
    -----
    W&H pp. 252-254
    """

    def __init__(self, period, harmonics=None, discount=1.):
        period = int(period)
        theta = 2 * np.pi / period
        h = period // 2

        self.period = period

        comps = []

        model = None
        for j in np.arange(1, h):
            comp = FourierForm(theta=theta * j)
            if model is None:
                model = comp
            else:
                model += comp

            comps.append(comp)

        if period % 2 == 1:
            last_comp = FourierForm(theta=h * theta)
        else:
            last_comp = Polynomial(1, lam=-1.)

        model += last_comp
        comps.append(last_comp)

        self.comps = comps
        self.model = model

        Component.__init__(self, self.model.F, self.model.G,
                           discount=discount)

    @property
    def L(self):
        # W&H p. 254
        return np.vstack([np.dot(self.F, npl.matrix_power(self.G, i))
                          for i in range(self.period)])

def _e_vector(n):
    result = np.zeros(n)
    result[0] = 1
    return result

def perm_matrix(k):
    """
    Permutation matrix (is there a function for this?)
    """
    result = jordan_form(k, 0)
    result[k-1, 0] = 1
    return result

def test_perm_matrix():
    k = 6
    P = perm_matrix(k)
    assert(np.array_equal(np.linalg.matrix_power(P, k), np.eye(k)))
    assert(np.array_equal(np.linalg.matrix_power(P, k + 1), P))

def jordan_form(dim, lam=1):
    """
    Compute Jordan form matrix J_n(lambda)
    """
    inds = np.arange(dim)
    result = np.zeros((dim, dim))
    result[inds, inds] = lam

    # set superdiagonal to ones
    result[inds[:-1], 1 + inds[:-1]] = 1
    return result

def fourier_matrix(theta):
    import math

    arr = np.empty((2, 2))
    cos = math.cos(theta)
    sin = math.sin(theta)

    arr[0,0] = arr[1,1] = cos
    arr[0,1] = sin
    arr[1,0] = -sin

    return _zero_out(arr)

def simulate_dlm():
    pass

def fourier_coefs(seq):
    seq = np.asarray(seq)

    p = len(seq)
    theta = 2 * np.pi / p

    angles = theta * np.outer(np.arange(p // 2 + 1), np.arange(p))
    a = (np.cos(angles) * seq).sum(axis=1) * 2 / p
    b = (np.sin(angles) * seq).sum(axis=1) * 2 / p

    a[0] /= 2
    a[-1] /= 2
    return _zero_out(a), _zero_out(b)

def plot_fourier_rep(seq, harmonic=None):
    a, b = fourier_coefs(seq)
    theta = 2 * np.pi / len(seq)

    p = len(seq)

    if harmonic is None:
        # plot all harmonics
        def f(t):
            angles = theta * np.arange(p // 2 + 1) * t
            return (a * np.cos(angles)).sum() + (b * np.sin(angles)).sum()
    else:
        def f(t):
            j = harmonic
            angle = theta * j * t
            return a[j] * np.cos(angle) + b[j] * np.sin(angle)

    plotf(np.vectorize(f, [np.float64]), 0, len(seq) + 1)
    plt.vlines(np.arange(p), 0, seq)
    plt.axhline(0)

def _zero_out(arr, tol=1e-15):
    return np.where(np.abs(arr) < tol, 0, arr)

class DLMResults(object):

    def __init__(self):
        pass

if __name__ == '__main__':
    seq = [1.65, 0.83, 0.41, -0.70, -.47, .40, -.05, -1.51,
           -0.19, -1.02, -0.87, 1.52]

    fef = FullEffectsFourier(4)
