"""
Bayesian statistics and then some
"""

from __future__ import division

import pdb

from pandas import Series, DataMatrix
from pandas.util.testing import set_trace

import numpy as np

from matplotlib import ticker, cm
import matplotlib as mpl
import scipy.stats as stats
import scipy.optimize as opt
import scipy.special as special
import scipy.integrate as integ
import matplotlib.pyplot as plt

from scipy.special import gamma, gammaln, binom
from pandas.util.testing import set_trace
from numpy.linalg import inv
import pymc

from scikits.statsmodels.tsa.stattools import acf

def quantile(data, quantiles, axis=None):
    """
    Dumb
    """
    from scipy.stats import scoreatpercentile as _q
    return np.array([_q(data, quantile * 100)
                     for quantile in quantiles])

#-------------------------------------------------------------------------------
# Distribution-related functions

def tnorm(lower, upper, mean, sd):
    """
    Sample from a truncated normal distribution

    Parameters
    ----------
    lower : float or array-like
    upper : float or array-like
    mean : float or array_like
    sd : float or array_like

    Note
    ----
    Arrays passed must all be of the same length. Computes samples
    using the \Phi, the normal CDF, and Phi^{-1} using a standard
    algorithm:

    draw u ~ uniform(|Phi((l - m) / sd), |Phi((u - m) / sd))
    return m + sd * \Phi^{-1}(u)

    Returns
    -------
    samples : ndarray or float
    """
    ulower = special.ndtr((lower - mean) / sd)
    uupper = special.ndtr((upper - mean) / sd)

    if isinstance(ulower, np.ndarray):
        n = len(ulower)
        u = (uupper - ulower) * np.random.rand(n) + ulower
    else:
        u = (uupper - ulower) * np.random.rand() + ulower

    return mean + sd * special.ndtri(u)

def rgamma(a, b, n=None):
    if n:
        return np.random.gamma(a, scale=1./b, size=n)
    else:
        return np.random.gamma(a, scale=1./b)

rnorm = np.random.normal
rtnorm = stats.truncnorm.rvs
dnorm = stats.norm.pdf

def nans(shape):
    return np.empty(shape, dtype=float) * np.NaN

def stats_gamma_dist(a, b):
    return Series(
        {'mean' : a / b,
         'variance' : a / b**2,
         '95% conf range' : get_ci(stats.gamma(a, scale=1./b))}
        )

def stats_beta_dist(a, b):
    var = a * b / ((a + b + 1) * (a + b)**2)
    return Series(
        {'mean' : a / (a + b),
         'variance' : var,
         'stdev' : np.sqrt(var),
         'mode' : (a - 1) / (a + b - 2),
         '95% conf range' : get_ci(stats.beta(a, b))}
        )

def hpd_beta(y, n, h=.1, a=1, b=1, plot=False, **plot_kwds):
    apost = y + a
    bpost = n - y + b
    if apost > 1 and bpost > 1:
        mode = (apost - 1)/(apost + bpost - 2)
    else:
        raise Exception("mode at 0 or 1: HPD not implemented yet")

    post = stats.beta(apost, bpost)

    dmode = post.pdf(mode)

    lt = opt.bisect(lambda x: post.pdf(x) / dmode - h, 0, mode)
    ut = opt.bisect(lambda x: post.pdf(x) / dmode - h, mode, 1)

    coverage = post.cdf(ut) - post.cdf(lt)
    if plot:
        plt.figure()
        plotf(post.pdf)
        plt.axhline(h*dmode)
        plt.plot([ut, ut], [0, post.pdf(ut)])
        plt.plot([lt, lt], [0, post.pdf(lt)])
        plt.title(r'$p(%s < \theta < %s | y)$' %
                  tuple(np.around([lt, ut], 2)))

    return lt, ut, coverage, h

def density_plot(y, thresh=1e-10, style='k'):
    kde = stats.kde.gaussian_kde(y)
    plot_f_support(kde.evaluate, y.max(), y.min(),
                    thresh=thresh, style=style)

def plot_f_support(f, hi, lo, thresh=1e-10, style='k', N=5000):
    intervals = np.linspace(lo - 10 * np.abs(lo),
                            hi + 10 * np.abs(hi), num=N)
    pdfs = f(intervals)

    # try to find support
    above = (pdfs > thresh)
    supp_l, supp_u = above.argmax(), (len(above) - above[::-1].argmax() - 1)
    xs = np.linspace(intervals[supp_l], intervals[supp_u], num=N)
    plt.plot(xs, f(xs), style)

#-------------------------------------------------------------------------------
# MCMC utils

def effective_size(theta_mcmc):
    from rpy2.robjects.packages import importr
    from rpy2.robjects import FloatVector

    coda = importr('coda')
    es = coda.effectiveSize(FloatVector(theta_mcmc))
    return es[0]

#-------------------------------------------------------------------------------
# Graphing functions

def joint_contour(z_func, xlim=(0, 1), ylim=(0, 1), n=50,
                  ncontours=20):
    x = np.linspace(*xlim, num=n)
    y = np.linspace(*ylim, num=n)

    X, Y = np.meshgrid(x, y)

    plt.figure()
    cs = plt.contourf(X, Y, z_func(X, Y), ncontours, cmap=cm.gray_r)
    plt.colorbar()
    plt.xlim(xlim)
    plt.ylim(ylim)

def _joint_normal(mu, s2, k, nu, X, Y):
    s2_vals = stats.gamma.pdf(Y, nu / 2, scale=2. / (s2 * nu))
    theta_vals = stats.norm.pdf(X, mu, 1 / np.sqrt(Y * k))

    return s2_vals * theta_vals

def graph_data(f, xstart=0, xend=1, n=1000):
    inc = (xend - xstart) / n
    xs = np.arange(xstart, xend + inc, inc)

    ys = f(xs)

    return xs, ys

def plotf(f, xstart=0, xend=1, n=1000, style='b', axes=None, label=None):
    """
    Continuous
    """
    xs, ys = graph_data(f, xstart=xstart, xend=xend, n=n)
    if axes is not None:
        plt.plot(xs, ys, style, label=label, axes=axes)
    else:
        plt.plot(xs, ys, style, label=label)

    # plt.vlines(xs, [0], ys, lw=2)

    plt.xlim([xstart - 1, xend + 1])

def plot_pdf(dist, **kwds):
    plotf(dist.pdf, **kwds)

def plot_discrete_pdf(f, xstart=0, xend=100, style='b', axes=None,
                      label=None):
    n = xend - xstart
    xs, ys = graph_data(f, xstart=xstart, xend=xend, n=n)

    print xs, ys

    plt.vlines(xs, [0], ys, lw=2)
    # plt.plot(xs, ys, style, label=label)
    plt.xlim([xstart - 1, xend])

def posterior_ci_plot(dist, ci=[0.025, 0.975]):
    pass

def make_plot(x, y, style='k', title=None, xlabel=None, ylabel=None, path=None):
    plt.figure(figsize=(10,5))
    plt.plot(x, y, style)
    adorn_plot(title=title, ylabel=ylabel, xlabel=xlabel)
    plt.savefig(path, bbox_inches='tight')

def mysave(path):
    plt.savefig(path, bbox_inches='tight', dpi=150)


def adorn_plot(title=None, ylabel=None, xlabel=None):
    plt.title(title, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)

def plot_acf(y, lags=100):
    acf = acf(y, nlags=lags)

    plt.figure(figsize=(10, 5))
    plt.vlines(np.arange(lags+1), [0], acf)

    plt.axhline(0, color='k')

def plot_acf_multiple(ys, lags=20):
    """

    """
    # hack
    old_size = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = 8

    plt.figure(figsize=(10, 10))
    xs = np.arange(lags + 1)

    acorr = np.apply_along_axis(lambda x: acf(x, nlags=lags), 0, ys)

    k = acorr.shape[1]
    for i in range(k):
        ax = plt.subplot(k, 1, i + 1)
        ax.vlines(xs, [0], acorr[:, i])

        ax.axhline(0, color='k')
        ax.set_ylim([-1, 1])

        # hack?
        ax.set_xlim([-1, xs[-1] + 1])

    mpl.rcParams['font.size'] = old_size

def ex_p6():
    plt.figure()

    w = 5.

    colors = 'bgrcm'

    for i in range(1, 5):
        theta_0 = 0.2 * i
        dist = stats.beta(w * theta_0, w * (1 - theta_0))

        plotf(dist.pdf, style=colors[i], label=str(theta_0))

    plt.legend(loc='best')

def all_perms(iterable):
    if len(iterable) <=1:
        yield iterable
    else:
        for perm in all_perms(iterable[1:]):
            for i in range(len(perm)+1):
                yield perm[:i] + iterable[0:1] + perm[i:]
