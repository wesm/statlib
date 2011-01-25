"""
Bayesian statistics and then some
"""

from __future__ import division

import pdb

from pandas import Series
from pandas.util.testing import set_trace

import numpy as np

import scipy.stats as stats
import scipy.optimize as opt
import scipy.special as special
import matplotlib.pyplot as plt

from scipy.special import gammaln

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
    return {'mean' : a / b,
            'variance' : a / b**2,
            '95% conf range' : get_ci(stats.gamma(a, scale=1./b))}

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

#-------------------------------------------------------------------------------
# MCMC utils

def effective_size(theta_mcmc):
    from rpy2.robjects.packages import importr
    from rpy2.robjects import FloatVector

    coda = importr('coda')
    es = coda.effectiveSize(FloatVector(theta_mcmc))
    return es[0]

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



def gamma_pdf(x, a, b):
    return np.exp((a - 1) * log(x) + a * log(b) - b * x - gammaln(a))


def chain_dot(*arrs):
    """
    Returns the dot product of the given matrices.

    Parameters
    ----------
    arrs: argument list of ndarray

    Returns
    -------
    Dot product of all arguments.

    Example
    -------
    >>> import numpy as np
    >>> from scikits.statsmodels.tools import chain_dot
    >>> A = np.arange(1,13).reshape(3,4)
    >>> B = np.arange(3,15).reshape(4,3)
    >>> C = np.arange(5,8).reshape(3,1)
    >>> chain_dot(A,B,C)
    array([[1820],
       [4300],
       [6780]])
    """
    return reduce(lambda x, y: np.dot(y, x), arrs[::-1])

