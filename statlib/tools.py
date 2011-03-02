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

def nan_array(*shape):
    arr = np.empty(shape, dtype=float)
    arr.fill(np.NaN)

    return arr

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

def hpd_unimodal(dist, mode_guess=0., lo=0., hi=1e6, alpha=0.10):
    # TODO: fix this to work with unimodal but not symmetric dist'n

    mode = opt.fmin(lambda x: -dist.pdf(x), 0)[0]

    lt = opt.bisect(lambda x: dist.pdf(x) / mode - alpha, lo, mode)
    ut = opt.bisect(lambda x: dist.pdf(x) / mode - alpha, mode, hi)

    coverage = dist.cdf(ut) - dist.cdf(lt)

    return lt, ut, coverage

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


#-------------------------------------------------------------------------------
# Array utils


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


def zero_out(arr, tol=1e-15):
    return np.where(np.abs(arr) < tol, 0, arr)

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

    return zero_out(arr)

def fourier_coefs(vals):
    vals = np.asarray(vals)

    p = len(vals)
    theta = 2 * np.pi / p

    angles = theta * np.outer(np.arange(p // 2 + 1), np.arange(p))
    a = (np.cos(angles) * vals).sum(axis=1) * 2 / p
    b = (np.sin(angles) * vals).sum(axis=1) * 2 / p

    a[0] /= 2
    a[-1] /= 2
    return zero_out(a), zero_out(b)
