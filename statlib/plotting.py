from __future__ import division

import numpy as np

from matplotlib import cm
import matplotlib as mpl
import scipy.stats as stats
import matplotlib.pyplot as plt

from scikits.statsmodels.tsa.stattools import acf, pacf

#-------------------------------------------------------------------------------
# Graphing functions

def density_plot(y, thresh=1e-10, style='k', ax=None):
    """
    Make kernel density plot of input data
    """
    kde = stats.kde.gaussian_kde(y)
    plot_support(kde.evaluate, y.max(), y.min(),
                 thresh=thresh, style=style, ax=ax)

def plot_support(f, hi, lo, thresh=1e-10, style='k', N=5000, ax=None):
    """
    Attempt to plot input function (e.g. a probability density) over its support
    (places where its value exceeds a certain threshold)

    Parameters
    ----------
    f : function
        Vectorized function, e.g. a scipy.stats distribution pdf
    hi : float
        Upper bound for search
    lo : float
        Lower bound for search
    thresh : float

    Returns
    -------
    handle : result of plot function
    """

    intervals = np.linspace(lo - np.abs(lo), hi + np.abs(hi), num=N)
    pdfs = f(intervals)

    # try to find support
    above = (pdfs > thresh)

    if not above.any():
        raise Exception('Did not find any points exceeding support threshold')

    supp_l, supp_u = above.argmax(), (len(above) - above[::-1].argmax() - 1)
    xs = np.linspace(intervals[supp_l], intervals[supp_u], num=N)

    if ax is None:
        ax = plt.gca()

    return ax.plot(xs, f(xs), style)

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

def plot_acf(y, lags=100, partial=False):
    if partial:
        the_acf = pacf(y, nlags=lags)
    else:
        the_acf = acf(y, nlags=lags)

    plt.figure(figsize=(10, 5))
    plt.vlines(np.arange(lags+1), [0], the_acf)

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


def plot_fourier_rep(vals, harmonic=None):
    a, b = tools.fourier_coefs(vals)
    theta = 2 * np.pi / len(vals)

    p = len(vals)

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

    plotf(np.vectorize(f, [np.float64]), 0, len(vals) + 1)
    plt.vlines(np.arange(p), 0, vals)
    plt.axhline(0)

