import numpy as np

import scipy.stats as stats
import scipy.special as special

import matplotlib.pyplot as plt
import statlib.plotting as plotting

#-------------------------------------------------------------------------------
# Distribution-related functions

def rtrunc_norm(mean, sd, lower, upper, size=None):
    """
    Sample from a truncated normal distribution

    Parameters
    ----------
    mean : float or array_like
    sd : float or array_like
    lower : float or array-like
    upper : float or array-like

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

    if size is None:
        size = len(ulower) if isinstance(ulower, np.ndarray) else 1
    else:
        raise ValueError('if array of bounds passed, size must be None')

    u = (uupper - ulower) * np.random.rand(size) + ulower
    return mean + sd * special.ndtri(u)

def rgamma(a, b, n=None):
    """
    Sample from gamma(a, b) distribution using rate parameterization. For
    reducing cognitive dissonance moving between R and Python
    """
    return np.random.gamma(a, scale=1./b, size=n)

rnorm = np.random.normal
dnorm = stats.norm.pdf
rtnorm = stats.truncnorm.rvs

# rmvnorm = np.random.multivariate_normal

def make_t_ci(df, level, scale, alpha=0.10):
    sigma = stats.t(df).ppf(1 - alpha / 2)
    upper = level + sigma * scale
    lower = level - sigma * scale
    return lower, upper

def rmvnorm(mu, cov, size=1):
    """
    Compute multivariate normal draws from possibly singular covariance matrix
    using singular value decomposition (SVD)
    """
    # TODO: find reference
    U, s, Vh = np.linalg.svd(cov)
    cov_sqrt = np.dot(U * np.sqrt(s), Vh)

    # N(0, 1) draws
    draws = np.random.randn(size, len(cov))
    return np.dot(draws, cov_sqrt) + mu

# TODO arbitrary mixture...e.g. betas or gammas

class NormalMixture(object):
    """
    Simple class for storing mixture of normals

    Parameters
    ----------
    """
    def __init__(self, means, variances, weights):
        if not (len(means) == len(variances) == len(weights)):
            raise ValueError('inputs must be all same length sequences')

        self.means = np.asarray(means)
        self.variances = np.asarray(variances)
        self.weights = np.asarray(weights)

        self.dists = [stats.norm(m, np.sqrt(v))
                      for m, v in zip(self.means, self.variances)]

    def pdf(self, x):
        """
        Evaluate mixture probability density function
        """
        density = 0
        for w, dist in zip(self.weights, self.dists):
            density += w * dist.pdf(x)

        return density

    def plot(self, style='k', ax=None, **plot_kwds):

        max_sd = np.sqrt(self.variances.max())

        # overkill perhaps
        lo = self.means.min() - 4 * max_sd
        hi = self.means.max() + 4 * max_sd

        if ax is None:
            ax = plt.gca()

        plotting.plot_support(self.pdf, lo, hi, style=style,
                              **plot_kwds)

if __name__ == '__main__':
    sigma = np.array([[1, 2], [2, 4]], dtype=float)
    mean = [0, 1]

    x = rmvnorm(mean, sigma, size=10000)
