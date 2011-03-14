from numpy import sqrt, log, exp, pi
import numpy as np

from scipy.stats import norm, chi2
from scipy.special import gammaln as gamln, gamma as gam
import scipy.stats as stats
import scipy.special as special

## Non-central T distribution

from scipy.stats import rv_continuous

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
    if n:
        return np.random.gamma(a, scale=1./b, size=n)
    else:
        return np.random.gamma(a, scale=1./b)

rnorm = np.random.normal
dnorm = stats.norm.pdf
rtnorm = stats.truncnorm.rvs
rmvnorm = np.random.multivariate_normal

def make_t_ci(df, level, scale, alpha=0.10):
    sigma = stats.t(df).ppf(1 - alpha / 2)
    upper = level + sigma * scale
    lower = level - sigma * scale
    return lower, upper
