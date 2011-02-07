from numpy import sqrt, log, exp, pi
import numpy as np

from scipy.stats import norm, chi2
from scipy.special import gammaln as gamln, gamma as gam
import scipy.stats as stats
import scipy.special as special

## Non-central T distribution

from scipy.stats import rv_continuous


def make_t_ci(df, level, scale, alpha=0.10):
    sigma = stats.t(df).ppf(1 - alpha / 2)
    upper = level + sigma * scale
    lower = level - sigma * scale
    return lower, upper
