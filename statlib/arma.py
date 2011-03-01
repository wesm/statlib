from numpy.random import randn
from numpy.linalg import inv
import numpy as np
import numpy.linalg as LA
import statlib.plotting as plotting

from pandas.util.testing import set_trace as st

from scikits.statsmodels.tsa.arima import ARMA as sm_arma
from scikits.statsmodels.tsa.stattools import acf
from scikits.statsmodels.tools.decorators import cache_readonly

import scikits.statsmodels.api as sm

class ARModel(object):

    def __init__(self, data, p=1):
        self.data = data
        self.p = p

        self.arx = data - data.mean()
        self.exog, self.endog = _prep_arvars(self.arx, p)

    @cache_readonly
    def ref_results(self):
        return sm.OLS(self.exog, self.endog).fit()

    @property
    def dlm_rep(self):
        """
        DLM form of AR(p) state transition matrix
        """
        beta = LA.lstsq(self.endog, self.exog)[0]
        lower = np.c_[np.eye(self.p-1), np.zeros((self.p-1, 1))]
        return np.vstack((beta, lower))

    def decomp(self):
        """
        Compute decomposition of input time series into components

        Notes
        -----
        W&H p. 302
        Compute G = B A B^-1
        Decompose x_t,j = (H theta_t)_j where H = B'E B^-1
        E = (1, 0, ..., 0)

        Returns
        -------
        tup : tuple (waves, mods, decomp)
            wavelengths : length k
            moduli : length k
            decomp : n x k
        """
        p = self.p
        X = self.endog

        vals, vecs = LA.eig(self.dlm_rep)

        H = np.dot(np.diag(vecs[0]), inv(vecs))

        mods = np.abs(vals)
        angles = np.angle(vals)
        angles[angles == 0] = np.pi

        # Put everything in order of increasing angles and ignore negative
        # angles
        sorter = np.argsort(angles)
        angles = angles[sorter]
        mask = angles > 0

        angles = angles[mask]
        mods = mods[sorter][mask]
        vecs = vecs[sorter][mask]

        # convert to real
        H = H[sorter][mask].real
        # double the complex H coefficients, z_t,j = x_t,d + x_t,h
        # complex components are at the beginning because we set the real
        # component angles to pi
        H[:(-mask).sum()] *= 2

        states = np.c_[np.zeros((p, p-1)),   # "initial states"
                       X.T,                  # regressor matrix
                       self.arx[:-(p+1):-1]] # time t
        decomp = np.dot(H, states).T

        waves = 2 * np.pi / angles

        return waves, mods, decomp

    def plot_acf(self, lags=50, partial=True):
        plotting.plot_acf(self.data, lags, partial=partial)

class TestARModel(object):

    def test_decomp(self):
        pass

def _prep_arvars(x, p):
    x = x - x.mean()
    y = x[p:]
    X = np.empty((p, len(y)))

    for i in range(1, p + 1):
        X[i-1] = x[p-i:-i]

    return y, X.T

def ma_coefs(coefs, maxn=10):
    p = len(coefs)
    phis = np.zeros(maxn+1)
    # phis[0] = 1

    # recursively compute MA coefficients

    # for i in range(1, p):
    #     for j in range(1, maxn):
    #         phis[j] += coefs[i] * phis[j - i]

    for i in xrange(1, maxn + 1):
        for j in xrange(1, i+1):
            if j > p:
                break
            phis[i] += phis[i-j] * coefs[j-1]

    return phis

class ARMA(object):
    """
    Estimation of ARMA(p, q) model via MCMC

    Notes
    -----
    see Prado & West, pp. 75-77
    cf. Marriott, Ravishanker, Gelfand, Pai (1996)
    """

    pass

def ar_simulate(phi, s, n, dist='normal'):
    """

    """
    if dist == 'normal':
        out = randn(n, 1)
    else:
        raise ValueError

    out[0] *= np.sqrt(s) # marginal variance
    out[1:] *= np.sqrt((1 - phi**2) * s) # innovation variance

    for t in xrange(1, n):
        out[t] += phi * out[t - 1]

    return out

if __name__ == '__main__':
    from statlib.linmod import BayesLM
    import statlib.datasets as ds
    eeg = ds.eeg_data()

    # model = sm_arma(eeg)
    # res = model.fit(order=(8, 0), method='css', disp=-1)

    model = ARModel(eeg, p=8)

    p = 8
    y, X = _prep_arvars(eeg, p)

    beta_prior = (np.zeros(p), np.eye(p) * 100)
    var_prior = (2, 1000)

    model2 = BayesLM(y, X, beta_prior, var_prior)
