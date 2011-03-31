from numpy.random import randn
from numpy.linalg import inv
import numpy as np
import numpy.linalg as LA
import scipy.stats as stats
import statlib.plotting as plotting
import statlib.dlm as dlm

from pandas.util.testing import set_trace, debug

from scikits.statsmodels.tools.decorators import cache_readonly
import scikits.statsmodels.tsa.api as tsa_api

import scikits.statsmodels.api as sm

class ARModel(object):

    def __init__(self, data, p=1):
        self.data = data
        self.p = p

        self.arx = data - data.mean()
        self.endog, self.exog = _prep_arvars(self.arx, p)

    @cache_readonly
    def ref_results(self):
        return tsa_api.AR(self.arx).fit(self.p, trend='nc')

    @property
    def dlm_rep(self):
        """
        DLM form of AR(p) state transition matrix
        """
        beta = LA.lstsq(self.exog, self.endog)[0]
        return ar_dlm_rep(beta)

    def fit_dlm(self, m0, C0, n0, s0, discount=1.):
        """
        Fit dynamic linear model to the data, with the AR parameter as the state
        vector

        Parameters
        ----------
        m0 : ndarray (p)
        M0 : ndarray (p x p)
        n0 : int / float
        s0 : float

        Returns
        -------
        model : DLM
        """
        return dlm.DLM(self.endog, self.exog, m0=m0, C0=C0, n0=n0, s0=s0,
                       state_discount=discount)

    def decomp(self):
        beta = LA.lstsq(self.exog, self.endog)[0]
        return ARDecomp(beta)

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

def ar_dlm_rep(phi):
    p = len(phi)
    lower = np.c_[np.eye(p-1), np.zeros((p-1, 1))]
    return np.vstack((phi, lower))

class ARDecomp(object):
    """
    Compute decomposition of input time series into components using eigenvalue
    decomposition

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

    def __init__(self, phi):
        self.phi = phi
        self.p = len(phi)

        (self.modulus, self.frequency,
         self.wavelength, self.H) = self._compute_decomp()

    def __repr__(self):
        from cStringIO import StringIO
        from pandas import DataMatrix

        table = DataMatrix({'wavelength' : self.wavelength,
                            'modulus' : self.modulus,
                            'frequency' : self.frequency},
                           columns=['modulus', 'wavelength', 'frequency'])

        buf = StringIO()

        print >> buf, 'AR(%d) decomposition' % self.p
        print >> buf, table

        return buf.getvalue()

    def _compute_decomp(self):
        # see class docstring
        vals, vecs = LA.eig(ar_dlm_rep(self.phi))

        # (B_t'F) B_t^-1, where F = [1, 0, ..., 0]
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
        waves = 2 * np.pi / angles

        # convert to real
        H = H[sorter][mask].real
        # double the complex H coefficients, z_t,j = x_t,d + x_t,h
        # complex components are at the beginning because we set the real
        # component angles to pi
        H[:(-mask).sum()] *= 2

        return mods, angles, waves, H

    def decompose_ts(self, y, X):
        """
        y : response variable (n)
            Need for time t
        X : lagged responses (n x p)
        """
        states = np.c_[np.zeros((self.p, self.p-1)),   # "initial states"
                       X.T,                  # regressor matrix
                       y[:-(self.p+1):-1]] # time t
        return np.dot(self.H, states).T

def ar_decomp(phi, y, X):
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

    return waves, mods, decomp


def ar_model_like(data, maxp, m0, M0, n0, s0):
    """
    Compute approximate marginal likelihood for univariate AR(p) model for
    p=0,...maxp.

    Parameters
    ----------
    data
    maxp : int
        Maximum model order
    m0 : ndarray (maxp x maxp)
        Prior mean for state
    M0 : ndarray (maxp x maxp)
        Prior state variance matrix
    n0 : int / float
        Prior degrees of freedom for obs var
    s0 : float
        Prior estimate of observation variance

    Notes
    -----
    p(\theta) ~ N(m0, M0)
    \nu ~ IG(n0 / 2, n0 * s0 / 2)
    """
    from scipy.special import gammaln as gamln

    arx = data - data.mean()
    nobs = len(arx)

    result = np.zeros(maxp + 1)
    for p in xrange(maxp + 1):
        llik = 0.
        model = ARModel(data, p=p)

        mt = m0[:p]; Mt = M0[:p, :p]
        st = s0; nt = n0

        # dlm_model = model.fit_dlm(m0[:p], M0[:p, :p], n0, s0)
        # F = dlm_model.F
        # y = dlm_model.y

        # for t in xrange(p, nobs - p):
        #     nt = dlm_model.df[t]
        #     q = dlm_model.forc_var[t-1]
        #     st = dlm_model.var_est[t - 1]
        #     e = y[t - 1] - np.dot(F[t - 1], dlm_model.mu_mode[t - 1])

        #     # Student-t log pdf
        #     sd = np.sqrt(q * st)
        #     llik += stats.t(nt).logpdf(e / sd) - np.log(sd)

        for t in xrange(p, nobs):
            # DLM update equations
            x = arx[t-p:t][::-1]
            A = np.dot(Mt, x); q = np.dot(x, A) + 1; A /= q
            e = arx[t] - np.dot(x, mt)

            if t >= 2 * maxp:
                sd = np.sqrt(q * st)
                llik += stats.t(nt).logpdf(e / sd) - np.log(sd)
            mt = mt + A * e
            st = (nt * st + np.dot(e, e) / q) / (nt + 1)
            nt = nt + 1
            Mt = Mt - np.outer(A, A) * q

        result[p] = llik

    popt = result.argmax()
    maxlik = result[popt]

    return result

if __name__ == '__main__':
    from statlib.linmod import BayesLM
    import statlib.datasets as ds
    eeg = ds.eeg400_data()

    # model = sm_arma(eeg)
    # res = model.fit(order=(8, 0), method='css', disp=-1)

    model = ARModel(eeg, p=8)

    p = 8
    y, X = _prep_arvars(eeg, p)

    m0, M0 = np.zeros(p), np.eye(p) * 100
    n0, s0 = 2, 1000
    beta_prior = (m0, M0)
    var_prior = (n0, s0)

    model2 = BayesLM(y, X, beta_prior, var_prior)

    maxp = 20
    result = ar_model_like(eeg, maxp, np.zeros(maxp),
                           np.eye(maxp) * 1, 1, 1)
