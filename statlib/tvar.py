"""
Time-varying autoregression (TVAR) model
"""

import numpy as np

import pandas as pn

from statlib.arma import ARDecomp
from statlib.components import AR
import statlib.dlm as dlm
reload(dlm)

class TVAR(dlm.DLM):
    """
    Time-varying autoregression model of order p, TVAR(p)

    Parameters
    ----------
    y : ndarray
        data
    p : int
        lag order
    m0 : ndarray k
        Prior mean for state vector
    C0 : ndarray k x k
        Prior covariance for state vector
    n0 : int
        Prior degrees of freedom for observation variance
    s0 : float
        Prior point estimate of observation variance
    state_discount
    var_discount
    """

    def __init__(self, y, p=1., m0=None, C0=None, n0=None, s0=None,
                 state_discount=0.9, var_discount=1.):
        self.orig_data = y
        self.lag_order = p

        arx = y - y.mean()
        comp = AR(arx, lags=p, intercept=False)
        dlm.DLM.__init__(self, arx[p:], comp.F, G=comp.G,
                         m0=m0, C0=C0, n0=n0, s0=s0,
                         state_discount=state_discount,
                         var_discount=var_discount)

    def decomp(self, smoothed=True):
        """

        """
        results = {}

        if smoothed:
            mus, C, df = self.backward_smooth()
        else:
            mus = self.mu_mode

        for i, mu in enumerate(mus):
            if i == 0:
                continue

            dec = ARDecomp(mu)
            results[i] = dec.result

        panel = pn.WidePanel.fromDict(results, intersect=False)
        panel = panel.swapaxes(0, 2).swapaxes(1, 2)
        return panel

def tvar_gridsearch(model, prange, trange, vrange):
    """
    Search over discount factors and lag orders using one-step predictive
    log-likelihood densities to determine "optimal" TVAR(p) model
    """
    result = np.empty((len(prange), len(trange), len(vrange)))
    maxp = max(prange)

    # could be more efficient, but eh
    for i, p in enumerate(prange):
        print p
        for j, t in enumerate(trange):
            for k, v in enumerate(vrange):
                iter_model = TVAR(model.orig_data, p=p,
                                  m0=model.m0[:p], C0=model.C0[:p, :p],
                                  n0=model.df0, s0=model.s0,
                                  state_discount=t, var_discount=v)

                result[i, j, k] = iter_model.pred_like[maxp-p:].sum()

    return result

if __name__ == '__main__':
    import statlib.datasets as ds

    eeg = ds.eeg_Cz()

    p = 12.
    m0 = np.zeros(p)
    C0 = np.eye(p)
    n0 = 2
    s0 = 50

    model = TVAR(eeg, p=p, m0=m0, C0=C0, n0=n0, s0=s0,
                 state_discount=0.994,
                 var_discount=0.95)
    decomp = model.decomp()

    # result = tvar_gridsearch(model, range(12, 13),
    #                          np.linspace(0.9, 1, num=10),
    #                          np.linspace(0.95, 0.95, num=1))
