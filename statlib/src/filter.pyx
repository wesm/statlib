# cython: profile=True

from numpy cimport ndarray, double_t

import numpy as np
from statlib.tools import chain_dot, nan_array

def filter_python_multi(Y, F, G, delta, df0, v0, m0, C0):
    """
    Univariate DLM update equations with unknown observation variance

    Done with numpy.matrix objects so works with multivariate case. About 3-4x
    slower than specializing to the 1d case and using vectors / regular ndarrays
    """
    nobs = len(Y)
    ndim = len(G)

    mode = nan_array(nobs + 1, ndim)
    a = nan_array(nobs + 1, ndim)
    C = nan_array(nobs + 1, ndim, ndim)
    S = nan_array(nobs + 1)
    Q = nan_array(nobs)
    R = nan_array(nobs + 1, ndim, ndim)

    df = df0 + np.arange(nobs + 1) # precompute
    S[0] = St = v0 * 1.

    mode[0] = m0
    C[0] = Ct = np.asmatrix(C0)

    mt = np.asmatrix(m0).T
    Y = np.asmatrix(Y).T
    F = np.asmatrix(F)
    G = np.asmatrix(G)

    # allocate result arrays
    for i in xrange(nobs):
        # column vector, for W&H notational consistency
        Yt = Y[i].T
        Ft = F[i].T

        # advance index: y_1 through y_nobs, 0 is prior
        t = i + 1

        # derive innovation variance from discount factor
        at = mt
        Rt = Ct
        if t > 1:
            # only discount after first time step!
            if G is not None:
                at = G * mt
                Rt = G * Ct * G.T / delta
            else:
                Rt = Ct / delta

        Qt = Ft.T * Rt * Ft + St
        At = Rt * Ft / Qt

        # forecast theta as time t
        ft = Ft.T * at
        e = Yt - ft

        # update mean parameters
        mt = at + At * e

        St = St + (St / df[t]) * (e * e.T / Qt - 1)
        Ct = (St / S[t-1]) * (Rt - At * At.T * Qt)

        S[t] = St
        C[t] = Ct
        Qt[t-1] = Qt
        mode[t] = mt.T
        a[t] = at.T
        R[t] = Rt

    return mode, a, C, S, Q, R

def filter_python_old(Y, F, G, delta, df0, v0, m0, C0):
    """
    Univariate DLM update equations with unknown observation variance
    """
    nobs = len(Y)
    ndim = len(G)

    mode = nan_array(nobs + 1, ndim)
    a = nan_array(nobs + 1, ndim)
    C = nan_array(nobs + 1, ndim, ndim)
    S = nan_array(nobs + 1)
    Q = nan_array(nobs)
    R = nan_array(nobs + 1, ndim, ndim)

    mode[0] = m0
    C[0] = C0

    df = df0 + np.arange(nobs + 1) # precompute
    S[0] = v0

    # allocate result arrays
    for i, obs in enumerate(Y):
        # column vector, for W&H notational consistency
        Ft = F[i:i+1].T

        # advance index: y_1 through y_nobs, 0 is prior
        t = i + 1

        # derive innovation variance from discount factor
        at = mode[t - 1]
        Rt = C[t - 1]
        if t > 1:
            # only discount after first time step!
            if G is not None:
                at = np.dot(G, mode[t - 1])
                Rt = chain_dot(G, C[t - 1], G.T) / delta
            else:
                Rt = C[t - 1] / delta

        Qt = chain_dot(Ft.T, Rt, Ft) + S[t-1]
        At = np.dot(Rt, Ft) / Qt

        # forecast theta as time t
        ft = np.dot(Ft.T, at)
        e = obs - ft

        # update mean parameters
        mode[t] = at + np.dot(At, e)
        S[t] = S[t-1] + (S[t-1] / df[t]) * ((e ** 2) / Qt - 1)
        C[t] = (S[t] / S[t-1]) * (Rt - np.dot(At, At.T) * Qt)

        a[t] = at
        Q[t-1] = Qt
        R[t] = Rt

    return mode, a, C, df, S, Q, R

def filter_python(ndarray[double_t, ndim=1] Y,
                  ndarray F, ndarray G,
                  double_t delta, double_t beta,
                  double_t df0, double_t v0,
                  ndarray m0, ndarray C0):
    """
    Univariate DLM update equations with unknown observation variance

    delta : state discount
    beta : variance discount
    """
    cdef:
        Py_ssize_t i, t, nobs, ndim
        ndarray[double_t, ndim=1] df, Q, S
        ndarray a, C, R, mode

        ndarray at, mt, Ft, At, Ct, Rt
        double_t obs, ft, e, nt, dt, St, Qt

    nobs = len(Y)
    ndim = len(G)

    mode = nan_array(nobs + 1, ndim)
    a = nan_array(nobs + 1, ndim)
    C = nan_array(nobs + 1, ndim, ndim)
    R = nan_array(nobs + 1, ndim, ndim)
    S = nan_array(nobs + 1)
    Q = nan_array(nobs)
    df = nan_array(nobs + 1)

    mode[0] = mt = m0
    C[0] = Ct = C0
    df[0] = nt = df0
    S[0] = St = v0

    dt = df0 * v0

    # allocate result arrays
    for i in range(nobs):
        obs = Y[i]

        # column vector, for W&H notational consistency
        # Ft = F[i]
        Ft = F[i:i+1].T

        # advance index: y_1 through y_nobs, 0 is prior
        t = i + 1

        # derive innovation variance from discount factor
        at = mt
        Rt = Ct
        if t > 1:
            # only discount after first time step?
            if G is not None:
                at = np.dot(G, mt)
                Rt = chain_dot(G, Ct, G.T) / delta
            else:
                Rt = Ct / delta

        # Qt = chain_dot(Ft.T, Rt, Ft) + St
        Qt = chain_dot(Ft.T, Rt, Ft) + St
        At = np.dot(Rt, Ft) / Qt

        # forecast theta as time t
        ft = np.dot(Ft.T, at)
        e = obs - ft

        # update mean parameters
        mode[t] = mt = at + (At * e).squeeze()
        dt = beta * dt + St * e * e / Qt
        nt = beta * nt + 1
        St = dt / nt

        S[t] = St
        Ct = (S[t] / S[t-1]) * (Rt - np.dot(At, At.T) * Qt)
        Ct = (Ct + Ct.T) / 2 # symmetrize

        df[t] = nt
        Q[t-1] = Qt
        C[t] = Ct
        a[t] = at
        R[t] = Rt

    return mode, a, C, df, S, Q, R
