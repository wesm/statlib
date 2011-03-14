from numpy cimport double_t, ndarray

cimport numpy as cnp
import numpy as np

import_array()

cdef extern from "math.h":
    double sqrt(double x)

def univ_sv_ffbs(ndarray[double_t, ndim=1] y,
                 double_t phi,
                 ndarray[double_t, ndim=1] v_mean,
                 ndarray[double_t, ndim=1] v_std,
                 ndarray[double_t, ndim=1] lam,
                 double_t w):
    '''
    Forward filter-backward sample for univariate SV model
    '''

    cdef:
        ndarray[double_t, ndim=1] mode, a, C, R, draws
        Py_ssize_t t = 0
        double_t at, At, Rt, Qt, ft, err, obs
        double_t B, fR, fm

    # e.g. AR(1) model

    cdef Py_ssize_t T = len(y)

    mode = np.zeros(T + 1)
    a = np.zeros(T + 1)
    C = np.zeros(T + 1)
    R = np.zeros(T + 1)

    # simple priors...
    mode[0] = mu
    C[0] = v / (1 - phi **2)

    # Forward filter

    for i in range(T):
        obs = y[i]
        t = i + 1

        if t > 1:
            at = phi * mode[t - 1]
            Rt = phi * phi * C[t - 1] + w
        else:
            at = mode[0]
            Rt = C[0]

        Vt = lam[t - 1] * v
        Qt = Rt + Vt
        At = Rt / Qt

        # forecast theta as time t
        ft = at
        err = obs - ft

        # update mean parameters
        mode[t] = at + At * err
        C[t] = Rt - At * At * Qt
        a[t] = at
        R[t] = Rt

    # Backward sample
    mu = np.zeros(T + 1)

    # initial values for smoothed dist'n
    fR = C[-1]
    fm = mode[-1]

    draws = np.random.randn(T + 1)

    for t in xrange(T + 1):
        if t < T:
            # B_{t} = C_t G_t+1' R_t+1^-1
            B = C[t] * phi / R[t+1]

            # smoothed mean
            fm = mode[t] + B * (mode[t+1] - a[t+1])
            fR = C[t] + B * B * (C[t+1] - R[t+1])

        mu[t] = fm + sqrt(fR) * draws[t]

    return mu

def _forward_filter():
    pass

def _backward_sample():
    pass
