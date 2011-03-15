cimport cython

from numpy cimport double_t, ndarray, import_array

cimport numpy as cnp
import numpy as np

import_array()

cdef extern from "math.h":
    double sqrt(double x)

def univ_ffbs(ndarray[double_t, ndim=1] Y, # data
              ndarray[double_t, ndim=1] F,
              ndarray[double_t, ndim=1] G, # system "matrices"
              ndarray[double_t, ndim=1] V, # observation variance sequence
              double_t W,   # evolution variance
              double_t m0,  # prior mean for theta_0
              double_t C0): # prior variance for theta_0
    '''
    Forward filter-backward sample for univariate model for known variance
    sequence

    e.g. AR(1) model
    '''
    cdef:
        ndarray[double_t, ndim=1] mode, a, C, R
        double_t Ft, Rt, Qt, Gt, At, ft, at, err, obs
        Py_ssize_t T = len(Y)
        Py_ssize_t i, t

    mode = np.zeros(T + 1)
    a = np.zeros(T + 1)
    C = np.zeros(T + 1)
    R = np.zeros(T + 1)

    # set priors
    mode[0] = m0
    C[0] = C0

    # Forward filter
    for i in range(T):
        obs = Y[i]
        Ft = F[i]
        Gt = G[i]

        t = i + 1

        if t > 1:
            at = Gt * mode[t - 1]
            Rt = Gt * C[t - 1] * Gt + W
        else:
            at = mode[0]
            Rt = C[0]

        Qt = Ft * Rt * Ft + V[i]
        At = Ft * Rt / Qt

        # forecast theta as time t
        ft = Ft * at
        err = obs - ft

        # update mean parameters
        mode[t] = at + At * err
        C[t] = Rt - At * At * Qt
        a[t] = at
        R[t] = Rt

    # Backward sample
    # initial values for smoothed dist'n
    cdef ndarray[double_t, ndim=1] draws, mu_draws
    mu_draws = np.zeros(T + 1)
    draws = np.random.randn(T + 1)

    mu_draws[T] = mode[T] + sqrt(C[T]) * draws[T]

    cdef double_t Bt, Ht, ht

    t = T - 1
    while t >= 0:
        # B_{t} = C_t G_t+1' R_t+1^-1
        Bt = C[t] * G[t] / R[t+1]

        # P&W p. 130 eq 4.12 and 4.13
        ht = mode[t] + Bt * (mu_draws[t+1] - a[t+1])
        Ht = C[t] - Bt * R[t+1] * Bt

        mu_draws[t] = ht + sqrt(Ht) * draws[t]

        t -= 1

    return mu_draws

def sample_discrete(ndarray[double_t, ndim=2] probs):
    '''
    Random
    '''

    cdef cnp.int32_t i, K = probs.shape[1], T = len(probs)
    cdef ndarray[double_t, ndim=1] draws = np.random.rand(T)
    cdef ndarray[cnp.int32_t, ndim=1] output = np.empty(T, dtype=np.int32)

    cdef double_t the_sum, draw
    for i from 0 <= i < T:
        the_sum = 0
        draw = draws[i]
        for j from 0 <= j < K:
            the_sum += probs[i, j]

            if the_sum >= draw:
                output[i] = j
                break

    return output

def _forward_filter():
    pass

def _backward_sample():
    pass
