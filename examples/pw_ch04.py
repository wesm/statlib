from __future__ import division

from numpy.random import randn
import numpy as np

from statlib.tools import chain_dot
import statlib.plotting as plotting
import statlib.distributions as dist
import numpy.linalg as la
import scipy.stats as stats
import matplotlib.pyplot as plt

from pandas.util.testing import set_trace as st

m_ = np.array

def simulate_48(pi=0.9, phi=0.9, w=1, v=1, kappa=2, T=200):
    flips = stats.binom(1, pi, size=T).rvs(T)

    noise = randn(T) * np.sqrt(v)
    noise[flips == 0] *= kappa

    inov = randn(T) * np.sqrt(w)

    mu = 0
    out = np.empty(T)
    for t in xrange(T):
        mu = phi * mu + inov[t]
        out[t] = mu + noise[t]

    return flips, out

def mcmc(y, niter=1000, nburn=0, kappa=2, pi=0.9):

    T = len(y)
    p = 1

    kappa2 = kappa ** 2

    alpha_v0 = 1. / 2
    beta_v0 = 1. / 2

    def _update_v(mu, lam):
        alpha = alpha_v0 + T / 2
        beta = beta_v0 + ((y - mu[1:])**2 / lam).sum() / 2
        return 1 / dist.rgamma(alpha, beta)

    alpha_w0 = 1 / 2
    beta_w0 = 1. / 2
    def _update_w(mu, phi):
        alpha = alpha_w0 + T / 2
        beta = beta_w0 + ((mu[1:] - phi * mu[:-1])**2).sum() / 2

        return 1 / dist.rgamma(alpha, beta)

    def _update_phi(mu, w):
        # reference prior for phi
        sumsq = (mu[:-1]**2).sum()
        mean = (mu[1:] * mu[:-1]).sum() / sumsq
        var = w / sumsq
        return dist.rnorm(mean, np.sqrt(var))

    def _update_mu(v, w, phi, lam):
        # FFBS
        # allocate result arrays

        mode = np.zeros((T + 1, p))
        a = np.zeros((T + 1, p))
        C = np.zeros((T + 1, p))
        R = np.zeros((T + 1, p))

        # simple priors...
        mode[0] = 0
        C[0] = np.eye(p)

        # Forward filter

        Ft = m_([[1]])
        for i, obs in enumerate(y):
            t = i + 1

            at = phi * mode[t - 1] if t > 1 else mode[0]
            Rt = phi ** 2 * C[t - 1] + w if t > 1 else C[0]

            Vt = lam[t - 1] * v

            Qt = chain_dot(Ft.T, Rt, Ft) + Vt
            At = np.dot(Rt, Ft) / Qt

            # forecast theta as time t
            ft = np.dot(Ft.T, at)
            err = obs - ft

            # update mean parameters
            mode[t] = at + np.dot(At, err)
            C[t] = Rt - np.dot(At, np.dot(Qt, At.T))
            a[t] = at
            R[t] = Rt

        # Backward sample
        mu = np.zeros((T + 1, p))

        # initial values for smoothed dist'n
        fR = C[-1]
        fm = mode[-1]
        for t in xrange(T + 1):
            if t < T:
                # B_{t} = C_t G_t+1' R_t+1^-1
                B = np.dot(C[t] * phi, la.inv(np.atleast_2d(R[t+1])))

                # smoothed mean
                fm = mode[t] + np.dot(B, mode[t+1] - a[t+1])
                fR = C[t] + chain_dot(B, C[t+1] - R[t+1], B.T)

            mu[t] = dist.rmvnorm(fm, np.atleast_2d(fR))

        return mu.squeeze()

    def _update_lam(mu, v):
        mult = pi / (1 - pi) * kappa
        odds = mult * np.exp(-(y - mu[1:])**2 * (1 - 1/kappa2) / (2 * v))

        # p(\lambda = 1)
        prob = odds / (1 + odds)

        draws = stats.binom(1, prob, size=len(prob)).rvs(len(prob))

        # if we drew 0, set to kappa^2
        draws[draws == 0] = kappa2

        return draws

    v = w = 1
    phi = 1
    lam = np.ones(T)

    mus = np.zeros((niter, T+1))
    lams = np.zeros((niter, T))
    vs = np.zeros(niter)
    ws = np.zeros(niter)
    phis = np.zeros((niter, p))

    for i in xrange(-nburn, niter):
        if not i % 100: print i
        # sample mu
        mu = _update_mu(v, w, phi, lam)

        # sample lambda
        lam = _update_lam(mu, v)

        # sample phi
        phi = _update_phi(mu, w)

        # sample v
        v = _update_v(mu, lam)

        # sample w
        w = _update_w(mu, phi)

        if i >= 0:
            mus[i] = mu
            lams[i] = lam
            vs[i] = vp
            ws[i] = w
            phis[i] = phi

    return mus, lams, vs, ws, phis

def plot_results(y, mus, lams, flips):
    fig, axes = plt.subplots(nrows=2, figsize=(12, 8))

    ax1 = axes[0]

    for i, obs in enumerate(y):
        if flips[i]:
            ax1.plot([i], [obs], 'wo', ms=6)
        else:
            ax1.plot([i], [obs], 'ko', ms=6)

    ax1.plot(mus.mean(0)[1:])

    ax2 = axes[1]
    ax2.plot(lams.mean(0))
    ax2.set_ylim([1, 4])

if __name__ == '__main__':
    np.random.seed(1)
    flips, y = simulate_48(w=1, T=200)

    # plt.figure()
    # plt.plot(y)
    # plt.show()

    mus, lams, vs, ws, phis = mcmc(y, nburn=500)
