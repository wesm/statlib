import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from statlib.tools import nan_array, chain_dot
from statlib.dlm import ConstantDLM, st
import statlib.tools as tools
import statlib.plotting as plotting

import statlib.datasets as datasets
from pandas import DataMatrix

class DLMMixture(object):
    """
    Mixture of DLMs (Class I type model)

    Parameters
    ----------
    models : dict

    Notes
    -----
    cf. W&H Section 12.2
    """
    def __init__(self, models):
        self.models = models
        self.names = sorted(models.keys())

        mod = self.models.values()[0]
        self.pred_like = DataMatrix(dict((k, v.pred_like)
                                         for k, v in models.iteritems()),
                                    index=mod.dates)

    @property
    def post_model_prob(self):
        cumprod = self.pred_like.cumprod()
        return cumprod / cumprod.sum(1)

    def plot_post_prob(self):
        ratio = self.post_model_prob
        ratio.plot(subplots=True, sharey=True)
        ax = plt.gca()
        ax.set_ylim([0, 1])

    def get_weights(self, t):
        weights = self.post_model_prob
        return weights.xs(weights.index[t])

    def plot_mu_density(self, t, index=0, support_thresh=0.1):
        """
        Plot posterior densities for single model parameter over the set of
        mixture components

        Parameters
        ----------
        t : int
            time index, relative to response variable
        index : int
            parameter index to plot

        Notes
        -----
        cf. West & Harrison Figure 12.3. Automatically annotating individual
        component curves would probably be difficult.
        """
        ix = index
        dists = {}
        for name in self.names:
            model = self.models[name]
            df = model.df[t]
            mode = model.mu_mode[t + 1, ix]
            scale = np.sqrt(model.mu_scale[t + 1, ix, ix])
            dists[name] = stats.t(df, loc=mode, scale=scale)

        plot_mixture(dists, self.get_weights(t),
                           support_thresh=support_thresh)

    def plot_forc_density(self, t, support_thresh=0.1):
        """
        Plot posterior densities for 1-step forecasts

        Parameters
        ----------
        t : int
            time index, relative to response variable

        Notes
        -----
        cf. West & Harrison Figure 12.4.
        """

        dists = {}
        for name in self.names:
            model = self.models[name]
            df = model.df[t]
            mode = model.forecast[t]
            scale = np.sqrt(model.forc_var[t])
            dists[name] = stats.t(df, loc=mode, scale=scale)

        plot_mixture(dists, self.get_weights(t),
                     support_thresh=support_thresh)

def plot_mixture(dists, weights, hi=1, lo=-1, support_thresh=0.1):
    """

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    def mix_pdf(x):
        tot = 0

        for name, dist in dists.iteritems():
            tot += weights[name] * dist.pdf(x)

        return tot

    # plot mixture
    mix = plotting.plot_support(mix_pdf, hi, lo,
                                thresh=support_thresh,
                                style='k',
                                ax=ax)

    for _, dist in dists.iteritems():
        comp = plotting.plot_support(dist.pdf, hi, lo,
                                     thresh=support_thresh,
                                     style='k--',
                                     ax=ax)

    ax.legend((mix, comp), ('Mixture', 'Component'))


class MultiProcessDLM(object):
    """
    Class II type DLM approximate mixture model with constant prior model
    selection probabilities.

    - Constant, unknown variance
    -

    Parameters
    ----------


    Notes
    -----
    cf. W&H Section 12.4
    """
    def __init__(self, y, F, models, order, prior_model_prob,
                 mean_prior=None, var_prior=None, approx_steps=1.):

        self.dates = y.index
        self.y = np.array(y)
        self.nobs = len(y)

        if self.y.ndim == 1:
            pass
        else:
            raise Exception

        F = np.array(F)
        if F.ndim == 1:
            F = F.reshape((len(F), 1))
        self.F = F

        self.ndim = self.F.shape[1]
        self.nmodels = len(models)

        self.names = order
        self.models = [models[name] for name in order]
        self.prior_model_prob = np.array([prior_model_prob[name]
                                          for name in order])

        # only can do one step back for now
        self.approx_steps = 1
        # self.approx_steps = int(approx_steps)

        self.mean_prior = mean_prior
        self.var_prior = var_prior

        # set up result storage for all the models
        self.marginal_prob = nan_array(self.nobs + 1, self.nmodels)
        self.post_prob = nan_array(self.nobs + 1,
                                   self.nmodels, self.nmodels)

        self.mu_mode = nan_array(self.nobs + 1, self.nmodels, self.ndim)
        self.mu_forc_mode = nan_array(self.nobs + 1, self.nmodels, self.ndim)
        self.mu_scale = nan_array(self.nobs + 1, self.nmodels,
                                  self.ndim, self.ndim)
        self.mu_forc_var = nan_array(self.nobs + 1, self.nmodels, self.nmodels,
                                     self.ndim, self.ndim)

        self.forecast = np.zeros((self.nobs + 1, self.nmodels))

        # set in initial values
        self.marginal_prob[0] = self.prior_model_prob
        self.mu_mode[0], self.mu_scale[0] = mean_prior

        # observation variance stuff
        n, d = var_prior
        self.df = n + np.arange(self.nobs + 1) # precompute
        self.var_est = nan_array(self.nobs + 1, self.nmodels)
        self.var_scale = nan_array(self.nobs + 1, self.nmodels)
        self.var_est[0] = d / n
        self.var_scale[0] = d

        # forecasts are computed via mixture for now
        self.forc_var = nan_array(self.nobs, self.nmodels, self.nmodels)

        self._compute_parameters()

    def _compute_parameters(self):
        for i, obs in enumerate(self.y):
            Ft = self._get_Ft(i)

            # advance index: y_1 through y_nobs, 0 is prior
            t = i + 1

            at = self._update_at(t)
            Rt = self._update_Rt(t)
            Qt = self._update_Qt(t, Ft, Rt)
            At = self._update_At(Ft, Rt, Qt)

            # forecast theta as time t
            ft = self._update_ft(Ft, at)
            errs = obs - ft
            mt = self._update_mt(at, At, errs)

            # update scale, then compute scale / degrees of freedom
            dt = self._update_dt(t, Qt, errs)
            St = dt / self.df[t]

            Ct = self._update_Ct(t, Rt, Qt, At, St)

            # update posterior model probabilities

            # p_t(j_t, j_t-1)
            post_prob = self._update_postprob(t, errs, Qt)
            # p_t(j_t)
            marginal_post = post_prob.sum(axis=1)

            # collapse variance estimates
            coll_St = self._collapse_var(St, post_prob, marginal_post)

            # p*_t(j_t, j_t-1) = (St(jt) / St(jt, jtp)) * pt(jt, jtp) / pt(jt)
            pstar = ((coll_St / St.T) * (post_prob.T / marginal_post)).T

            # collapse posterior means / scales
            coll_m, coll_C = self._collapse_params(pstar, mt, Ct)

            # store results
            self.marginal_prob[t] = marginal_post
            self.post_prob[t] = post_prob

            self.mu_mode[t] = coll_m
            self.mu_scale[t] = coll_C
            self.forc_var[t-1] = Qt

            self.var_est[t] = coll_St
            self.var_scale[t] = coll_St * self.df[t] # top of p. 469

            # both are same for each jt
            self.mu_forc_mode[t] = at[0]
            self.forecast[t] = ft[0]
            self.mu_forc_var[t] = Rt

    def _update_at(self, t):
        def calc_update(jt, jtp):
            prior_mode = self.mu_mode[t - 1, jtp]
            G = self.models[jt].G
            # use prior for t=1, though not sure exactly why
            return np.dot(G, prior_mode) if t > 1 else prior_mode

        return self._fill_updates(calc_update, (self.ndim,))

    def _update_Rt(self, t):
        def calc_update(jt, jtp):
            model = self.models[jt]
            G = model.G
            prior_scale = self.mu_scale[t - 1, jtp]
            if t > 1:
                # only discount after first time step! hmm
                Wt = model.get_Wt(prior_scale)
                Rt = chain_dot(G, prior_scale, G.T) + Wt
            else:
                # use prior for t=1, because Mike does
                Rt = prior_scale

            return Rt

        return self._fill_updates(calc_update, (self.ndim, self.ndim))

    def _update_Qt(self, t, Ft, Rt):
        def calc_update(jt, jtp):
            prior_obs_var = (self.var_est[t - 1, jtp] *
                             self.models[jt].obs_var_mult)
            return chain_dot(Ft.T, Rt[jt, jtp], Ft) + prior_obs_var

        return self._fill_updates(calc_update, ())

    def _update_ft(self, Ft, at):
        def calc_update(jt, jtp):
            return np.dot(Ft.T, at[jt, jtp])

        return self._fill_updates(calc_update, ())

    def _update_At(self, Ft, Rt, Qt):
        def calc_update(jt, jtp):
            return np.dot(Rt[jt, jtp], Ft) / Qt[jt, jtp]

        return self._fill_updates(calc_update, (self.ndim,))

    def _update_mt(self, at, At, errs):
        def calc_update(jt, jtp):
            return at[jt, jtp] + np.dot(At[jt, jtp], errs[jt, jtp])

        return self._fill_updates(calc_update, (self.ndim,))

    def _update_dt(self, t, Qt, errs):
        # W&H p. 468
        def calc_update(jt, jtp):
            prior = self.var_scale[t - 1, jtp]
            prior_var = self.var_est[t - 1, jtp]

            return prior + prior_var * errs[jt, jtp] ** 2 / Qt[jt, jtp]

        return self._fill_updates(calc_update, ())

    def _update_Ct(self, t, Rt, Qt, At, St):
        def calc_update(jt, jtp):
            R = Rt[jt, jtp]
            Q = Qt[jt, jtp]
            S = St[jt, jtp]

            # need to make this a column vector
            A = np.atleast_2d(At[jt, jtp]).T

            prior_var = self.var_est[t - 1, jtp]
            return (S / prior_var) * (R - np.dot(A, A.T) * Q)

        return self._fill_updates(calc_update, (self.ndim, self.ndim))

    def _fill_updates(self, func, shape):
        total_shape = (self.nmodels, self.nmodels) + shape
        result = nan_array(*total_shape)
        for jt in range(self.nmodels):
            for jtp in range(self.nmodels):
                result[jt, jtp] = np.squeeze(func(jt, jtp))

        return result

    def _update_postprob(self, t, errs, Qt):
        # degrees of freedom for T dist'n
        dist = stats.t(self.df[t - 1])

        rQ = np.sqrt(Qt)
        pi = self.prior_model_prob
        prior = self.marginal_prob[t - 1]
        result = ((prior * dist.pdf(errs / rQ) / rQ).T * pi).T

        # normalize
        return result / result.sum()

    def _collapse_var(self, St, post_prob, marginal_post):
        result = nan_array(self.nmodels)

        # TODO: vectorize
        for jt in range(self.nmodels):
            prec = 0
            for jtp in range(self.nmodels):
                prec += post_prob[jt, jtp] / (St[jt, jtp] * marginal_post[jt])

            result[jt] = 1 / prec

        return result

    def _collapse_params(self, pstar, mt, Ct):
        coll_C = nan_array(self.nmodels, self.ndim, self.ndim)
        coll_m = nan_array(self.nmodels, self.ndim)
        for jt in range(self.nmodels):
            C = np.zeros((self.ndim, self.ndim))
            m = np.zeros(self.ndim)

            # collapse modes
            for jtp in range(self.nmodels):
                m += pstar[jt, jtp] * mt[jt, jtp]

            coll_m[jt] = m

            # collapse scales
            for jtp in range(self.nmodels):
                mdev = coll_m[jt] - mt[jt, jtp]
                C += pstar[jt, jtp] * (Ct[jt, jtp] + np.outer(mdev, mdev))

            coll_C[jt] = C

        return coll_m, coll_C

    def _get_Ft(self, t):
        return self.F[0:1].T

    def plot_mu_density(self, t, index=0, support_thresh=None):
        ix = index
        dists = {}
        weights = {}

        thresh = 0
        for i in range(self.nmodels):
            df = self.df[t]
            mode = self.mu_mode[t + 1, i, ix]
            scale = np.sqrt(self.mu_scale[t + 1, i, ix, ix])
            dist = stats.t(df, loc=mode, scale=scale)
            dists[i] = dist
            weights[i] = self.marginal_prob[t + 1, i]

            thresh = max(thresh, dist.pdf(mode))

        if support_thresh is not None:
            thresh = support_thresh
        else:
            # HACK
            thresh /= 1000

        plot_mixture(dists, weights,
                     hi=self.mu_mode[:, :, ix].max(),
                     lo=self.mu_mode[:, :, ix].min(),
                     support_thresh=thresh)

    def plot_forc_density(self, t, support_thresh=None):
        """
        Plot mixture components of one-step forecast density

        Parameters
        ----------
        t : int
        support_thresh : float or None

        Notes
        -----
        cf. Eq. 12.38 in W&H
        """
        dists = {}
        weights = {}
        thresh = 0

        for jt in range(self.nmodels):
            for jtp in range(self.nmodels):

                mode = self.forecast[t, jtp]
                scale = np.sqrt(self.forc_var[t - 1, jt, jtp])

                dist = stats.t(self.df[t], loc=mode, scale=scale)

                dists[jt, jtp] = dist
                thresh = max(thresh, dist.pdf(mode))

        if support_thresh is not None:
            thresh = support_thresh
        else:
            thresh /= 100

        # pi(j_t) p_{t-1}(j_{t-1})
        weights = np.outer(self.prior_model_prob,
                           self.marginal_prob[t - 1])

        plot_mixture(dists, weights,
                     hi=self.forecast[t].max(),
                     lo=self.forecast[t].min(),
                     support_thresh=thresh)

    def plot_post_prob(self):
        _, axes = plt.subplots(nrows=self.nmodels, sharex=True,
                               sharey=True)

        rng = np.arange(len(self.marginal_prob))
        for i in range(self.nmodels):
            ax = axes[i]
            ax.bar(rng, self.marginal_prob[:, i], color='k', width=0.5)

    def plot_forecast(self, pause=False):
        # Figure 12.5

        fig = plt.figure()

        ax = fig.add_subplot(111)
        forcs = self.forecast[1:]
        probs = self.marginal_prob[:-1]

        if not pause:
            ax.plot(self.dates, self.y, 'k')
        else:
            ax.plot(self.dates, self.y, 'w')

        fig.autofmt_xdate()

        for i, d in enumerate(self.dates):
            for j in range(self.nmodels):
                ax.plot([d], forcs[i, j], 'ko', ms=probs[i, j] * 6)

            if pause:
                ax.plot(self.dates[:i+1], self.y[:i+1], 'k')
                plt.draw_if_interactive()
                raw_input('Press a key to continue')

        hlines = [11, 24, 36]
        for h in hlines:
            ax.axvline(self.dates[h], color='k')

class Model(object):
    """
    Just a dummy for doing Section 12.4, for now
    """
    def __init__(self, G, deltas, obs_var_mult=1.):
        self.G = G
        self.deltas = np.asarray(deltas)
        self.obs_var_mult = obs_var_mult

    def get_Wt(self, Cprior):
        # Eq. 12.18
        disc_var = np.diag(Cprior) * (1 / self.deltas - 1)
        return chain_dot(self.G, np.diag(disc_var), self.G.T)
