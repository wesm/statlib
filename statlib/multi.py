import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from statlib.tools import nan_array, chain_dot
from statlib.dlm import st
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

        self._plot_mixture(dists, self.get_weights(t),
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

        self._plot_mixture(dists, self.get_weights(t),
                           support_thresh=support_thresh)

    def _plot_mixture(self, dists, weights, support_thresh=0.1):
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
        mix = plotting.plot_support(mix_pdf, -1, 1,
                                    thresh=support_thresh,
                                    style='k',
                                    ax=ax)

        for name in self.names:
            comp = plotting.plot_support(dists[name].pdf, -1, 1,
                                         thresh=support_thresh,
                                         style='k--',
                                         ax=ax)

        ax.legend((mix, comp), ('Mixture', 'Component'))

class ApproximateMixture(object):
    """
    Class II type DLM mixture model. A model is chosen at each time point

    Parameters
    ----------
    models : dict

    Notes
    -----
    cf. W&H Section 12.2
    """
    def __init__(self):
        pass

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
        self.model_prob = nan_array(self.nobs + 1, self.nmodels)

        self.mu_mode = nan_array(self.nobs + 1, self.nmodels, self.ndim)
        self.mu_forc_mode = nan_array(self.nobs + 1, self.nmodels, self.ndim)
        self.mu_scale = nan_array(self.nobs + 1, self.nmodels,
                                  self.ndim, self.ndim)
        self.mu_forc_var = nan_array(self.nobs + 1, self.nmodels, self.nmodels,
                                     self.ndim, self.ndim)

        self.forecast = nan_array(self.nobs + 1, self.nmodels)

        # relative multiplier!
        self.var_est = nan_array(self.nobs + 1, self.nmodels)

        # forecasts are computed via mixture for now
        self.forc_var = nan_array(self.nobs, self.nmodels, self.nmodels)

        n, d = var_prior

        # set in initial values
        self.model_prob[0] = self.prior_model_prob
        self.mu_mode[0], self.mu_scale[0] = mean_prior
        self.var_est[0] = d / n
        self.df = n + np.arange(self.nobs + 1) # precompute

        self._compute_parameters()

    def _compute_parameters(self):
        """
        Compute parameter estimates for Gaussian Univariate DLM

        Parameters
        ----------

        Notes
        -----
        West & Harrison pp. 111-112

        Returns
        -------

        """
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
            St = self._update_St(t, Qt, errs)
            Ct = self._update_Ct(t, Rt, Qt, At, St)

            # update posterior model probabilities

            # p_t(j_t, j_t-1)
            post_prob = self._update_postprob(t, errs, Qt)
            # p_t(j_t)
            marginal_post = post_prob.sum(axis=1)

            # collapse variance estimates
            coll_St = self._collapse_var(St, post_prob, marginal_post)

            # select one j_t-1
            pstar = (coll_St / St[:, 0]) * (post_prob[:, 0] / marginal_post)

            # collapse posterior means / scales
            coll_m, coll_C = self._collapse_params(pstar, mt, Ct)

            # store results
            self.mu_mode[t] = coll_m
            self.mu_scale[t] = coll_C
            self.var_est[t] = coll_St
            self.forc_var[t-1] = Qt

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

    def _update_St(self, t, Qt, errs):
        def calc_update(jt, jtp):
            prior = self.var_est[t - 1, jtp]
            Q = Qt[jt, jtp]
            e = errs[jt, jtp]
            return prior + (prior / self.df[t]) * ((e ** 2) / Q - 1)

        return self._fill_updates(calc_update, ())

    def _update_Ct(self, t, Rt, Qt, At, St):
        def calc_update(jt, jtp):
            R = Rt[jt, jtp]
            A = At[jt, jtp]
            Q = Qt[jt, jtp]
            S = St[jt, jtp]
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

        # rQ = np.sqrt(Qt)
        # pi = self.prior_model_prob
        # prior = self.model_prob[t - 1]
        # result = pi * prior * dist.pdf(errs / rQ) / rQ

        def calc_update(jt, jtp):
            rQ = np.sqrt(Qt[jt, jtp])

            # W&H eq. 12.40
            pi = self.prior_model_prob[jt]
            prior = self.model_prob[t - 1, jtp]
            return pi * prior * dist.pdf(errs[jt, jtp] / rQ) / rQ

        result = self._fill_updates(calc_update, ())

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
                m += pstar[jtp] * mt[jt, jtp]

            coll_m[jt] = m

            # collapse scales
            for jtp in range(self.nmodels):
                mdev = coll_m[jt] - mt[jt, jtp]
                C += pstar[jtp] * (Ct[jt, jtp] + np.outer(mdev, mdev))

            coll_C[jt] = C

        return coll_m, coll_C

    def _get_Ft(self, t):
        return self.F[0:1].T

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

if __name__ == '__main__':
    cp6 = datasets.table_111()

    E2 = [[1, 0]]

    G = tools.jordan_form(2)

    order = ['standard', 'outlier', 'level', 'growth']
    prior_model_prob = {'standard' : 0.85,
                        'outlier' : 0.07,
                        'level' : 0.05,
                        'growth' : 0.03}

    models = {}

    models['standard'] = Model(G, [0.9, 0.9], obs_var_mult=1.)
    models['outlier'] = Model(G, [0.9, 0.9], obs_var_mult=100.)
    models['level'] = Model(G, [0.01, 0.9], obs_var_mult=1.)
    models['growth'] = Model(G, [0.9, 0.01], obs_var_mult=1.)

    # priors
    m0 = np.array([600., 10.])
    C0 = np.diag([10000., 25.])
    n0, d0 = 10, 1440

    multi = MultiProcessDLM(cp6, E2, models, order,
                            prior_model_prob,
                            mean_prior=(m0, C0),
                            var_prior=(n0, d0),
                            approx_steps=1)


