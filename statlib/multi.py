import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from statlib.tools import nan_array, chain_dot
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

        self.names = order
        self.models = [models[name] for name in order]
        self.prior_model_prob = np.array([prior_model_prob[name]
                                          for name in order])
        self.approx_steps = int(approx_steps)

        self.mean_prior = mean_prior
        self.var_prior = var_prior

        self.nmodels = len(models)
        self.ndim = len(mean_prior[0])

        self.mu_mode = nan_array(self.nobs + 1, self.ndim)
        self.mu_forc_mode = nan_array(self.nobs + 1, self.ndim)
        self.mu_scale = nan_array(self.nobs + 1, self.ndim, self.ndim)
        self.var_est = nan_array(self.nobs + 1)
        self.forc_var = nan_array(self.nobs)
        self.R = nan_array(self.nobs + 1, self.ndim, self.ndim)

        self.mu_mode[0], self.mu_scale[0] = mean_prior
        n, d = var_prior
        self.var_est[0] = d / n
        self.df = n + np.arange(self.nobs + 1) # why not precompute

        # self._compute_parameters()

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
        # allocate result arrays
        mode = self.mu_mode
        C = self.mu_scale
        df = self.df
        S = self.var_est
        G = self.G

        for i, obs in enumerate(self.y):
            # column vector, for W&H notational consistency
            Ft = self._get_Ft(i)

            # advance index: y_1 through y_nobs, 0 is prior
            t = i + 1

            self._update_at(t)
            self._update_Rt(t)
            self._update_ft(t)
            self._update_Qt(t)

            for jt in range(self.nmodels):
                for jtp in range(self.nmodels):
                    pass

            # forecast theta as time t
            f_t = np.dot(Ft.T, a_t)
            err = obs - f_t

            # update mean parameters


            self.mu_forc_mode[t] = a_t
            self.forc_var[t-1] = Qt
            self.R[t] = Rt

    def _update_at(self, t):
        if t > 1:
            a_t = np.dot(G, mode[t - 1])
        else:
            a_t = mode[0]

    def _update_Rt(self, t):
        if t > 1:
            # only discount after first time step! hmm
            Rt = chain_dot(G, C[t - 1], G.T) / self.disc
        else:
            Rt = C[0]

    def _compute_Rt(self, t, jt, jtp):
        self.mu_scale

    def _update_Qt(self, t):
        Qt = chain_dot(Ft.T, Rt, Ft) + S[t-1]

    def _update_At(self, t):
        At = np.dot(Rt, Ft) / Qt

    def _update_mt(self, t):
        mode[t] = a_t + np.dot(At, err)

    def _update_St(self, t):
        S[t] = S[t-1] + (S[t-1] / df[t]) * ((err ** 2) / Qt - 1)

    def _update_Ct(self, t):
        C[t] = (S[t] / S[t-1]) * (Rt - np.dot(At, At.T) * Qt)

    def _get_Ft(self, t):
        return self.F[0:1].T

class Model(object):
    """
    Just a dummy for doing Section 12.4, for now
    """
    def __init__(self, G, deltas, obs_var_mult=1.):
        self.G = G
        self.deltas = np.asarray(deltas)

    def get_Wt(self, Cprior):
        # Eq. 12.18
        disc_var = np.diag(Cprior) * (1 / self.deltas - 1)
        return chain_dot(self.G, np.diag(disc_var), self.G.T)

if __name__ == '__main__':
    cp6 = datasets.table_111()
    F = [[1]]

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

    multi = MultiProcessDLM(cp6, F, models, order,
                            prior_model_prob,
                            mean_prior=(m0, C0),
                            var_prior=(n0, d0),
                            approx_steps=1)


