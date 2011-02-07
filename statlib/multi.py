import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import statlib.plotting as plotting

from statlib.dlm import ConstantDLM
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

if __name__ == '__main__':
    cp6 = datasets.table_111()

    W_matrices = {}
    # construct evolution matrices
    w_standard = np.array([[0.11, 0.01],
                           [0.01, 0.01]])
    w_outlier = np.array([[0.11, 0.01],
                          [0.01, 0.01]])

    w_level_chg = np.array([[10.01, 0.01],
                          [0.01, 0.01]])

    w_growth_chg = np.array([[1.1, 1],
                             [1, 1]])


