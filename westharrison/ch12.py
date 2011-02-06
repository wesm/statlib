from scipy.linalg import block_diag

import numpy as np
import matplotlib.pyplot as plt

import statlib.dlm as dlm
import statlib.plotting as plotting

reload(dlm)
from statlib.dlm import *
import datasets

from pandas import DataMatrix

y = datasets.table_22()
x = [[1]]
mean_prior = (0, 1)
var_prior = (1, 0.01)

class DLMMixture(object):
    """


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

        Notes
        -----
        cf. West & Harrison Figure 12.3. Automatically annotating individual
        component curves would probably be difficult.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ix = index

        dists = {}
        for name in self.names:
            model = self.models[name]
            df = model.df[t]
            mode = model.mu_mode[t + 1, ix]
            scale = np.sqrt(model.mu_scale[t + 1, ix, ix])
            dists[name] = stats.t(df, loc=mode, scale=scale)

        def mix_pdf(x):
            tot = 0
            weights = self.get_weights(t)
            for name, dist in dists.iteritems():
                tot += weights[name] * dist.pdf(x)

            return tot

        # plot mixture
        mix = plotting.plot_f_support(mix_pdf, -1, 1,
                                      thresh=support_thresh,
                                      style='k',
                                      ax=ax)

        for name in self.names:
            comp = plotting.plot_f_support(dists[name].pdf, -1, 1,
                                           thresh=support_thresh,
                                           style='k--',
                                           ax=ax)

        ax.legend((mix, comp), ('Mixture', 'Component'))

discounts = np.arange(0.7, 1.01, 0.1)

models = {}
for delta in discounts:
    models['%.2f' % delta] = ConstantDLM(y, x, mean_prior=mean_prior,
                                         var_prior=var_prior, discount=delta)

mix = DLMMixture(models)
