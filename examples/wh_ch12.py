from scipy.linalg import block_diag

import numpy as np
import matplotlib.pyplot as plt

import statlib.dlm as dlm
import statlib.multi as multi
import statlib.plotting as plotting

reload(dlm)
reload(multi)
from statlib.dlm import *
from statlib.multi import *
import statlib.datasets as datasets

from pandas import DataMatrix

def get_multi_model():
    cp6 = datasets.table_111()

    E2 = [[1, 0]]

    G = tools.jordan_form(2)

    order = ['standard', 'outlier', 'level', 'growth']
    prior_model_prob = {'standard' : 0.85,
                        'outlier' : 0.07,
                        'level' : 0.05,
                        'growth' : 0.03}

    models = {}

    standard = Model(G, [0.9, 0.9], obs_var_mult=1.)
    models['standard'] = standard
    models['outlier'] = Model(G, [0.9, 0.9], obs_var_mult=100.)
    models['level'] = Model(G, [0.01, 0.9], obs_var_mult=1.)
    models['growth'] = Model(G, [0.9, 0.01], obs_var_mult=1.)

    # priors
    m0 = np.array([600., 10.])
    C0 = np.diag([10000., 25.])
    n0, d0 = 10, 1440

    mean_prior = (m0, C0)
    var_prior = (n0, d0)

    multi = MultiProcessDLM(cp6, E2, models, order,
                            prior_model_prob,
                            mean_prior=mean_prior,
                            var_prior=var_prior,
                            approx_steps=1)

    return multi

if __name__ == '__main__':

    y = datasets.table_22()
    x = [[1]]
    mean_prior = (0, 1)
    var_prior = (1, 0.01)
    discounts = np.arange(0.7, 1.01, 0.1)

    models = {}
    for delta in discounts:
        models['%.2f' % delta] = DLM(y, x, mean_prior=mean_prior,
                                     var_prior=var_prior,
                                     discount=delta)

    mix = DLMMixture(models)
    multi = get_multi_model()
