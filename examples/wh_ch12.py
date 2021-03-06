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
import statlib.tools as tools

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

    # order = ['standard', 'outlier']
    # prior_model_prob = {'standard' : 0.90,
    #                     'outlier' : 0.10}

    models = {}

    standard = Model(G, [0.9, 0.9], obs_var_mult=1.)
    models['standard'] = standard
    models['outlier'] = Model(G, [0.9, 0.9], obs_var_mult=100.)
    models['level'] = Model(G, [0.01, 0.9], obs_var_mult=1.)
    models['growth'] = Model(G, [0.9, 0.01], obs_var_mult=1.)

    # priors
    m0 = np.array([600., 10.])
    C0 = np.diag([10000., 25.])
    n0, d0 = 10., 1440.

    multi = MultiProcessDLM(cp6, E2, models, order,
                            prior_model_prob,
                            m0=m0, C0=C0, n0=n0, s0=d0 / n0,
                            approx_steps=1)

    return multi

if __name__ == '__main__':

    y = datasets.table_22()
    x = [[1]]
    m0, C0 = (0, 1)
    n0, s0 = (1, 0.01)
    discounts = np.arange(0.7, 1.01, 0.1)

    models = {}
    for delta in discounts:
        models['%.2f' % delta] = DLM(y, x, m0=m0, C0=C0, n0=n0, s0=s0,
                                     state_discount=delta)

    # G = tools.jordan_form(2)
    # cp6 = datasets.table_111()
    # x = [1, 0]
    # m0, C0 = ([0, 0.45], [[0.005, 0],
    #                       [0, 0.0025]])
    # n0, s0 = (1, 1)
    # std_dlm = DLM(cp6.values, x, G=G, m0=m0, C0=C0, n0=n0, s0=s0,
    #               state_discount=0.9)

    mix = DLMMixture(models)
    multi = get_multi_model()
