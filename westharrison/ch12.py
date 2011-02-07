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

y = datasets.table_22()
x = [[1]]
mean_prior = (0, 1)
var_prior = (1, 0.01)
discounts = np.arange(0.7, 1.01, 0.1)

models = {}
for delta in discounts:
    models['%.2f' % delta] = ConstantDLM(y, x, mean_prior=mean_prior,
                                         var_prior=var_prior, discount=delta)

mix = DLMMixture(models)
