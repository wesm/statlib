import numpy as np
import matplotlib.pyplot as plt

import statlib.dlm as dlm
reload(dlm)
from statlib.dlm import *
import datasets

gas = datasets.table_81()

p = 12
var_prior = (1, 0.05)
model = Polynomial(1) + FullEffectsFourier(12, harmonics=[1, 4])
model = Polynomial(1) + FullEffectsFourier(12)
k = model.F.shape[1]
mean_prior = (np.zeros(k), np.eye(k))
dlm = ConstantDLM(gas, model.F, G=model.G, mean_prior=mean_prior,
                  var_prior=var_prior, discount=.95)
