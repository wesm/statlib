from scipy.linalg import block_diag

import numpy as np
import matplotlib.pyplot as plt

import statlib.dlm as dlm
reload(dlm)
from statlib.dlm import *
import datasets

data = datasets.table_102()

index = data['INDEX']
sales = data['SALES']

# p = 12
var_prior = (6, 6 * 0.15 ** 2)

phi_effects = np.array([0.783, 1.283, 0.983, # seasonal effects
                        1.083, 1.183, -0.017,
                        -1.217, -1.017, -1.017,
                        -1.517, -0.517, -0.017])
phi_scale = np.empty((12, 12))
phi_scale.fill(-0.0034)
phi_scale[np.diag_indices(12)] = 0.0367

seasonal_comp = FullEffectsFourier(12, harmonics=[1, 3, 4],
                                   discount=0.95)
# seasonal_comp = FullEffectsFourier(12, discount=0.95)

model = (Polynomial(2, discount=0.9) +
         Regression(index, discount=0.98) +
         seasonal_comp)

# transformation matrix
H = seasonal_comp.H
seasonal_prior_mean = np.dot(H, phi_effects)
seasonal_prior_scale = zero_out(chain_dot(H, phi_scale, H.T))

prior_mean = np.array(np.concatenate(([9.5, 1., -0.7],
                                      seasonal_prior_mean)))

prior_scale = block_diag(np.diag([0.09, 0.09, 0.01]),
                         seasonal_prior_scale)

mean_prior = (prior_mean, prior_scale)

# k = model.F.shape[1]

dlm = DLM(sales, model.F, G=model.G,
          mean_prior=mean_prior,
          var_prior=var_prior,
          discount=model.discount)
