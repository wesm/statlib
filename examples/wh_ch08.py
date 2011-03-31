import numpy as np
import matplotlib.pyplot as plt

import statlib.dlm as dlm
reload(dlm)
from statlib.dlm import *
from statlib.components import *
import statlib.datasets as datasets

gas = datasets.table_81()

p = 12
model = Polynomial(2) + FullEffectsFourier(12, harmonics=[1, 4])
# model = Polynomial(1) + FullEffectsFourier(12)
k = model.F.shape[1]
m0, C0 = (np.zeros(k), np.eye(k))
n0, s0 = (1, 0.05)
dlm = DLM(gas, model.F, G=model.G,
          m0=m0, C0=C0, n0=n0, s0=s0,
          state_discount=.95,
          var_discount=1)
