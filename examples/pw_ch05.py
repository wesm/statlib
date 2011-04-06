import numpy as np

import statlib.tvar as tvar
reload(tvar)
from statlib.tvar import TVAR
import statlib.datasets as ds

eeg = ds.eeg_Cz()

p = 12.
m0 = np.zeros(p)
C0 = np.eye(p)
n0 = 2
s0 = 50

model = TVAR(eeg, p=p, m0=m0, C0=C0, n0=n0, s0=s0,
             state_discount=0.994,
             var_discount=0.95)
decomp = model.decomp()

# result = tvar_gridsearch(model, range(12, 13),
#                          np.linspace(0.9, 1, num=10),
#                          np.linspace(0.95, 0.95, num=1))
