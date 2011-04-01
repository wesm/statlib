
import numpy as np
import pandas as pn

from statlib.tvar import TVAR
import statlib.datasets as ds

# Cz = ds.eeg_Cz()

eeg_path = 'eeg19subsampled'

def make_subsampled_dataset():
    channames = np.loadtxt('/home/wesm/research/mike/eegdata/channames',
                           dtype=object)
    data = np.loadtxt('/home/wesm/research/mike/eegdata/eeg19_data.dat')

    # subsample to match P&W datasets
    data = data[1999:][::5][:3600]
    dm = pn.DataMatrix(data, columns=channames)

    dm.save(eeg_path)

eegdata = pn.DataMatrix.load(eeg_path)

p = 12.
state_discount = 0.994
var_discount = 1

m0 = np.zeros(p)
C0 = np.eye(p)
n0 = 2
s0 = 50

# fit a bunch of univariate TVAR models

models = {}
decomps = {}

for chan, series in eegdata.iteritems():
    print chan

    model = TVAR(series, p=p, m0=m0, C0=C0, n0=n0, s0=s0,
                 state_discount=state_discount,
                 var_discount=var_discount)
    models[chan] = model
    decomps[chan] = decomp = model.decomp()

freqs = {}
mods = {}
for chan, decomp in decomps.iteritems():
    freqs[chan] = decomp['frequency'][0]
    mods[chan] = decomp['modulus'].max(1)

freqs = DataMatrix(freqs)
mods = DataMatrix(mods)
