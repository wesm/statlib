
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt

import statlib.tvar as tvar
import statlib.datasets as ds
import statlib.plotting as plotting

import scikits.statsmodels.tsa.api as tsa

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

def acf_pacf_plot(series, lags=50, figsize=(10,10)):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize)

    plotting.plot_acf(series, lags, ax=axes[0])
    axes[0].set_title('Autocorrelation (ACF)')

    plotting.plot_acf(series, lags, ax=axes[1], partial=True)
    axes[1].set_title('Partial autocorrelation (PACF)')


p = 12.
state_discount = 0.994
var_discount = 1

m0 = np.zeros(p)
C0 = np.eye(p)
n0 = 2
s0 = 50

def fit_tvars(data):

    models = {}
    decomps = {}

    for chan, series in data.iteritems():
        print chan

        model = tvar.TVAR(series, p=p, m0=m0, C0=C0, n0=n0, s0=s0,
                          state_discount=state_discount,
                          var_discount=var_discount)
        models[chan] = model
        decomps[chan] = decomp = model.decomp()

    freqs = {}
    mods = {}
    for chan, decomp in decomps.iteritems():
        freqs[chan] = decomp['frequency'][0]
        mods[chan] = decomp['modulus'].max(1)

    freqs = pn.DataMatrix(freqs)
    mods = pn.DataMatrix(mods)

    return freqs, mods

def marglike_plot(series, prange=[12], vrange=[1], trange=[1]):
    maxp = max(prange)
    m0 = np.zeros(maxp)
    C0 = np.eye(maxp)
    n0 = 2
    s0 = 50

    model = tvar.TVAR(series, p=maxp, m0=m0, C0=C0, n0=n0, s0=s0,
                      state_discount=state_discount,
                      var_discount=var_discount)

    return tvar.tvar_gridsearch(model, prange, trange, vrange).squeeze()

eegdata = pn.DataMatrix.load(eeg_path)
Cz = eegdata['Cz'].values

def decomp_plot(model, figsize=(12, 6)):
    decomp = model.decomp()

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    min_freq = 2 * np.pi * decomp['frequency'].min(1)
    axes[0].plot(min_freq.values)
    axes[0].set_title('Frequency')
    axes[0].set_ylim([1, 5])

    max_mod = decomp['modulus'].max(1)
    axes[1].plot(max_mod.values)
    axes[1].set_title('Modulus')
    axes[1].set_ylim([0.7, 1.2])

model = tvar.TVAR(Cz, p=p, m0=m0, C0=C0, n0=n0, s0=s0,
                  state_discount=state_discount,
                  var_discount=var_discount)

# freqs, mods = fit_tvars(eegdata)
# freqs.save('freqs')
# mods.save('mods')

freqs = pn.DataMatrix.load('freqs')
mods = pn.DataMatrix.load('mods')

'''
eegdata.corr()['Cz']
plot(Cz)
acf_pacf_plot(Cz)
marglike_plot(Cz, range(40))
'''

'''
vrange = np.linspace(0.8, 1, 20)
trange = np.linspace(0.97, 1, 30)
prange = range(40)
# marglik = marglike_plot(Cz, prange=prange)
# marglik = marglike_plot(Cz, vrange=vrange)
marglik = marglike_plot(Cz, trange=trange)

plt.figure()
ax = plt.gca()
# ax.plot(prange, marglik - marglik.max())
# ax.plot(vrange, marglik - marglik.max())
ax.plot(trange, marglik - marglik.max())
plt.show()
'''
