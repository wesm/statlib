from datetime import datetime
import numpy as np
import re
import pandas as pn
import scipy.stats as stats

import statlib.dlm as dlm
reload(dlm)
from statlib.dlm import *

def load_table():
    path = 'westharrison/data/Table2.2.data.txt'
    sep = '\s+'

    lines = [re.split(sep, l.strip()) for l in open(path)]

    data = []
    for line in lines:
        # drop year
        data.extend(line[1:])

    data = np.array(data, dtype=float) / 100
    dates = pn.DateRange(datetime(1975, 1, 1), periods=len(data),
                         timeRule='EOM')

    return pn.Series(data, index=dates)

def linear_dlm(y):
    pass

def ex_21():
    data = load_table()

    mean_prior = (0, 1)
    var_prior = (1, 0.01)

    discount_factors = np.arange(0.6, 1.01, 0.05)

    fig = plt.figure(figsize=(12, 12))

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    rng = data.index

    ax1.plot(rng, np.asarray(data), 'b')

    rmse = []
    mad = []
    like = []

    for disc in discount_factors:
        model = dlm.DLM(data, mean_prior=mean_prior,
                        var_prior=var_prior, discount=disc)

        ax1.plot(rng, model.forecast, '%.2f' % (1 - disc))

        rmse.append(model.rmse)
        mad.append(model.mad)
        like.append(model.pred_like)

    like = np.array(like).prod(axis=1)
    llr = np.log(like / like[-1])

    ax2.plot(discount_factors, rmse, label='RMSE')
    ax2.plot(discount_factors, mad, label='MAD')
    ax3.plot(discount_factors, llr, label='LLR')

    ax2.legend()
    ax3.legend()


if __name__ == '__main__':
    data = load_table()

    mean_prior = (0, 1)
    var_prior = (1, 0.01)

    model3 = dlm.DLM(data, mean_prior=mean_prior, var_prior=var_prior,
                    discount=0.1)

    model = dlm.DLM(data, mean_prior=mean_prior, var_prior=var_prior,
                    discount=1.)
    model2 = dlm.DLM(data, mean_prior=mean_prior, var_prior=var_prior,
                    discount=0.9)

