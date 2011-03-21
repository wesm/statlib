import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from statlib.tools import quantile

import statlib.dlm as dlm
reload(dlm)
from statlib.dlm import DLM, st

import statlib.datasets as datasets

def ex_21():
    y = datasets.table_22()
    x = np.ones(len(y), dtype=float)

    mean_prior = (0, 1)
    var_prior = (1, 0.01)

    discount_factors = np.arange(0.6, 1.01, 0.05)

    fig = plt.figure(figsize=(12, 12))

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    rng = y.index

    ax1.plot(rng, np.asarray(y), 'b')

    rmse = []
    mad = []
    like = []

    y_pred = []
    V = []
    mu = []

    for disc in discount_factors:
        model = dlm.DLM(y, x, mean_prior=mean_prior,
                        var_prior=var_prior, discount=disc)

        ax1.plot(rng, model.forecast, '%.2f' % (1 - disc))

        rmse.append(model.rmse)
        mad.append(model.mad)
        like.append(model.pred_like)

        # level and error bars for y_116, mu_115, V

        # mu_115
        level, lower, upper = model.mu_ci(prior=False)
        mu.append((level[-1], lower[-1], upper[-1]))

        # var_est final
        phi_a = model.df[-1] / 2.
        phi_b = model.var_est[-1] * phi_a
        sampler = lambda sz: 1 / np.random.gamma(phi_a, scale=1./phi_b, size=sz)
        lower, upper = boot_ci(sampler)
        V.append((model.var_est[-1], lower, upper))

        # y_116
        dist = stats.t(model.df[-1], loc=model.mu_mode[-1, 0],
                       scale=model.mu_scale[-1, 0] / disc + model.var_est[-1])
        lower, upper = boot_ci(dist.rvs)
        y_pred.append((model.mu_mode[-1], lower, upper))

    like = np.array(like).prod(axis=1)
    llr = np.log(like / like[-1])

    ax2.plot(discount_factors, rmse, label='RMSE')
    ax2.plot(discount_factors, mad, label='MAD')
    ax3.plot(discount_factors, llr, label='LLR')

    ax2.legend()
    ax3.legend()

    fig = plt.figure(figsize=(12, 12))

    ax1 = fig.add_subplot(311)
    level, lower, upper = zip(*mu)
    ax1.plot(discount_factors, level, 'k', label=r'$\mu_{115}$')
    ax1.plot(discount_factors, lower, 'k--')
    ax1.plot(discount_factors, upper, 'k--')

    ax1.legend()

    ax2 = fig.add_subplot(312)
    level, lower, upper = zip(*V)
    ax2.plot(discount_factors, level, 'k', label=r'$V$')
    ax2.plot(discount_factors, lower, 'k--')
    ax2.plot(discount_factors, upper, 'k--')

    ax2.legend()

    ax3 = fig.add_subplot(313)
    level, lower, upper = zip(*y_pred)

    ax3.plot(discount_factors, level, 'k', label=r'$y_{116}$')
    ax3.plot(discount_factors, lower, 'k--')
    ax3.plot(discount_factors, upper, 'k--')
    ax3.legend()

def boot_ci(sample, samples=10000):
    draws = sample(samples)
    return quantile(draws, [0.05, 0.95])

if __name__ == '__main__':
    y = datasets.table_22()
    x = [1]

    mean_prior = (0, 1)
    var_prior = (1, 0.01)

    model3 = DLM(y, x, mean_prior=mean_prior, var_prior=var_prior,
                 discount=0.1)

    model = DLM(y, x, mean_prior=mean_prior, var_prior=var_prior,
                discount=.9)
    model2 = DLM(y, x, mean_prior=mean_prior, var_prior=var_prior,
                 discount=0.9)

