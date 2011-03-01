import numpy as np
import matplotlib.pyplot as plt
import statlib.dlm as dlm
reload(dlm)
from statlib.dlm import DLM

import datasets

def ex_310():
    y, x = datasets.table_33()

    discount_factors = np.arange(0.60, 1.01, 0.05)

    rows, columns = 3, 3

    rmse = []
    mad = []
    like = []
    var_est = []
    pred_var = []

    fig, axes = plt.subplots(3, 3, sharey=True, figsize=(12, 12))

    for i, disc in enumerate(discount_factors):
        model = dlm.DLM(y, x, mean_prior=mean_prior,
                        var_prior=var_prior, discount=disc)

        rmse.append(model.rmse)
        mad.append(model.mad)
        like.append(model.pred_like)

        var_est.append(model.var_est[-1])
        pred_var.append(model.forc_var[-1]) # model.var_est[-1] + model.mu_scale[-1] / disc

        ax = axes[i / rows][i % rows]

        # plot posterior
        model.plot_mu(prior=False, ax=ax)

        ax.set_title(str(disc))
        ax.set_ylim([0.4, 0.48])

    like = np.array(like).prod(axis=1)
    llr = np.log(like / like[-1])

    fig = plt.figure(figsize=(12, 12))

    ax1 = fig.add_subplot(311)
    ax1.plot(discount_factors, rmse, label='RMSE')
    ax1.plot(discount_factors, mad, label='MAD')
    ax2 = fig.add_subplot(312)
    ax2.plot(discount_factors, llr, label='LLR')
    ax1.legend()
    ax2.legend()

    # plot s_42 and q_42
    ax3 = fig.add_subplot(313)
    ax3.plot(discount_factors, var_est, label='S')
    ax3.plot(discount_factors, pred_var, label='Q')
    ax3.legend()

if __name__ == '__main__':
    y, x = datasets.table_33()

    mean_prior = (0.45, 0.0025)
    var_prior = (1, 1)

    model = DLM(y, x, mean_prior=mean_prior,
                var_prior=var_prior, discount=0.7)
