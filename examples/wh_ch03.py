import numpy as np
import matplotlib.pyplot as plt
import statlib.dlm as dlm
reload(dlm)
from statlib.dlm import DLM

import statlib.datasets as datasets

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
        model = dlm.DLM(y, x, m0=m0, C0=C0, n0=n0, s0=s0,
                        state_discount=disc)

        rmse.append(model.rmse)
        mad.append(model.mad)
        like.append(model.pred_like)

        var_est.append(model.var_est[-1])
        # model.var_est[-1] + model.mu_scale[-1] / disc
        pred_var.append(model.forc_var[-1])

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

    m0, C0 = (0.45, 0.0025)
    n0, s0 = (1, 1)

    model = dlm.DLM(y, x, m0=m0, C0=C0, n0=n0, s0=s0,
                    state_discount=0.6)
