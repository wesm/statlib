import numpy as np
import matplotlib.pyplot as plt

from statlib.dlm import DLM
import datasets

if __name__ == '__main__':
    y, x = datasets.table_33()

    X = np.vstack((np.ones(len(x)),x )).T

    mean_prior = ([0, 0.45], [[0.0025, 0],
                              [0, 0.0025]])
    var_prior = (1, 1)

    mean_prior = (0.45, 0.0025)
    var_prior = (1, 1)

    model = DLM(y, x, mean_prior=mean_prior,
                var_prior=var_prior, discount=0.7)
