import numpy as np
import statlib.plotting as plotting
from scikits.statsmodels.tsa.arima import ARMA as sm_arma

class ARMA(object):
    pass


if __name__ == '__main__':
    import statlib.datasets as ds
    eeg = ds.eeg_data()

    model = sm_arma(eeg)
    res = model.fit(order=(8, 0), method='css', disp=-1)
