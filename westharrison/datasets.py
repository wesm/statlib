from datetime import datetime
import re

import numpy as np
import pandas as pn

def table_22():
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

def table_33():
    path = 'westharrison/data/Table3.3.data.txt'
    sep = '\s+'

    lines = [re.split(sep, l.strip()) for l in open(path)]

    y_data = []
    f_data = []
    saw_f = False
    for line in lines:
        if line[0] == 'Y':
            continue
        elif line[0] == 'F':
            saw_f = True
            continue

        # drop year
        if saw_f:
            f_data.extend(line[1:])
        else:
            y_data.extend(line[1:])

    y_data = np.array(y_data, dtype=float)
    f_data = np.array(f_data, dtype=float)

    dates = pn.DateRange(datetime(1975, 1, 1), periods=len(y_data),
                         timeRule='Q@MAR')

    Y = pn.Series(y_data, index=dates)
    F = pn.Series(f_data, index=dates)

    return Y, F
