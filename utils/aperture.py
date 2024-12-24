#!usr/env/python3

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def aperture(data, thresh, period=None):
    """
    Формирует апертурный сигнал из исходного с заданными порогововым значением и периодичностью записи
    
    :param data: исходный сигнал
    :param thresh: пороговое значение
    :param period: периодичность записи
    """
    
    result = np.zeros(data.shape)
    result.fill(np.nan)
    result[0] = data[0]
    result[-1] = data[-1]
    
    if not period:
        period = len(data) + 1

    base = data[0]
    for i in range(1, len(data)):
        if np.abs(data[i] - base) >= thresh or i % period == 0:
            result[i] = data[i]
            base = data[i]

    return result
