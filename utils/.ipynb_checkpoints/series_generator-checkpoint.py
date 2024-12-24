#!usr/env/python3

import numpy as np


def noise(time, noise_level=1, seed=None):
    """
    Генерирует шум для временного ряда
    
    :param time: координаты времени исходного временного ряда
    :param noise_level: уровень шума
    :param seed: исходные начальные условия
    """
    
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level
