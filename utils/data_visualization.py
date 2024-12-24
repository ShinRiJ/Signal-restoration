#!usr/bin/env/python3


import numpy as np
import matplotlib.pyplot as plt


def plot_series(time, series, format_="-", start=0, end=None):  
    """
    Отображает визуализацию графика
    
    :param time: координаты времени
    :param series: временной ряд
    :param format_: формат визуализации данных
    :param start: начальная точка визуализации
    :param end: конечная точка визуализации
    """
    
    plt.figure(figsize=(10, 6))    
    if type(series) is tuple:
      for series_num in series:
        plt.plot(time[start:end], series_num[start:end], format_)
    else:
      plt.plot(time[start:end], series[start:end], format_)
    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.grid(True)
    plt.show()
    