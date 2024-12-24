#!usr/bin/env/python3

import numpy as np
import tensorflow as tf


def calculate_window_size(n_pred, n_regr):
    n_pred /= 2
    return int(n_regr + (2 * n_pred) + (2 * n_regr * (n_pred - 1)))


def get_predictor_indexes(w_size, n_preds):
    """
    Возвращает индексы предикторов в оконной функции
    
    :param w_size: размер окна
    :param n_prds: количество предикторов
    """
    
    return np.linspace(0, w_size - 1, num=n_preds, endpoint=True, dtype=np.uint)


def get_regressor_indexes(w_size, n_regr):
    """
    Возвращает индексы регрессоров в оконной функции
    
    :param w_size: размер окна
    :param n_regr: количество регрессоров
    """
        
    wnd = np.zeros((w_size, 1), dtype=bool)
    wnd[::n_regr + 1] = 1
    mask = np.where(np.any(wnd, axis=1))[0]
    middle = mask.shape[0] // 2
    return np.array(range(mask[middle - 1] + 1, mask[middle]))


def windowed_dataset_multi(series, window_size, batch_size, shuffle_buffer, n_regr, shuffle=True):
    """
    Функция формирующая PrefetchDataset для процесса обучения
    
    :param series: исходный временной ряд
    :param window_size: размер окна
    :param batch_size: размер пакета
    :param shuffle_buffer: размер буфера перемешивания
    :param n_regr: число регрессоров
    :param shuffle: флаг перемешивания
    
    """

    regr_idxes = list(get_regressor_indexes(window_size, n_regr))
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.map(lambda wnd: (wnd[0::n_regr + 1], tf.transpose(tf.gather(wnd, [regr_idxes]))))

    # dataset = dataset.map(lambda wnd: (wnd[0::step], wnd[window_size // 2 - side: window_size // 2 + side - 1]))
    # dataset.map(lambda wnd: (wnd[0::step], wnd[window_size // 2]))
    if shuffle:
      dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(1)    
    return dataset

def windowed_dataset_ar(series, series_original, window_size, batch_size, shuffle_buffer, shuffle=True):

    series = np.hstack((series, series_original))
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.map(lambda wnd: (wnd[:, 0], wnd[:, 1]))

    if shuffle:
      dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset

