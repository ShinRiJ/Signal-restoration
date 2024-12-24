#!usr/bin/env/python3

import numpy as np
import scipy


def linear_interpolation(data, length):
    """
    Возвращает линейную интерполяцию числового ряда
    
    :param data: исходные данные для интерполяции
    :param length: период интерполяции
    """
    
    # TODO: расширить возможности интерполяции с помощью задания индексов
    
    xs = np.argwhere(~np.isnan(data)).squeeze()
    ys = data[xs]
    return scipy.interpolate.interp1d(xs, ys)(np.linspace(0, length, length, endpoint=False))


def fillna(data):
    """
    Заполнение пропущенных значений предыдущими
    
    :param data: исходные данные
    """
    
    mask = np.isnan(data)
    idxes = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idxes, axis=1, out=idxes)
    return data[np.arange(idxes.shape[0])[:,None], idxes]



#Модель RNN
def build_rnn_model(in_idxes, out_idxes):

  model_in = Input(shape=len(in_idxes))

  x = Dense(64, 'linear')(model_in)
    
  x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(model_in)
  x = tf.keras.layers.Bidirectional(LSTM(32, return_sequences=True))(x)
  x = tf.keras.layers.Bidirectional(LSTM(32, return_sequences=False))(x)

  x = Dense(64, 'linear')(model_in)
  x = Dense(32, 'linear')(x)

  x = Dense(len(out_idxes), 'linear')(x)
  return Model(inputs=[model_in], outputs=[x])

# Модель CNN 
def cnn_multimodel(in_idxes, out_idxes):
    
  model_in = Input(shape=len(in_idxes))
  
  x = tf.expand_dims(model_in, axis=-1)
  x = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same')(x)
  x = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same')(x)
  x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
  x = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(x)
  x = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(x)
  x = tf.keras.layers.AveragePooling1D(pool_size=2)(x)

  x = tf.keras.layers.Flatten()(x)
    
  x = tf.keras.layers.Dense(32, activation='relu')(x)
  x = tf.keras.layers.Dense(len(out_idxes))(x)
  return Model(inputs=[model_in], outputs=[x])