from utils.data_utils import calculate_window_size, get_predictor_indexes, get_regressor_indexes, windowed_dataset_multi
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, AveragePooling1D, Flatten, concatenate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Model
from scipy.interpolate import interp1d
from scipy import fft, arange
import plotly.express as px
import xgboost as xgb
import tensorflow as tf
from sklearn.multioutput import MultiOutputRegressor
import tensorflow_datasets as tfds
import streamlit as st
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import time

#Модель MLP
def build_mlp_model(n_input, n_output):
    model_in = Input(shape=n_input)
    
    x = Dense(128, 'linear')(model_in)
    x = Dense(64, 'linear')(x)
    x = Dense(32, 'linear')(x)

    x = Dense(n_output, 'linear')(x)
    
    return Model(inputs=[model_in], outputs=[x])

# Модель CNN 
def build_cnn_model(in_idxes, out_idxes):

    model_in = Input(shape=in_idxes)

    x = tf.expand_dims(model_in, axis=-1)
    x = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=2)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(out_idxes)(x)
    return Model(inputs=[model_in], outputs=[x])

#Модель RNN
def build_rnn_model(in_idxes, out_idxes):
    model_in = Input(shape=in_idxes)

    x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(model_in)
    x = tf.keras.layers.Bidirectional(LSTM(32, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(LSTM(32, return_sequences=False))(x)

    x = Dense(128, 'linear')(model_in)
    x = Dense(64, 'linear')(x)
    x = Dense(32, 'linear')(x)

    x = Dense(out_idxes, 'linear')(x)
    return Model(inputs=[model_in], outputs=[x])

#Модель XGB
def XGB(dataset, depth = 5, trees = 250, booster='gbtree', lr = 0.05):

    test = tfds.as_numpy(dataset)
    X_train, y_train = zip(*test)
    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)
    y_train = y_train.reshape(-1, y_train.shape[1])
    xgb_reg = MultiOutputRegressor(xgb.XGBRegressor(max_depth=depth, n_estimators=trees, n_jobs=4,objectvie='reg:squarederror', booster=booster, learning_rate=lr, tree_method="gpu_hist")) #

    xgb_reg.fit(X_train, y_train)

    return xgb_reg

# Восстанавливает сигнал
def restore_signal(series, model, w_size, in_idxes, out_idxes):
    preds = np.full(len(series), np.nan)
    n_pred = len(in_idxes)
    n_regrs = len(out_idxes)

    for i in range(0, (len(series) - w_size) // (n_regrs + 1)): # len(series) - (len(series) % window_size)
        preds[in_idxes + (n_regrs + 1) * i] = series[in_idxes + (n_regrs + 1) * i]
        preds[out_idxes + (n_regrs + 1) * i] = np.array(model(series[in_idxes +  (n_regrs + 1) * i].reshape(1, -1)))

    return preds

def restore_signal_xgb(series, w_size, in_idxes, out_idxes, model):
    preds = np.full(len(series), np.nan)
    n_pred = len(in_idxes)
    n_regrs = len(out_idxes)

    for i in range(0, (len(series) - w_size) // (n_regrs + 1)): # len(series) - (len(series) % window_size)
        preds[in_idxes + (n_regrs + 1) * i] = series[in_idxes + (n_regrs + 1) * i]
        preds[out_idxes + (n_regrs + 1) * i] = model.predict(series[in_idxes +  (n_regrs + 1) * i].reshape(1, -1))

    return preds

def get_datasets(data, split_ratio, n_preds, n_regrs):
    # Разделение исходного временного ряда на выборки
    split_idx = int(len(data) * SPLIT_RATIO)
    training_series = data.iloc[:split_idx]
    validation_series = data.iloc[split_idx:]
    
    # Стандартизация величин
    training_series_std = std_scaler.fit_transform(training_series)
    validation_series_std = std_scaler.transform(validation_series)
        
    # Формирование оконных выборок
    training_dataset = windowed_dataset_multi(training_series_std.reshape(-1,), window_size, 
                                                 BATCH_SIZE, SHUFFLE_BUFFER_SIZE, n_regrs)

    validation_dataset = windowed_dataset_multi(validation_series_std.reshape(-1,), window_size, 
                                                 BATCH_SIZE, SHUFFLE_BUFFER_SIZE, n_regrs)
    
    return training_dataset, validation_dataset

st.set_page_config(layout="wide",)

hide_st_style = """
                <style>
                footer {visibility: hidden;}
                </style>
                """

st.markdown(hide_st_style, unsafe_allow_html = True)

st.sidebar.title('Загрузка исходных данных:')
spectra = st.sidebar.file_uploader("Загрузка файла", type=['xls', 'xlsx'])

if spectra is not None:
    st.sidebar.success("Файл успешно загружен")
    df = pd.read_excel(spectra)
    df[df.columns[0]] = pd.to_datetime(df['Время'])
    df.set_index('Время', inplace=True)
else:
    st.sidebar.warning("Необходимо загрузить исходные данные")
    
uploadbtn = st.sidebar.button("Начать")

st.markdown('## Алгоритм интеллектуального прореживания сигнала:')

if spectra is not None:
    st.checkbox('Данные загружены', value=True, disabled=True)
else:
    st.checkbox('Данные загружены', value=False, disabled=True)

if "uploadbtn_state" not in st.session_state:
    st.session_state.uploadbtn_state = False

if (uploadbtn or st.session_state.uploadbtn_state) and spectra is not None:
    st.session_state.uploadbtn_state = True
    
    st.markdown('## Исходные данные:')
    
    values = st.slider('Выберете диапазон исследуемой выборки для обучения:', 0, df.shape[0], (df.shape[0]//2 - int(df.shape[0]*0.1), df.shape[0]//2 + int(df.shape[0]*0.1)), key = 101)
    values_test = st.slider('Выберете диапазон исследуемой выборки для оценивания:', 0, df.shape[0], (df.shape[0]//2 - int(df.shape[0]*0.1), df.shape[0]//2 + int(df.shape[0]*0.1)), key = 102)
    
    fig_first = px.line()

    fig_first.add_scatter(x=df.index, y=df[df.columns[0]], name="Исследуемая выборка: Полная выборка")
    fig_first.add_scatter(x=df.index[values[0]:values[1]], y=df[df.columns[0]][values[0]:values[1]], name="Исследуемая выборка: Обучение")
    fig_first.add_scatter(x=df.index[values_test[0]:values_test[1]], y=df[df.columns[0]][values_test[0]:values_test[1]], name="Исследуемая выборка: Тестовая")
 
    st.plotly_chart(fig_first, theme= None, use_container_width=True)
    
    if(st.checkbox('Исследуемая выборка выделена')):
        st.session_state.step_two = True
    else:
        st.session_state.step_two = False
    
    if(st.session_state.step_two):
        st.markdown('## Исследуемая выборка:')

        graph_steam = df.iloc[values[0]: values[1]]
        
        resampled_steam = df.iloc[values[0]: values[1]]. \
                        resample('0.1S'). \
                        last(). \
                        interpolate('pchip', order=3)
        
        fig_second = px.line()
        fig_second.add_scatter(x=graph_steam.index, y=graph_steam[graph_steam.columns[0]], name="Исследуемая выборка")
        st.plotly_chart(fig_second, theme= None, use_container_width=True)
                          
        resampled_steam_test = df.iloc[values_test[0]: values_test[1]]. \
                            resample('0.1S'). \
                            last(). \
                            interpolate('pchip', order=3)
        
        method = st.selectbox("Первичный метод исследования:", [' ', 'XGBoost', 'DNN', 'RNN', 'CNN'], index = 0)
        
        if "step_three" not in st.session_state:
            st.session_state.step_three = False
            
        if(method != ' '):
            st.success("Выбранный метод: " + method)

            min_max_period = st.columns(3)
            min_period = min_max_period[0].number_input("Минимальный период, сек", value=0, min_value = 0, step = 1)
            max_period = min_max_period[1].number_input("Максимальный период, сек", value=20, min_value = min_period, step = 1)
            step_period = min_max_period[2].number_input("Шаг исследования, сек", value=1, min_value = 1, step = 1)
            
            if max_period < min_period:
                st.error("Минимальный период должен быть меньше максимального!")
            else:
                st.markdown(f"Минимальный период, сек: {0.1 if min_period == 0 else min_period}")
                st.markdown(f"Максимальный период, сек: {max_period}")
                st.markdown(f"Шаг изменения периода, сек: {step_period}")
            
            if(st.checkbox('Первичный метод исследования выбран')):
                st.session_state.step_three = True
            else:
                st.session_state.step_three = False 
        else:
            st.warning("Необходимо выбрать метод первичного исследования")
            
        if(st.session_state.step_three):

            if "step_four" not in st.session_state:
                st.session_state.step_four = False

            st.markdown('## Обучение:')
            # Параметры
            SPLIT_RATIO = 0.9
            BATCH_SIZE = 64 # Размер батча
            SHUFFLE_BUFFER_SIZE = 10000 # Размер буфера для шафла
            N_EPOCHS = 25 # Максимальное количество эпох обучения
            LEARNING_RATE = 1e-3 # Скорость обучения
            
            std_scaler = StandardScaler()
            
            early_stopper = EarlyStopping('val_loss', min_delta=0.001, patience=3, verbose=1)
            lr_reducer = ReduceLROnPlateau('val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)
            
            n_regrs_samples_fit = list(np.arange(min_period * 10, (max_period * 10) + 1, step_period * 10, dtype=int))
            
            if(min_period == 0):
                n_regrs_samples_fit[0] = 1
            
            if(not st.session_state.step_four):
                metrics = {
                        'mse': [],
                        'mae': [],
                        'r2':  []
                    }
                progress_bar = st.progress(0)
                progress_text = st.empty()
                progress_text.text("Прогресс обучения и оценки моделей: 0 %")
            else:
                progress_bar = st.progress(1.0)
                progress_text = st.empty()
                progress_text.text("Прогресс обучения и оценки моделей: 100 %")
            
            if(method == 'XGBoost' and st.session_state.step_three):

                if(st.session_state.step_four == False):
                    for i, n_regrs in enumerate(n_regrs_samples_fit):    
                        n_preds = 4
                        
                        print(f"Количество регрессоров: {n_regrs}")

                        # Определение величины окна
                        window_size = calculate_window_size(n_preds, n_regrs)

                        # Определение индексов предикторов
                        in_idxes =  get_predictor_indexes(window_size, n_preds)

                        # Определение индексов регрессоров
                        out_idxes = get_regressor_indexes(window_size, n_regrs)

                        # Формирование выборок
                        training_dataset, validation_dataset = get_datasets(resampled_steam, SPLIT_RATIO, len(in_idxes), len(out_idxes))

                        # Инициализация модели
                        tf.keras.backend.clear_session()
                        
                        xgb_model = XGB(training_dataset)
                        
                        predict_std = std_scaler.transform(resampled_steam_test).squeeze()
                        y_test_pred = restore_signal_xgb(predict_std, window_size, in_idxes, out_idxes, xgb_model)
                        series_pred_xgb = std_scaler.inverse_transform(y_test_pred.reshape(1, -1)).transpose()
                        
                        metrics['mse'].append(mean_squared_error(series_pred_xgb[window_size:-window_size], resampled_steam_test[window_size:-window_size]))
                        metrics['mae'].append(mean_absolute_error(series_pred_xgb[window_size:-window_size], resampled_steam_test[window_size:-window_size]))
                        metrics['r2'].append(r2_score(series_pred_xgb[window_size:-window_size], resampled_steam_test[window_size:-window_size]))

                        np.savetxt(f'{n_regrs}.res', series_pred_xgb[window_size:-window_size], fmt='%.6f')

                        progress_bar.progress((i + 1)/len(n_regrs_samples_fit))
                        progress_text.text(f"Прогресс обучения и оценки моделей {method}: {round(100 * (i + 1)/len(n_regrs_samples_fit), 2)} %")

                    np.savetxt('metrics.out', ( metrics['r2'], metrics['mae'], metrics['mse']), fmt='%.6f')

                    st.session_state.step_four = True

                if(st.session_state.step_four == True):
                    metrics_copy = np.loadtxt('metrics.out')

                    st.markdown('## Результаты (оценки):')
                    st.markdown('Коэффициент детерминации моделей:')
                    fig_result_r2 = px.line().update_layout(xaxis_title="Период, с")
                    fig_result_r2.add_scatter(x=[x/10 for x in n_regrs_samples_fit], y= metrics_copy[0], name="R2")
                    st.plotly_chart(fig_result_r2, theme= None, use_container_width=True)
                    
                    st.markdown('Среднаяя абсолютная ошибка моделей:')
                    fig_result_mae = px.line().update_layout(xaxis_title="Период, с")
                    fig_result_mae.add_scatter(x=[x/10 for x in n_regrs_samples_fit], y= metrics_copy[1], name="MAE")
                    st.plotly_chart(fig_result_mae, theme= None, use_container_width=True)

                    st.markdown('Средняя квадратичная ошибка моделей:')
                    fig_result_mse = px.line().update_layout(xaxis_title="Период, с")
                    fig_result_mse.add_scatter(x=[x/10 for x in n_regrs_samples_fit], y= metrics_copy[2], name="MSE")
                    st.plotly_chart(fig_result_mse, theme= None, use_container_width=True)
                    
                    
                    chosen_period = st.number_input("Введите интересуемый период записи:", value=min_period + 1, min_value = min_period, max_value = max_period,  step = step_period)
                    
                    st.markdown(f'Для периода {0.1 if chosen_period == 0 else chosen_period} секунд:')
                    st.markdown(f'Объём за сутки: {24*60*60 / (chosen_period + 1)} значений')
                    st.markdown(f'R2: {metrics_copy[0][chosen_period - min_period]}')
                    st.markdown(f'MAE: {metrics_copy[1][chosen_period - min_period]}')
                    st.markdown(f'MSE: {metrics_copy[2][chosen_period - min_period]}')

                    if 'step_five' not in st.session_state:
                        st.session_state['step_five'] = {}

                    if(st.checkbox('Период выбран')):
                        st.session_state['step_five'] = True
                    else:
                        st.session_state['step_five'] = False

                    if(st.session_state.step_five):
                        w_size = calculate_window_size(4, n_regrs_samples_fit[chosen_period])
                        st.markdown('## Результаты (демонстрация восстановления):')
                        fig_result_chosen_period = px.line().update_layout(xaxis_title="Период, с")
                        fig_result_chosen_period.add_scatter(x=resampled_steam_test[w_size:-w_size].index, y= resampled_steam_test[w_size:-w_size][resampled_steam_test.columns[0]], name="Тестовая выборка")
                        fig_result_chosen_period.add_scatter(x=resampled_steam_test[w_size:-w_size].index, y= np.loadtxt(f'{1 if chosen_period == 0 else chosen_period * 10}.res'), name="Восстановление")
                        st.plotly_chart(fig_result_chosen_period, theme= None, use_container_width=True)

                    while(True):
                        time.sleep(1)

            elif(method == 'DNN' and st.session_state.step_three):
                for i, n_regrs in enumerate(n_regrs_samples_fit):    
                    n_preds = 4

                    # Определение величины окна
                    window_size = calculate_window_size(n_preds, n_regrs)

                    # Определение индексов предикторов
                    in_idxes =  get_predictor_indexes(window_size, n_preds)

                    # Определение индексов регрессоров
                    out_idxes = get_regressor_indexes(window_size, n_regrs)

                    # Формирование выборок
                    training_dataset, validation_dataset = get_datasets(resampled_steam, SPLIT_RATIO, len(in_idxes), len(out_idxes))

                    # Инициализация модели
                    tf.keras.backend.clear_session()
                    
                    mlp_model = build_mlp_model(len(in_idxes), len(out_idxes))
                    mlp_model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(LEARNING_RATE, momentum=0.9))
                    
                    history = mlp_model.fit(
                        training_dataset,
                        validation_data=validation_dataset,
                        epochs=N_EPOCHS,
                        callbacks=[early_stopper, lr_reducer]
                    )

                    # Восстановление пропущенных значений
                    series_pred = restore_signal(
                        std_scaler.transform(resampled_steam_test).reshape(-1,),
                        mlp_model,
                        window_size,
                        in_idxes,
                        out_idxes
                    )
                    series_pred = std_scaler.inverse_transform(series_pred.reshape(-1, 1))
                    
                    metrics['mse'].append(mean_squared_error(series_pred[window_size:-window_size], resampled_steam_test[window_size:-window_size]))
                    metrics['mae'].append(mean_absolute_error(series_pred[window_size:-window_size], resampled_steam_test[window_size:-window_size]))
                    metrics['r2'].append(r2_score(series_pred[window_size:-window_size], resampled_steam_test[window_size:-window_size]))

                    progress_bar.progress((i + 1)/len(n_regrs_samples_fit))
                    progress_text.text(f"Прогресс обучения и оценки моделей {method}: {round(100 * (i + 1)/len(n_regrs_samples_fit), 2)} %")

            elif(method == 'RNN' and st.session_state.step_three):
                print("Not implemented")
            elif(method == 'CNN' and st.session_state.step_three):
                print("Not implemented")
