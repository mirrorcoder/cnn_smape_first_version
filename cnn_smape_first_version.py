import pandas as pd
import numpy as np
from keras.layers import Reshape
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

MODE = "FIT1"
INPUT_FILE = 'ETHUSDT_1h_2806.csv'
FILE_FOR_BEST_MODEL = 'best_model_cnn_smape_first_version.h5'
COLUMNS_TO_KEEP = ['Open', 'High', 'Low', 'Close', 'Volume', 'Taker buy base asset volume']

def sMAPE(y_true, y_pred):
    return K.mean(2 * K.abs(y_pred - y_true) / (K.abs(y_pred) + K.abs(y_true)), axis=[-1, -2])

def trend_accuracy(y_true, y_pred):
    y_true_deltas = y_true[:, 1:, :] - y_true[:, :-1, :]
    y_pred_deltas = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    is_same_direction = K.equal(K.sign(y_true_deltas), K.sign(y_pred_deltas))
    return K.mean(is_same_direction, axis=-1)

def r_squared(y_true, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    r2 = tf.subtract(1.0, tf.divide(residual, total))
    return r2

if MODE == 'FIT':
    # Чтение данных
    data = pd.read_csv(INPUT_FILE)
    data = data[COLUMNS_TO_KEEP]

    # Нормализация данных с использованием MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Преобразование данных в формат, подходящий для CNN
    def create_dataset(dataset, look_back=1, look_forward=5):
        X, Y = [], []
        for i in range(len(dataset)-look_back-look_forward-1):
            a = dataset[i:(i+look_back)]
            X.append(a)
            Y.append(dataset[(i+look_back):(i+look_back+look_forward), :4]) # изменили здесь, чтобы брать только первые 4 значения (Open, High, Low, Close)
        return np.array(X), np.array(Y)

    look_back = 24  # количество предыдущих временных шагов, используемых для предсказания
    look_forward = 5  # количество будущих временных шагов, которые нужно предсказать

    # Определение callback'ов
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = ModelCheckpoint(FILE_FOR_BEST_MODEL, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    tensorboard_cbk = TensorBoard(log_dir='/home/mirrorcoder/PycharmProjects/neurobot/cnn_smape_first_version.log')

    # Кросс-валидация с использованием TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(data_scaled):
        X_train, X_test = data_scaled[train_index], data_scaled[test_index]
        X_train, Y_train = create_dataset(X_train, look_back, look_forward)
        X_test, Y_test = create_dataset(X_test, look_back, look_forward)

        # Создание и компиляция модели
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(Y_train.shape[1] * Y_train.shape[2]))
        model.add(Reshape((Y_train.shape[1], Y_train.shape[2])))
        model.compile(optimizer='adam', loss=sMAPE, metrics=[r_squared])

        # Обучение модели
        history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=200, callbacks=[es, mc, tensorboard_cbk])
else:
    def create_dataset(dataset, look_back=1, look_forward=5):
        X, Y = [], []
        for i in range(len(dataset)-look_back-look_forward-1):
            a = dataset[i:(i+look_back)]
            X.append(a)
            Y.append(dataset[(i+look_back):(i+look_back+look_forward), :4]) # изменили здесь, чтобы брать только первые 4 значения (Open, High, Low, Close)
        return np.array(X), np.array(Y)

    # Загрузите модель из файла
    model = load_model(FILE_FOR_BEST_MODEL, custom_objects={'sMAPE': sMAPE, 'r_squared': r_squared})

    # Загрузка данных
    data = pd.read_csv(INPUT_FILE)  # замените на имя вашего файла
    data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Taker buy base asset volume']]

    # Применение того же MinMaxScaler для нормализации
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Создание второго скалера только для первых четырех столбцов
    scaler_4 = MinMaxScaler()
    scaler_4.fit(data.iloc[:, :4])

    # Преобразование данных в формат, подходящий для CNN
    look_back = 24
    look_forward = 5
    X, Y = create_dataset(data_scaled, look_back, look_forward)

    # Предсказание
    predictions = model.predict(X)

    # Инверсия нормализации для визуализации
    predictions_inverse = scaler_4.inverse_transform(predictions.reshape(-1, 4))
    Y_inverse = scaler_4.inverse_transform(Y.reshape(-1, 4))

    import mplfinance as mpf


    def plot_candles(real, predicted):
        real_df = pd.DataFrame(real, columns=['Open', 'High', 'Low', 'Close'], index=pd.date_range(start="2012-01-01", end="2012-01-05", freq='D'))
        pred_df = pd.DataFrame(predicted, columns=['Open', 'High', 'Low', 'Close'], index=pd.date_range(start="2012-01-01", end="2012-01-05", freq='D'))

        # Create subplot
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

        # Plot real data
        mpf.plot(real_df, type='candle', ax=axes[0])
        axes[0].set_title('Real')

        # Plot predicted data
        mpf.plot(pred_df, type='candle', ax=axes[1])
        axes[1].set_title('Predicted')

        plt.savefig('prediction_graph.png')
        plt.close()

    # Вызов функции отображения
    plot_candles(Y_inverse[-101 * look_forward:-100 * look_forward], predictions_inverse[-101 * look_forward:-100 *look_forward])
