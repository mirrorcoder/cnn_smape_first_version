import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import talib
from sklearn.preprocessing import LabelEncoder


MODE = "FIT1"
INPUT_FILE = 'ETHUSDT_1h_0507.csv'
FILE_FOR_BEST_MODEL = 'best_model_cnn_second_version.h5'
COLUMNS_TO_KEEP = ['Open', 'High', 'Low', 'Close', 'Volume', 'Taker buy base asset volume']
SCALER_CNN_SMAPE = 'scaler_cnn_smape.pkl'


def add_indicators(data):
    # Add RSI
    data['RSI'] = talib.RSI(data['Close'].values, timeperiod=14)

    # Add MACD
    macd_line, signal_line, hist = talib.MACD(data['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    data['MACD'] = macd_line
    data['MACD_signal'] = signal_line

    return data

def categorize_outputs(data, periods=3):
    # Calculate percentage change over specified periods
    data['Pct_change'] = (data['Close'].shift(-periods+1) - data['Open']) / data['Open']
    # Handle NaN values
    data['Pct_change'].fillna(0, inplace=True) # replace NaNs with 0
    data.dropna(inplace=True) # or remove rows with NaNs
    # Categorize outputs
    bins = [-np.inf, -0.03, -0.01, 0.01, 0.03, np.inf]
    labels = ['Strong decrease', 'Weak decrease', 'Stable', 'Weak increase', 'Strong increase']
    data['Category'] = pd.cut(data['Pct_change'], bins=bins, labels=labels)

    return data


def sMAPE(y_true, y_pred):
    return K.mean(2 * K.abs(y_pred - y_true) / (K.abs(y_pred) + K.abs(y_true)), axis=[-1, -2])


def trend_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=0)
    y_pred = tf.squeeze(y_pred, axis=0)

    # Use tf.subtract on slices of y_true and y_pred to get the differences
    true_delta = tf.subtract(y_true[1:], y_true[:-1])
    pred_delta = tf.subtract(y_pred[1:], y_pred[:-1])

    true_sign = tf.sign(true_delta)
    pred_sign = tf.sign(pred_delta)

    correct_preds = tf.equal(true_sign, pred_sign)
    return tf.reduce_mean(tf.cast(correct_preds, dtype=tf.float32))

def r_squared(y_true, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    r2 = tf.subtract(1.0, tf.divide(residual, total))
    return r2

# Преобразование данных в формат, подходящий для CNN
def create_dataset(dataset, look_back=1, look_forward=1, num_categories=5):
    X, Y = [], []
    for i in range(len(dataset) - look_back - look_forward):
        a = dataset[i:(i + look_back), :-num_categories]
        X.append(a)
        Y.append(dataset[i + look_back, -num_categories:])  # now selecting all output category columns
    return np.array(X), np.array(Y)

if MODE == 'FIT':
    # Чтение данных
    data = pd.read_csv(INPUT_FILE)
    data = data[COLUMNS_TO_KEEP]
    # Apply functions
    data = add_indicators(data)
    data = categorize_outputs(data)
    class_counts = data['Category'].value_counts()
    print(class_counts)
    # Преобразование категорий в one-hot encoding
    data = pd.get_dummies(data, columns=['Category'])

    # Масштабирование данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    with open(SCALER_CNN_SMAPE, 'wb') as f:
        pickle.dump(scaler, f)

    look_back = 48  # количество предыдущих временных шагов, используемых для предсказания
    look_forward = 1

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = ModelCheckpoint(FILE_FOR_BEST_MODEL, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    tensorboard_cbk = TensorBoard(log_dir='/home/mirrorcoder/PycharmProjects/neurobot/cnn_smape_first_version.log')

    # Преобразование категорий в числовые значения
    encoder = LabelEncoder()
    data_scaled[:, -1] = encoder.fit_transform(data_scaled[:, -1])

    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(data_scaled):
        X_train, X_test = data_scaled[train_index], data_scaled[test_index]
        X_train, Y_train = create_dataset(X_train, look_back, look_forward)
        X_test, Y_test = create_dataset(X_test, look_back, look_forward)

        # Convert Y_train from one-hot vectors to class integers
        Y_train_classes = np.argmax(Y_train, axis=1)

        # Calculate class counts
        class_counts = np.bincount(Y_train_classes)

        # Calculate class weights
        class_weights = len(Y_train) / (len(class_counts) * class_counts)

        # Create class weights dictionary
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        # Создание и компиляция модели
        model = Sequential()
        model.add(
            Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(
            Dense(5, activation='softmax'))  # Использование функции активации softmax для многоклассовой классификации
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Обучение модели
        history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=200,
                            callbacks=[es, mc, tensorboard_cbk], class_weight=class_weight_dict)
else:
    # Загрузите модель из файла
    model = load_model(FILE_FOR_BEST_MODEL)

    data = pd.read_csv(INPUT_FILE)
    data = data[COLUMNS_TO_KEEP]
    # Apply functions
    data = add_indicators(data)
    data = categorize_outputs(data)
    class_counts = data['Category'].value_counts()
    print(class_counts)
    # Преобразование категорий в one-hot encoding
    data = pd.get_dummies(data, columns=['Category'])

    scaler = None
    with open(SCALER_CNN_SMAPE, 'rb') as f:
        scaler = pickle.load(f)
    data_scaled = scaler.transform(data)

    look_back = 48  # количество предыдущих временных шагов, используемых для предсказания
    look_forward = 1

    X, Y = create_dataset(data_scaled, look_back, look_forward)

    # Предсказание
    predictions = model.predict(X)

    # Преобразование прогнозов и истинных меток из one-hot в метки классов
    predictions_classes = np.argmax(predictions, axis=1)
    Y_classes = np.argmax(Y, axis=1)

    # Подсчет правильно и неправильно угаданных классов
    correct = np.sum(Y_classes == predictions_classes)
    incorrect = len(Y_classes) - correct
    print(f'Correct predictions: {correct}')
    print(f'Incorrect predictions: {incorrect}')

    # Для более подробного отчета по классам можно использовать classification_report из sklearn
    print(classification_report(Y_classes, predictions_classes))

    # Для визуализации количества правильно и неправильно угаданных классов по каждому классу можно использовать confusion_matrix
    print(confusion_matrix(Y_classes, predictions_classes))