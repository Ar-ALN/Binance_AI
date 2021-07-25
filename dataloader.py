import pandas as pd
import numpy as np


def prepare_data(ratio=.9, time_step=100, prediction_forecast=50):
    data = pd.read_csv("data.csv", header=None)
    relative_candles = []
    for i in range(1, len(data)):
        relative_candles.append(((data[4][i] / data[4][i - 1]) - 1) * 10)

    useful_data_length = len(data) - time_step - prediction_forecast

    dataX, dataY = [], []
    for i in range(time_step, time_step + useful_data_length):
        dataX.append(relative_candles[i - time_step:i])

        avg_price = data[4][i: i + prediction_forecast].mean()
        current_price = data[4][i]
        prediction = 1 if avg_price > current_price else 0
        prediction = .5 if current_price * 1.003 > avg_price > current_price * 0.997 else prediction
        dataY.append(prediction)

    train_data_length = int(useful_data_length * ratio)
    test_data_length = useful_data_length - train_data_length - time_step

    x_train = np.array(dataX[:train_data_length]).reshape(train_data_length, time_step, 1)
    y_train = np.array(dataY[:train_data_length]).reshape(train_data_length, 1, 1)
    x_test = np.array(dataX[train_data_length + time_step:]).reshape(test_data_length, time_step, 1)
    y_test = np.array(dataY[train_data_length + time_step:]).reshape(test_data_length, 1, 1)

    return data, x_train, y_train, x_test, y_test
