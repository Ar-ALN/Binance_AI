import pandas as pd
import numpy as np
import pandas_ta as ta  # Useful do not delete


def prepare_data(ratio=.9, time_step=100, prediction_forecast=50, subset=None):
    df = pd.read_csv("data.csv")
    if subset is not None:
        df = df.iloc[:subset, :]
    df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)

    df.ta.ema(length=100, append=True)

    relative_candles = np.empty(shape=len(df))
    relative_candles[0] = 0
    for i in range(1, len(df)):
        relative_candles[i] = ((df["close"][i] / df["close"][i - 1]) - 1) * 20
    useful_data_length = len(df) - time_step - prediction_forecast


    dataX, dataY = [], []
    for i in range(time_step, time_step + useful_data_length):
        dataX.append(relative_candles[i - time_step:i])

        prediction = 1 if df["close"][i] > df["EMA_100"][i + prediction_forecast] else 0
        dataY.append(prediction)

    train_data_length = int(useful_data_length * ratio)
    test_data_length = useful_data_length - train_data_length - time_step

    x_train = np.array(dataX[:train_data_length]).reshape(train_data_length, time_step, 1)
    y_train = np.array(dataY[:train_data_length]).reshape(train_data_length, 1, 1)
    x_test = np.array(dataX[train_data_length + time_step:]).reshape(test_data_length, time_step, 1)
    y_test = np.array(dataY[train_data_length + time_step:]).reshape(test_data_length, 1, 1)

    return df, x_train, y_train, x_test, y_test
