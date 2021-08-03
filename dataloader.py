import pandas as pd
import numpy as np
import pandas_ta as ta  # Useful do not delete
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import math


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def prepare_data(ratio=.9, time_step=100, prediction_forecast=50, subset=None):
    df = pd.read_csv("data.csv")
    if subset is not None:
        df = df.iloc[:subset, :]
    df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)

    scaler = MinMaxScaler()
    df.ta.ema(length=30, append=True)
    df.ta.ema(length=10, append=True)
    df.ta.macd(append=True)
    df.ta.rsi(append=True)

    i = 0
    for index, row in df.iterrows():
        scale_on_price_func = lambda x, fac: ((row['close'] / x) - 1) * fac

        df.at[index, 'R_EMA_10'] = scale_on_price_func(row['EMA_10'], 20)
        df.at[index, 'R_EMA_30'] = scale_on_price_func(row['EMA_30'], 20)
        df.at[index, 'R_MACD_12_26_9'] = 1#sigmoid(row['MACD_12_26_9'] / 100)
        df.at[index, 'R_MACDh_12_26_9'] = 1#sigmoid(row['MACDh_12_26_9'] / 100)
        df.at[index, 'R_MACDs_12_26_9'] = 1#<sigmoid(row['MACDs_12_26_9'] / 100)
        df.at[index, 'R_RSI_14'] = row['RSI_14'] / 100
        df.at[index, 'R_volume'] = row['volume'] / 1000

        if i % 10000 == 0:
            print(f"preprocessing 1/3 at {int(i / len(df) * 100)}%")
        i += 1

    relative_candles = np.empty(shape=len(df))
    relative_candles[0] = 0
    for i in range(1, len(df)):
        relative_candles[i] = ((df["close"][i] / df["close"][i - 1]) - 1) * 20
        if i % 10000 == 0:
            print(f"preprocessing 2/3 at {int(i / len(df) * 100)}%")
    useful_data_length = len(df) - time_step - prediction_forecast - 30


    dataX, dataY = [], []
    for i in range(time_step + 30, time_step + useful_data_length + 30):
        time_step_input = np.array([
            relative_candles[i - time_step:i],
            df["R_volume"][i - time_step:i],
            df["R_EMA_10"][i - time_step:i],
            df["R_EMA_30"][i - time_step:i],
            df["R_MACD_12_26_9"][i - time_step:i],
            df["R_MACDh_12_26_9"][i - time_step:i],
            df["R_MACDs_12_26_9"][i - time_step:i],
            df["R_RSI_14"][i - time_step:i],
        ])
        dataX.append(time_step_input)

        prediction = 1 if df["close"][i] > df["EMA_30"][i + prediction_forecast] else 0
        dataY.append(prediction)
        if i % 10000 == 0:
            print(f"preprocessing 3/3 at {int(i / len(df) * 100)}%")

    train_data_length = int(useful_data_length * ratio)
    test_data_length = useful_data_length - train_data_length - time_step - 30

    x_train = np.array(dataX[:train_data_length]).reshape((train_data_length, time_step, len(dataX[0])))
    y_train = np.array(dataY[:train_data_length]).reshape((train_data_length, 1, 1))
    x_test = np.array(dataX[train_data_length + time_step + 30:]).reshape((test_data_length, time_step, len(dataX[0])))
    y_test = np.array(dataY[train_data_length + time_step + 30:]).reshape((test_data_length, 1, 1))

    return df, x_train, y_train, x_test, y_test
