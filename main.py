from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import mean_squared_error

from dataloader import prepare_data

time_step = 600

data, x_train, y_train, x_test, y_test = prepare_data(time_step=time_step, prediction_forecast=30, subset=15000)

plt.close("all")


def deep_network_LSTM(name_model, x_train, y_train, x_test, y_test, input_shape, activation_function='sigmoid', opt='adam',
                      epochs=50, batch_size=256):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(1, kernel_initializer="uniform", activation='linear'))
    model.compile(optimizer=opt, loss='mae', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        batch_size=batch_size, epochs=epochs, verbose=1)
    # model.save(name_model)
    # print("Saved model to disk")

    return model, history


def prediction_model_plot(model, x_train, y_train, x_test, y_test, data, time_step):
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    x = np.arange(len(data["close"]))

    train_predict_plot = np.empty_like(data["close"])
    train_predict_plot[:] = np.nan
    train_predict_plot[time_step:len(train_predict) + time_step] = train_predict.reshape(train_predict.shape[0])

    test_predict_plot = np.empty_like(data["close"])
    test_predict_plot[:] = np.nan
    test_predict_plot[time_step + len(train_predict): time_step + len(train_predict) + len(test_predict)] = test_predict.reshape(test_predict.shape[0])
    fig = plt.figure(figsize=(12, 1), dpi=1200)

    axe1 = fig.add_subplot(111)
    axe1.plot(x, data["close"], linewidth=0.033333)
    axe1.set_ylabel('values')

    axe2 = axe1.twinx()
    axe2.set_ylabel('train')
    axe2.plot(x, train_predict_plot, linewidth=0.033333, color='green')

    axe2 = axe1.twinx()
    axe2.set_ylabel('test')
    axe2.plot(x, test_predict_plot, linewidth=0.033333, color='pink')

    plt.show()


def plot_hp(model_plot, hyperparameter):
    train_hp = hyperparameter
    validation_hp = 'val_' + hyperparameter
    model = model_plot
    plt.figure(figsize=(12, 6), dpi=160)
    plt.plot(model.history[train_hp], label='train')
    plt.plot(model.history[validation_hp], alpha=0.7, label='validation')
    plt.xlabel(hyperparameter)
    plt.ylabel('Loss')
    plt.legend()
    plt.title(hyperparameter + ' vs Epochs for Model 1', size=25)
    plt.grid()
    plt.show()


model, history = deep_network_LSTM('model', x_train, y_train, x_test, y_test, x_train[0].shape, epochs=1200)
plot_hp(history, 'loss')
plot_hp(history, 'accuracy')
prediction_model_plot(model, x_train, y_train, x_test, y_test, data, time_step=time_step)
