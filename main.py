from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import mean_squared_error

from dataloader import prepare_data

time_step = 100

data, x_train, y_train, x_test, y_test = prepare_data(time_step=time_step, prediction_forecast=40)

plt.close("all")


def deep_network_LSTM(name_model, x_train, y_train, x_test, y_test, shape, activation_function='sigmoid', opt='adam',
                      epochs=100, batch_size=2048):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(shape, 1)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))
    # Create callbacks
    # callbacks = [EarlyStopping(monitor='val_loss', patience=5),
    # ModelCheckpoint('../models/model.h5'), save_best_only=True,
    # save_weights_only=False)]
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        batch_size=batch_size, epochs=epochs, verbose=1)
    # model.save(name_model)
    # print("Saved model to disk")

    return model, history


def prediction_model_plot(model, x_train, y_train, x_test, y_test, data, time_step):
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    x = np.arange(len(data[4]))

    train_predict_plot = np.empty_like(data[4])
    train_predict_plot[:] = np.nan
    train_predict_plot[time_step:len(train_predict) + time_step] = train_predict.reshape(train_predict.shape[0])

    test_predict_plot = np.empty_like(data[4])
    test_predict_plot[:] = np.nan
    test_predict_plot[time_step + len(train_predict): time_step + len(train_predict) + len(test_predict)] = test_predict.reshape(test_predict.shape[0])
    fig = plt.figure(figsize=(12, 1), dpi=1200)

    axe1 = fig.add_subplot(111)
    axe1.plot(x, data[4], linewidth=0.033333)
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


model1, history1 = deep_network_LSTM('model1', x_train, y_train, x_test, y_test, time_step, epochs=3)
plot_hp(history1, 'loss')
plot_hp(history1, 'accuracy')
prediction_model_plot(model1, x_train, y_train, x_test, y_test, data, time_step=time_step)
