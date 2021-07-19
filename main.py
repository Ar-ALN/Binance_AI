from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import mean_squared_error

plt.close("all")

data = pd.read_csv("data.csv", header=None)
print(data)
values = data[4].to_numpy()
for i in range(len(data) - 1):
    values[i] = ((values[i] / values[i + 1]) - 1) * 10
values = values[:len(values) - 1]

# ts = pd.Series(values[:, 1])
# plt.figure()
# plt.plot(ts)
# plt.show()

train_data = values[:-10000]
test_data = values[-10000:]
training_dataset_length = len(train_data)

# train_data = train_data[0:training_dataset_length, :]

# Used for visualization and test purposes
all_mid_data = np.concatenate([train_data, test_data], axis=0)


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step)]
        dataX.append(a)
        dataY.append(dataset[i + time_step])
    return np.array(dataX), np.array(dataY)


def deep_network_LSTM(name_model, x_train, y_train, x_test, y_test, shape, activation_function='sigmoid', opt='adam',
                      epochs=100, batch_size=64):
    # Initialising the RNN
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(shape, 1)))
    model.add(Dropout(0.2))

    # Adding a second LSTM layer and Dropout layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # Adding a third LSTM layer and Dropout layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # Adding a fourth LSTM layer and and Dropout layer
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    # Adding the output layer
    # For Full connection layer we use dense
    # As the output is 1D so we use unit=1
    model.add(Dense(units=1))
    # Create callbacks
    # callbacks = [EarlyStopping(monitor='val_loss', patience=5),
    # ModelCheckpoint('../models/model.h5'), save_best_only=True,
    # save_weights_only=False)]
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        batch_size=batch_size, epochs=epochs, verbose=1

                        )
    # model.save(name_model)
    # print("Saved model to disk")

    return (model, history)


def prediction_model_plot(model, x_train, y_train, x_test, y_test, values, look_back, x_min=39400, x_max=39500,
                          y_min=32000, y_max=37000):
    ### Lets Do the prediction and check performance metrics
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)
    ##Transformback to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    math.sqrt(mean_squared_error(y_train, train_predict))
    ### Test Data RMSE
    math.sqrt(mean_squared_error(y_test, test_predict))

    ### Plotting
    # shift train predictions for plotting

    trainPredictPlot = np.empty_like(values)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(values)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(values) - 1, :] = test_predict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(values))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.axis([x_min, x_max, y_min, y_max])  # permet de zoomer sur une partie de la courbe
    plt.show()


def plot_hp(model_plot, hyperparameter):
    train_hp = hyperparameter
    validation_hp = 'val_' + hyperparameter
    model = model_plot
    plt.figure(figsize=(8, 6))
    plt.plot(model.history[train_hp], label='train')
    plt.plot(model.history[validation_hp], alpha=0.7, label='validation')
    plt.xlabel(hyperparameter)
    plt.ylabel('Loss')
    plt.legend()
    plt.title(hyperparameter + ' vs Epochs for Model 1', size=25)
    plt.grid()
    plt.show()


def reshape_data(time_step, train_data, test_data):
    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    x_train, y_train = create_dataset(train_data, time_step)
    x_test, y_test = create_dataset(test_data, time_step)

    # reshape le tout
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)

    return x_train, y_train, x_test, y_test


####### Nous allons regarder ici si le time step inlfue sur la qualité de la prédiction
####### Nous allons tester des valeurs de time_step pour 100,150 et 200
####### Test à ajouter : 3 modèles de préductions de stock return (mise en valeur de la variation non le prix)

### Ici Model 1
print('model 1')
time_step1 = 100
x_train1, y_train1, x_test1, y_test1 = reshape_data(time_step1, train_data, test_data)
model1, history1 = deep_network_LSTM('model1', x_train1, y_train1, x_test1, y_test1, time_step1, epochs=2)
plot_hp(history1, 'loss')
plot_hp(history1, 'accuracy')
prediction1 = prediction_model_plot(model1, x_train1, y_train1, x_test1, y_test1, values, look_back=time_step1)

### Ici Model 2
print('model 2')
time_step2 = 100
x_train2, y_train2, x_test2, y_test2 = reshape_data(time_step2, train_data, test_data)
model2, history2 = deep_network_LSTM('model2', x_train2, y_train2, x_test2, y_test2, time_step2, epochs=2)
plot_hp(history2, 'loss')
plot_hp(history2, 'accuracy')
prediction2 = prediction_model_plot(model2, x_train2, y_train2, x_test2, y_test2, values, look_back=time_step2)

### Ici Model 3
print('model 3')
time_step3 = 100
x_train3, y_train3, x_test3, y_test3 = reshape_data(time_step3, train_data, test_data)
model3, history3 = deep_network_LSTM('model3', x_train3, y_train3, x_test3, y_test3, time_step3, epochs=2)
plot_hp(history3, 'loss')
plot_hp(history3, 'accuracy')
prediction3 = prediction_model_plot(model3, x_train3, y_train3, x_test3, y_test3, values, look_back=time_step3)
