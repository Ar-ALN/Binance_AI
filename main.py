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
#data = data.drop(columns=[4])
print(data)
values = data[[4]]
print(values)

scaler=MinMaxScaler(feature_range=(0,1))

#met tout entre 0 et 1
values = scaler.fit_transform(np.array(values).reshape(-1,1))


# ts = pd.Series(values[:, 1])
# plt.figure()
# plt.plot(ts)
# plt.show()

train_data = values[:-10000]
test_data = values[-10000:]
training_dataset_length = len(train_data)

train_data = train_data[0:training_dataset_length, :]

train_data = train_data[0:]
test_data = test_data[0:]

# Splitting the data
# x_train = []
# y_train = []

# for i in range(10, len(train_data)):
# x_train.append(train_data[i - 10:i, 0])
# y_train.append(train_data[i, 0])

# Convert to numpy arrays
# x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data into 3-D array
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(11000):
    EMA = gamma * train_data[ti] + (1 - gamma) * EMA
    train_data[ti] = EMA

# Used for visualization and test purposes
all_mid_data = np.concatenate([train_data, test_data], axis=0)

# reshape le tout
x_train =x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

def deep_network_LSTM(name_model,x_train, y_train, activation_function = 'sigmoid', opt = 'adam', epochs = 100, batch_size = 64):
    # Initialising the RNN
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(100, 1)))
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
    model.compile(optimizer= opt, loss='mean_squared_error',metrics=['accuracy'])
    history =model.fit(x_train, y_train,validation_data=(x_test,y_test),
                        batch_size=batch_size, epochs=epochs, verbose = 1

                        )
    model.save(name_model)
    print("Saved model to disk")

    return(history)

def prediction_model_plot(model,x_train,x_test, values):
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
    look_back = 100
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
    plt.show()

def plot_hp(model_plot,hyperparameter):

    train_hp = hyperparameter
    validation_hp = 'val_' + hyperparameter
    model = model_plot
    plt.figure(figsize=(8, 6))
    plt.plot(model.history[train_hp], label='train')
    plt.plot(model.history[validation_hp], alpha=0.7, label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Epochs for Model 1', size=25)
    plt.grid()

model1 = deep_network_LSTM('model1',x_train, y_train,epochs=1)
plot_hp(model1,'loss')
plot_hp(model1,'accuracy')
prediction1 = prediction_model_plot(model1,x_train,x_test, values)











