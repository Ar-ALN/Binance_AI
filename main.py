from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

plt.close("all")

data = pd.read_csv("data.csv", header=None)
data = data.drop(columns=[11])

values = data.values
values = (values - values.min(0)) / (values.max(0) - values.min(0))

# ts = pd.Series(values[:, 1])
# plt.figure()
# plt.plot(ts)
# plt.show()

data = pd.read_csv("data.csv")
train_data = values[:-1000]
test_data = values[-1000:]
training_dataset_length = len(train_data)

train_data = train_data[0:training_dataset_length, :]

# Splitting the data
x_train = []
y_train = []

for i in range(10, len(train_data)):
    x_train.append(train_data[i - 10:i, 0])
    y_train.append(train_data[i, 0])

# Convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data into 3-D array
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(11000):
    EMA = gamma * train_data[ti] + (1 - gamma) * EMA
    train_data[ti] = EMA

# Used for visualization and test purposes
all_mid_data = np.concatenate([train_data, test_data], axis=0)

# Initialising the RNN
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
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
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train, y_train,
                    batch_size=2048, epochs=150,

                    )

# Test data set
test_data = scaled_data[training_dataset_length - 10:, :]

# splitting the x_test and y_test data sets
x_test = []
y_test = features[training_dataset_length:, :]

for i in range(10, len(test_data)):
    x_test.append(test_data[i - 10:i, 0])

# Convert x_test to a numpy array
x_test = np.array(x_test)

# Reshape the data into 3-D array
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# check predicted values
predictions = model.predict(x_test)
# Undo scaling
predictions = scaler.inverse_transform(predictions)

# Calculate RMSE score
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse
