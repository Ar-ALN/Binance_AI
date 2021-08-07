import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import mean_squared_error
#from SumTree import SumTree
import time
import plotly
import copy
import numpy as np
import pandas as pd
import chainer
import chainer.functions as F
import chainer.links as L
from plotly import tools
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, iplot_mpl


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

plt.close("all")

data = pd.read_csv("data.csv")
data.head()
values = []

"""
originalValues = data[4].to_numpy()
values = []
for i in range(len(data) - 1):
    values.append(((originalValues[i] / originalValues[i + 1]) - 1) * 10)
values = np.array(values)
originalValues = originalValues[:len(originalValues) - 1]
"""

# ts = pd.Series(values[:, 1])
# plt.figure()
# plt.plot(ts)
# plt.show()

train_data = data[:-40000]
test_data = data[-10000:]
training_dataset_length = len(train_data)

# test_data = train_data[0:training_dataset_length, :]
# test_data = train_data[0:training_dataset_length, :]

# Used for visualization and test purposes
all_mid_data = np.concatenate([train_data, test_data], axis=0)

prediction_foresee = 60


class PER:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

def plot_loss_reward(total_losses, total_rewards):

    figure = plotly.subplots.make_subplots(rows=1, cols=2, subplot_titles=('loss', 'reward'), print_grid=False)
    figure.append_trace(Scatter(y=total_losses, mode='lines', line=dict(color='skyblue')), 1, 1)
    figure.append_trace(Scatter(y=total_rewards, mode='lines', line=dict(color='orange')), 1, 2)
    figure['layout']['xaxis1'].update(title='epoch')
    figure['layout']['xaxis2'].update(title='epoch')
    figure['layout'].update(height=400, width=900, showlegend=False)
    plt.show()

class Environment:

    def __init__(self,data, history_t=90): # data, how much data the agent uses to predict
        self.data = data
        self.history_t = history_t
        self.reset()

    def reset(self): # Function to initialize the agent's observation
        self.t = 0 # actual time the agent is
        self.done = False
        self.profits = 0
        self.positions = [] # All close prices were when the agent bought act ==1
        self.position_value = 0 # Value of the actual position regarding positions list
        self.history = [0 for _ in range(self.history_t)]
        return [self.position_value] + self.history # Returns a vector of what the agent observe in the environment

    def step(self, act):
        reward = 0

        #actions = 0: stay, 1: buy, 2: sell
        if act == 1: # if he buy
            self.positions.append(self.data.iloc[self.t, :][4]) # Fill the list 'positions' with the actual stock price (we just bought)

        elif act == 2: # if he sells
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0 # initialize profits (not the same as self.profits)
                for p in self.positions: # iterate through self.positions
                    profits += profits + self.data.iloc[self.t, :][4] - p # define the profits equal to diff between actual stock price and the positions price we have bought
                reward += profits # the reward the agent gain is equal to the profits we have made
                self.profits += profits # save the profits into self.profits
                self.positions = [] # reset self.positions because we sold all

        self.t += 1 # Go for the next price stock
        self.position_value = 0
        for p in self.positions: # iterate through self.positions
            self.position_value += (self.data.iloc[self.t, :][4] - p) # if we still have positions (we didn't sell in this iteration) we save the profits we have with into positions_value
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t, :][4] - self.data.iloc[(self.t - 1), :][4])

        # positive reward if we have made benefits, negative if not
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1

        return [self.position_value] + self.history, reward, self.done  # obs, reward, done


# Dueling Double DQN

def train_dddqn(env):
    """ <<< Double DQN -> Dueling Double DQN
    class Q_Network(chainer.Chain):

        def __init__(self, input_size, hidden_size, output_size):
            super(Q_Network, self).__init__(
                fc1 = L.Linear(input_size, hidden_size),
                fc2 = L.Linear(hidden_size, hidden_size),
                fc3 = L.Linear(hidden_size, output_size)
            )

        def __call__(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            y = self.fc3(h)
            return y

        def reset(self):
            self.zerograds()
    === """

    class Q_Network(chainer.Chain):

        def __init__(self, input_size, hidden_size, output_size):
            super(Q_Network, self).__init__(
                fc1=L.Linear(input_size, hidden_size),
                fc2=L.Linear(hidden_size, hidden_size),
                fc3=L.Linear(hidden_size, hidden_size // 2),
                fc4=L.Linear(hidden_size, hidden_size // 2),
                state_value=L.Linear(hidden_size // 2, 1),
                advantage_value=L.Linear(hidden_size // 2, output_size)
            )
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size

        def __call__(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            hs = F.relu(self.fc3(h))
            ha = F.relu(self.fc4(h))
            state_value = self.state_value(hs)
            advantage_value = self.advantage_value(ha)
            advantage_mean = (F.sum(advantage_value, axis=1) / float(self.output_size)).reshape(-1, 1)
            q_value = F.concat([state_value for _ in range(self.output_size)], axis=1) + (
                        advantage_value - F.concat([advantage_mean for _ in range(self.output_size)], axis=1))
            return q_value

        def reset(self):
            self.zerograds()

    """ >>> """

    Q = Q_Network(input_size=env.history_t + 1, hidden_size=100, output_size=3)
    Q_ast = copy.deepcopy(Q)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(Q)

    epoch_num = 50
    step_max = len(env.data) - 1
    memory_size = 200
    batch_size = 50
    epsilon = 1.0
    epsilon_decrease = 1e-3
    epsilon_min = 0.1
    start_reduce_epsilon = 800
    train_freq = 10
    update_q_freq = 20
    gamma = 0.97
    show_log_freq = 5

    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []

    start = time.time()
    for epoch in range(epoch_num):

        pobs = env.reset()
        step = 0
        done = False
        total_reward = 0
        total_loss = 0

        while not done and step < step_max:

            # select act
            pact = np.random.randint(3)
            if np.random.rand() > epsilon:
                pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))
                pact = np.argmax(pact.data)

            # act
            obs, reward, done = env.step(pact)

            # add memory
            memory.append((pobs, pact, reward, obs, done))
            if len(memory) > memory_size:
                memory.pop(0)

            # train or update q
            if len(memory) == memory_size:
                if total_step % train_freq == 0:
                    shuffled_memory = np.random.permutation(memory)
                    memory_idx = range(len(shuffled_memory))
                    for i in memory_idx[::batch_size]:
                        batch = np.array(shuffled_memory[i:i + batch_size])
                        b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                        b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                        b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)

                        q = Q(b_pobs)
                        """ <<< DQN -> Double DQN
                        maxq = np.max(Q_ast(b_obs).data, axis=1)
                        === """
                        indices = np.argmax(q.data, axis=1)
                        maxqs = Q_ast(b_obs).data
                        """ >>> """
                        target = copy.deepcopy(q.data)
                        for j in range(batch_size):
                            """ <<< DQN -> Double DQN
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])
                            === """
                            target[j, b_pact[j]] = b_reward[j] + gamma * maxqs[j, indices[j]] * (not b_done[j])
                            """ >>> """
                        Q.reset()
                        loss = F.mean_squared_error(q, target)
                        total_loss += loss.data
                        loss.backward()
                        optimizer.update()

                if total_step % update_q_freq == 0:
                    Q_ast = copy.deepcopy(Q)

            # epsilon
            if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                epsilon -= epsilon_decrease

            # next step
            total_reward += reward
            pobs = obs
            step += 1
            total_step += 1

        total_rewards.append(total_reward)
        total_losses.append(total_loss)

        if (epoch + 1) % show_log_freq == 0:
            log_reward = sum(total_rewards[((epoch + 1) - show_log_freq):]) / show_log_freq
            log_loss = sum(total_losses[((epoch + 1) - show_log_freq):]) / show_log_freq
            elapsed_time = time.time() - start
            print('\t'.join(map(str, [epoch + 1, epsilon, total_step, log_reward, log_loss, elapsed_time])))
            start = time.time()

    return Q, total_losses, total_rewards


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset) - 1 - time_step - prediction_foresee):
        a = dataset[i:i + time_step]
        dataX.append(a)
        # Get max of the 40 next values
        max_price = originalValues[i + time_step: i + time_step + prediction_foresee].max()
        current_price = originalValues[i]
        prediction = 1 if max_price > current_price else 0
        prediction = .5 if current_price * 1.005 > max_price > current_price * 0.995 else prediction
        dataY.append(prediction)
    return np.array(dataX), np.array(dataY)


def deep_network_LSTM(name_model, x_train, y_train, x_test, y_test, shape, activation_function='sigmoid', opt='adam',
                      epochs=100, batch_size=64):
    # Initialising the RNN
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(shape, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
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

    return (model, history)


def prediction_model_plot(model, x_train, y_train, x_test, y_test, look_back):
    global originalValues

    ### Lets Do the prediction and check performance metrics
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    ##Transformback to original form
    math.sqrt(mean_squared_error(y_train, train_predict))

    ### Test Data RMSE
    math.sqrt(mean_squared_error(y_test, test_predict))

    ### Plotting
    # shift train predictions for plotting
    # values = np.reshape(values, (len(values), 1))
    values = np.reshape(originalValues, (len(originalValues), 1))
    print(values.shape)
    x = np.arange(len(values))
    print(x)

    trainPredictPlot = np.empty_like(values)
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(values)
    testPredictPlot[:] = np.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1 + 80:len(values) - 1] = test_predict
    # plot baseline and predictions
    fig = plt.figure(figsize=(12, 1), dpi=1200)

    axe1 = fig.add_subplot(111)
    axe1.plot(x, values, linewidth=0.033333)
    axe1.set_ylabel('values')

    axe2 = axe1.twinx()
    axe2.set_ylabel('trainPredictPlot')
    axe2.hist(trainPredictPlot, 100, edgecolor="k",color='yellow')


    axe3 = axe1.twinx()
    axe3.set_ylabel('testPredictPlot')
    axe3.hist(testPredictPlot, 100, edgecolor="k", color='green')
    plt.plot(trainPredictPlot, linewidth=0.033333)
    plt.plot(testPredictPlot, linewidth=0.033333)
    #plt.axis([x_min, x_max, y_min, y_max])  # permet de zoomer sur une partie de la courbe
    plt.show()


def plot_hp(model_plot, hyperparameter,name_model):
    train_hp = hyperparameter
    validation_hp = 'val_' + hyperparameter
    model = model_plot
    plt.figure(figsize=(12, 6), dpi=80)
    plt.plot(model.history[train_hp], label='train')
    plt.plot(model.history[validation_hp], alpha=0.7, label='validation')
    plt.xlabel(hyperparameter)
    plt.ylabel('Loss')
    plt.legend()
    plt.title(hyperparameter + ' vs Epochs for ' + name_model, size=25)
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


####### Nous allons ici effectuer les test avec un apprentissage par renforcement
####### Dueling Double Deep Q networks DDDQN

### Ici Model 1

with tf.device('/GPU:0'):
    Q, total_losses, total_rewards = train_dddqn(Environment(train_data))
    plot_loss_reward(total_losses, total_rewards)
    #plot_train_test_by_q(Environment1(train), Environment1(test), Q, 'Dueling Double DQN')

"""
model1, history1 = deep_network_LSTM('model1', x_train, y_train, x_test, y_test, x_train[0].shape, epochs=140)
plot_hp(history1, 'loss')
plot_hp(history1, 'accuracy')
prediction_model_plot(model1, x_train, y_train, x_test, y_test, data, time_step=time_step)
"""