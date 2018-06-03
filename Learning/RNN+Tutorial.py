"""
Algorithm amendments:
1. Potentially change algorithm structure to optomize for predicting values instead of positive/negative
2. Potentially change alg structure to optomize for classifications
3. Change alg to predict on a test set

"""

from __future__ import print_function, division
from generate_crypto_data import normalize, top_n, get_past_day_price
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 100
num_currencies = 2
total_examples = 1440*(num_currencies + 1)
#The number of training examples
train_num = 1000*(num_currencies + 1)
total_series_length = total_examples - train_num
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 1
batch_size = 5
num_layers = 3
num_batches = total_series_length//batch_size//truncated_backprop_length

print("num_batches: ", num_batches)

def generateTestData():
    #Start by creating a random vector of data, half 0s and half 1s
    #x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    x = normalize(top_n(num_currencies))
    x = x[train_num:total_examples, :]

    #Shift the vector by the echo_step. I think the echo_step will be 1 for our main vector
    y = np.roll(x, echo_step)
    #Reset data that is extra to zero instead of just removing it from the x vector
    y[:, 0:echo_step] = 0

    #Flatten into vector
    x = x.flatten()
    y = y.flatten()
    #Run the algorithm through batches in order to increase performance
    #Turn our vector of shape (total_series_length, 1) into a matrix of batches
    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)

def generateTrainData():
    #Start by creating a random vector of data, half 0s and half 1s
    #x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    x = normalize(top_n(num_currencies))
    x = x[0:train_num, :]

    #Shift the vector by the echo_step. I think the echo_step will be 1 for our main vector

    y = np.roll(x, echo_step)
    y[y > 0] = 1
    y[y < 0] = 0
    #Reset data that is extra to zero instead of just removing it from the x vector
    y[:, 0:echo_step] = 0

    #Flatten into vector
    x = x.flatten()
    y = y.flatten()
    #Run the algorithm through batches in order to increase performance
    #Turn our vector of shape (total_series_length, 1) into a matrix of batches
    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)

def parameters():
    #Create tensors to store batch data
    batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
    batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

    init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])

    return batchX_placeholder, batchY_placeholder, init_state

def initialize_weights():
    W2 = tf.Variable(np.random.randn(state_size, num_classes), dtype=tf.float32)
    b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

    return W2, b2

def lstm_forward_prop (W2, b2, init_state, batchX_placeholder, batchY_placeholder):
    state_per_layer_list = tf.unstack(init_state, axis=0)
    rnn_tuple_state = tuple(
        [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
        for idx in range (num_layers)]
    )

    #Unpack Columns of input and label series
    #Split backX data into groups
    inputs_series = tf.split(axis=1, num_or_size_splits=truncated_backprop_length, value=batchX_placeholder)
    labels_series = tf.unstack(batchY_placeholder, axis=1)

    #Create an LSTM cell for each layer in our network
    stacked_rnn = []
    for _ in range(num_layers):
        stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True))

    cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple=True)
    states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, initial_state=rnn_tuple_state)

    logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
    predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

    return logits_series, labels_series, current_state, predictions_series

def cost(logits_series, labels_series):
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels) for logits, labels in zip(logits_series,labels_series)]
    total_loss = tf.reduce_mean(losses)

    return total_loss

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)

def test(x_test, y_test):
    print("Test Set")
    batchX_test = x_test
    batchY_test = y_test

    _current_state = np.zeros((num_layers, 2, batch_size, state_size))

    _current_state, _predictions_series  = sess.run(
        [ current_state, predictions_series],
        feed_dict={
            batchX_placeholder:batchX_test,
            batchY_placeholder:batchY_test
        })

    print(predictions_series)

    print("I don't really know what i'm doing")

def train_model(x_train, x_test, y_train, y_test):
    batchX_placeholder, batchY_placeholder, init_state = parameters()

    W2, b2 = initialize_weights()

    logits_series, labels_series, current_state, predictions_series = lstm_forward_prop(W2, b2, init_state, batchX_placeholder, batchY_placeholder)

    total_loss = cost(logits_series, labels_series)

    train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        plt.ion()
        plt.figure()
        plt.show()
        loss_list = []
        test_lost = []

        for epoch_idx in range(num_epochs):

            _current_state = np.zeros((num_layers, 2, batch_size, state_size))

            print("Epoch", epoch_idx)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * truncated_backprop_length
                end_idx = start_idx + truncated_backprop_length

                batchX = x_train[:,start_idx:end_idx]
                batchY = y_train[:,start_idx:end_idx]

                _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                    [total_loss, train_step, current_state, predictions_series],
                    feed_dict={
                        batchX_placeholder:batchX,
                        batchY_placeholder:batchY,
                        init_state:_current_state,
                    })

                loss_list.append(_total_loss)

                if batch_idx%100 == 0:
                    print("Step",batch_idx, "Loss", _total_loss)
                    plot(loss_list, _predictions_series, batchX, batchY)

def main():
    x_train, y_train = generateTrainData()
    x_test, y_test = generateTestData()

    train_model(x_train, x_test, y_train, y_test)

main()

plt.ioff()
plt.show()
