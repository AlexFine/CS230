"""
Algorithm amendments:
1. Potentially change algorithm structure to optomize for predicting values instead of positive/negative
2. Potentially change alg structure to optomize for classifications
3. Change alg to predict on a test set

"""

from __future__ import print_function, division
from generate_crypto_data import read_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 1000
num_currencies = 100
data_len = 3000
train_len = 2900
truncated_backprop_length = 30 #Hyper-parameter
state_size = 5 #Hyper-parameter
num_classes = 2
echo_step = 1
batch_size = 5 #Hyper-parameter
num_layers = 2 #Hyper-parameter
learning_rate = 0.01 #Hyper-parameter
beta1 = .9 #Hyper-parameter
beta2 = .999 #Hyper-parameter
num_inputs = 5

total_examples = data_len*(num_currencies + 1)
#The number of training examples
train_num = train_len*(num_currencies + 1)
total_series_length = train_num
num_batches = total_series_length//batch_size//truncated_backprop_length

test_num = (data_len - train_len)*(num_currencies + 1)
total_test_series_length = test_num
num_test_batches = total_test_series_length//batch_size//truncated_backprop_length

print("num_batches: ", num_batches)

def generateTestData():
    #Start by creating a random vector of data, half 0s and half 1s
    #x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    x = read_data("data/normalized_price_data/")
    x = x[train_len:data_len, :]

    #Set Y output vector
    y = x[1:, :]
    zero = np.zeros((1, len(x[1])))
    zero = zero.astype(float)
    y = np.vstack((y, zero))

    #Shift the vector by the echo_step. I think the echo_step will be 1 for our main vector
    #y = np.roll(x, echo_step)
    y[y > 0] = 1
    y[y < 0] = 0

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
    x = read_data("data/normalized_price_data/")
    #x = x[0:train_len, :]

    #Set Y output vector
    y = x[1:, 0]
    print(y.shape)
    y = np.append(y, 0)

    #Shift the vector by the echo_step. I think the echo_step will be 1 for our main vector
    #y = np.roll(x, echo_step)
    y[y > 0] = 1
    y[y < 0] = 0

    #Flatten into vector
    #x = x.flatten()
    y = y.flatten()
    #Run the algorithm through batches in order to increase performance
    #Turn our vector of shape (total_series_length, 1) into a matrix of batches
    print("X Shape: ", x.shape)
    x = x.reshape((batch_size, -1, num_inputs))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    print("X Shape: ", x.shape)
    print("Y Shape: ", y.shape)
    return (x, y)

def parameters():
    #Create tensors to store batch data
    #Edit to accomodate matrices
    batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length, num_inputs])
    batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

    init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])

    return batchX_placeholder, batchY_placeholder, init_state

def initialize_weights():
    W2 = tf.Variable(np.random.randn(state_size, num_classes), dtype=tf.float32)
    b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)
    #b2 = tf.Print(b2, [b2], "b2 Tensor")

    return W2, b2

def lstm_forward_prop (W2, b2, init_state, batchX_placeholder, batchY_placeholder):
    state_per_layer_list = tf.unstack(init_state, axis=0)
    rnn_tuple_state = tuple(
        [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
        for idx in range (num_layers)]
    )

    #Unpack Columns of input and label series
    #Split backX data into groups
    #Remove these lines
    #inputs_series = tf.split(axis=1, num_or_size_splits=truncated_backprop_length, value=batchX_placeholder)
    #labels_series = tf.unstack(batchY_placeholder, axis=1)

    #Create an LSTM cell for each layer in our network
    stacked_rnn = []
    for _ in range(num_layers):
        stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True))

    cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple=True)
    #NEW
    states_series, current_state = tf.nn.dynamic_rnn(cell, batchX_placeholder, initial_state=rnn_tuple_state)
    states_series = tf.reshape(states_series, [-1, state_size])

    #Logits shape [batch_size*truncated_backprop_length, num_classes]
    logits = tf.matmul(states_series, W2) + b2 #Broadcasted addition
    #Labels shape [batch_size*truncated_backprop_length]
    labels = tf.reshape(batchY_placeholder, [-1])

    #This code, and the prediction series, are really just for graphing
    #The cost function does the softmax automatically
    logits_series = tf.unstack(tf.reshape(logits, [batch_size, truncated_backprop_length, num_classes]), axis=1)
    predictions_series = [tf.nn.softmax(logit) for logit in logits_series]

    return logits, labels, current_state, predictions_series

def cost(logits, labels):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels)
    total_loss = tf.reduce_mean(losses)

    return total_loss

def plot(loss_list, predictions_series, batchX, batchY, accuracy_list):
    #plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)
    plt.plot(accuracy_list)

    """for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")"""

    plt.draw()
    plt.pause(0.0001)

def train_model(x_train, x_test, y_train, y_test):
    batchX_placeholder, batchY_placeholder, init_state = parameters()

    W2, b2 = initialize_weights()

    logits, labels, current_state, predictions_series = lstm_forward_prop(W2, b2, init_state, batchX_placeholder, batchY_placeholder)

    #Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits,  axis=1), tf.cast(labels, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    total_loss = cost(logits, labels)

    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta1, beta2 = beta2).minimize(total_loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, "models/81/model.ckpt")

        plt.ion()
        plt.figure()
        plt.show()
        loss_list = []
        accuracy_list = []
        avg_accuracy = []
        test_loss = []
        test_accuracy = []

        for epoch_idx in range(num_epochs):

            _current_state = np.zeros((num_layers, 2, batch_size, state_size))

            print("Epoch", epoch_idx)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * truncated_backprop_length
                end_idx = start_idx + truncated_backprop_length

                batchX = x_train[:, start_idx:end_idx, :]
                batchY = y_train[:,start_idx:end_idx]

                _total_loss, _train_step, _current_state, _predictions_series, _accuracy = sess.run(
                    [total_loss, train_step, current_state, predictions_series, accuracy],
                    feed_dict={
                        batchX_placeholder:batchX,
                        batchY_placeholder:batchY,
                        init_state:_current_state
                    })

                avg_accuracy.append(_accuracy)

                if batch_idx%100 == 0:
                    accuracy_list.append(_accuracy)
                    loss_list.append(_total_loss)
                    print("Step",batch_idx, "Loss", _total_loss)
                    print("Accuracy: ", np.sum(avg_accuracy)/len(avg_accuracy) *100, "%")
                    plot(loss_list, _predictions_series, batchX, batchY, accuracy_list)


            """for batch_idx in range(num_test_batches):
                start_idx = batch_idx * truncated_backprop_length
                end_idx = start_idx + truncated_backprop_length

                batchX = x_test[:,start_idx:end_idx]
                batchY = y_test[:,start_idx:end_idx]

                _test_loss, _accuracy = sess.run(
                    [total_loss, accuracy],
                    feed_dict={
                        batchX_placeholder:batchX,
                        batchY_placeholder:batchY,
                        init_state:_current_state
                    })

                test_loss.append(_test_loss)
                test_accuracy.append(_accuracy)

            print("Test Loss: ", np.sum(test_loss)/len(test_loss))
            print("Test Accuracy: ", np.sum(test_accuracy)/len(test_accuracy) * 100, "%")"""
            if epoch_idx % 10 == 0:
                #path_name = "models/" + str(int(np.sum(test_accuracy)/len(test_accuracy) * 100)) + "/model.ckpt"
                path_name = "models/" + str(int(np.sum(avg_accuracy)/len(avg_accuracy) * 100)) + "/model.ckpt"
                save_path = saver.save(sess, path_name)
                print("Model saved in file: %s" % save_path)

            avg_accuracy = []

def guess(x_train, x_test, y_train, y_test):
    y_train = y_train.flatten()
    accuracy_train = np.sum(y_train)/len(y_train)
    print("Random 1 Hour Guessing Accuracy On Training: ", accuracy_train*100, "%")

    y_test = y_test.flatten()
    accuracy_test = np.sum(y_test)/len(y_test)
    print("Random 1 Hour Guessing Accuracy On Testing: ", accuracy_test*100, "%")
    print(len(y_train))
    print(len(y_test))

    return 0

def main():
    x_train, y_train = generateTrainData()
    x_test, y_test = generateTestData()

    guess(x_train, x_test, y_train, y_test)

    train_model(x_train, x_test, y_train, y_test)

main()

plt.ioff()
plt.show()
