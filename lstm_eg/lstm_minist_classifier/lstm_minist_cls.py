# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

tf.set_random_seed(1)  # set random seed

# 导入数据
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)



# hyperparameters
lr = 0.001  # learning rate
training_iters = 100000  # train step 上限
batch_size = 128
n_inputs = 28  # MNIST data input(img shape:28*28)
n_steps = 28  # time steps
n_hidden_units = 128  # neurons in hidden layer
n_classes = 10  # MNIST classes(0-9 digits)
keep_prob = tf.placeholder(tf.float32)
# LSTM layer 的层数
layer_num = 2

# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 对weights biases初始值的定义
weights = {
    # shape(28,128)
    "in": tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape(128,10)
    "out": tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape(128,)
    "in": tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    # shape(10,)
    "out": tf.Variable(tf.constant(0.1, shape=[n_classes]))
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ###################################################################################
    # 原始的X是3维数据，我们需要把它变成2维数据才能使用weights的矩阵乘法
    # X==>(128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # X_in = W*X+b
    X_in = tf.matmul(X, weights["in"]) + biases["in"]
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # # cell
    # ####################################################################################
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)  # 初始化全零 state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    ####################################################################################
    # 把outputs变成列表[(batch,outputs)...]*steps
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # 选取最后一个output
    results = tf.matmul(outputs[-1], weights["out"]) + biases["out"]

    return results


logits = RNN(x, weights, biases)
pred = tf.nn.softmax(logits)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# Evaluate mode
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables
init = tf.global_variables_initializer()

# start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    step = 0

    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys,keep_prob:0.9})
        if step % 20 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_xs, y: batch_ys,keep_prob:0.9})
            print("step" + str(step) + ",Minibatch Loss=" + "{:.4f}".format(loss)
                  + ",Training Accuracy=" + "{:.3f}".format(acc))
        step += 1

    print("Optimization Finished!")

    # calculate accuracy for 128 mnist test image
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_inputs))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,keep_prob:0.9}))


#Testing Accuracy: 0.984375