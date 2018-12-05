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
n_hidden_units = 128  # neurons in hidden layer#可以换成1280等其它隐藏层
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
    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)  # 初始化全零 state
    # outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    # MultiRNNCel
    ####################################################################################
    # **步骤2：定义一层 LSTM_cell，只需要说明 hidden_size即隐层神经元大小, 它会自动匹配输入的 X 的维度,
    # lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=n_hidden_units)  #cell可以选择lstm也可以用gru
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden_units, forget_bias=1.0, state_is_tuple=True)

    # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)

    # **步骤4：调用 MultiRNNCell 来实现多层 LSTM
    mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)

    # **步骤5：用全零来初始化state
    init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
    print('init_state',init_state)
    '''
    lstmcell:
    init_state (
    LSTMStateTuple(c=<tf.Tensor 'MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros:0' shape=(128, num_hidden=128) dtype=float32>, h=<tf.Tensor 'MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1:0' shape=(128, 128) dtype=float32>), 
    LSTMStateTuple(c=<tf.Tensor 'MultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros:0' shape=(128, 128) dtype=float32>, h=<tf.Tensor 'MultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1:0' shape=(128, 128) dtype=float32>), 
    LSTMStateTuple(c=<tf.Tensor 'MultiRNNCellZeroState/DropoutWrapperZeroState_2/BasicLSTMCellZeroState/zeros:0' shape=(128, 128) dtype=float32>, h=<tf.Tensor 'MultiRNNCellZeroState/DropoutWrapperZeroState_2/BasicLSTMCellZeroState/zeros_1:0' shape=(128, 128) dtype=float32>))


    GRUcell:
    init_state (
    <tf.Tensor 'MultiRNNCellZeroState/DropoutWrapperZeroState/GRUCellZeroState/zeros:0' shape=(128, 12) dtype=float32>,
     <tf.Tensor 'MultiRNNCellZeroState/DropoutWrapperZeroState_1/GRUCellZeroState/zeros:0' shape=(128, 12) dtype=float32>, 
     <tf.Tensor 'MultiRNNCellZeroState/DropoutWrapperZeroState_2/GRUCellZeroState/zeros:0' shape=(128, 12) dtype=float32>)
    '''

    # **步骤6：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
    # ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size]
    # ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
    # ** state.shape = [layer_num, 2, batch_size, hidden_size],
    # ** 或者，可以取 h_state = state[-1][1] 作为最后输出
    # ** 最后输出维度是 [batch_size, hidden_size]
    outputs, final_state0 = tf.nn.dynamic_rnn(mlstm_cell, inputs=X_in, initial_state=init_state, time_major=False)
    #mlstm_cell，initial_state=init_state这两者要对应，num_hidden与initial_state第二维度相同
    print('outputs',outputs)#outputs Tensor("rnn/transpose_1:0", shape=(128, 28, 128), dtype=float32)


    #final_state0估计是（layer_num,(cell_state,hidden_state)）

    print('final_state0',final_state0)
    '''
    每一层最后时刻的c和h组成LSTMStateTuple(),三层就有三个
       (LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_3:0' shape=(128, 128) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_4:0' shape=(128, 128) dtype=float32>),
        LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_5:0' shape=(128, 128) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_6:0' shape=(128, 128) dtype=float32>), 
        LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_7:0' shape=(128, 128) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_8:0' shape=(128, 128) dtype=float32>))
'''
    print('final_state0[0]',final_state0[0])
    '''
    final_state0[0] 
    LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_3:0' shape=(128, 128) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_4:0' shape=(128, 128) dtype=float32>)

    '''

    #这两个h_state都是最后一层的，最后时刻的state
    # final_state = outputs[:, -1, :]  # 或者
    final_state = final_state0[-1][1]#取最后一层的LSTMStateTuple中的h,lstm时正确的，gru时状态都不一样，不能这样写

    print('outputs.shape',outputs.shape)#outputs.shape (128, 28, 128)

    print('final_state',type(final_state))
    print('final_state[0]',final_state[0][0].shape)
    print('final_state[2]',final_state[2][0].shape)

    # 方法1
    results = tf.matmul(final_state, weights['out']) + biases['out']


    # # 方法2
    # # hidden layer for output as the final results
    # ####################################################################################
    # # 把outputs变成列表[(batch,outputs)...]*steps
    # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # # 选取最后一个output
    # results = tf.matmul(outputs[-1], weights["out"]) + biases["out"]

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

# step0,Minibatch Loss=2.2071,Training Accuracy=0.273
# step20,Minibatch Loss=1.8815,Training Accuracy=0.594

# step0,Minibatch Loss=2.2090,Training Accuracy=0.281
# step20,Minibatch Loss=1.9054,Training Accuracy=0.562