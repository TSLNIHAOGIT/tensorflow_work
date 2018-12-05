import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
tf.set_random_seed(1) # set random seed

# 导入数据
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001 # learning rate
training_iters = 100000 # train step 上限
batch_size = 128
n_inputs = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # time steps
n_hidden_units = 128 # neurons in hidden layer
n_classes = 10 # MNIST classes (0-9 digits)

# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 初始值的定义
weights = {
# shape (28, 128)
'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
# shape (128, 10)
'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
# shape (128, )
'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
# shape (10, )
'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X, weights, biases):
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']#相当于先连一个全链接层
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # 使用 basic LSTM Cell.
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # 初始化全零 state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    '''
    此时，得到的outputs就是time_steps步里所有的输出。它的形状为(batch_size, time_steps, cell.output_size)。
    final_state是最后一步的隐状态，它的形状为(batch_size, cell.state_size)。
    
    
    代码中因为用了BasicLSTMCells，
    所以返回结果是一个tuple shape like (cell_state[batch, cell.output_size],hidden_state[batch, cell.output_size])，
    第一个是cell的state输出结果，后面的是隐层的输出结果，都是[50, 100]。之所以又一个cell_state可以参考lstm的结构。
    如果是用了BasicRNNCells的话，返回结果是一个[50, 100]的结果，也就是最后一个单元中隐层神经元的输出
    '''
    print('type(outputs)',type(outputs),type(final_state))
    '''
    type(outputs) 
    <class 'tensorflow.python.framework.ops.Tensor'> 
    <class 'tensorflow.python.ops.rnn_cell_impl.LSTMStateTuple'>

    '''
    print('outputs[i].shape',outputs[0].shape,outputs[1].shape,outputs.shape)
    print('final_state[i].shape',final_state[0].shape,final_state[1].shape)
    '''
    outputs.shape (128, 28, 128)
    final_state[i].shape (128, 128) (128, 128)
    '''

    # 使用 MultiRNNCell.


   #方法1直接用最后一个隐状态的输出，方法从所有状态输出中提取最后一个状态
    # final_state[0]是cell_state;final_state[1]才是需要的hidden_state
    #方法1
    results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # 方法2，把 outputs 变成 列表 [(batch, outputs)..] * steps
    #outputs输出是128维度向量，后面又接了全连接层
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    print('outputs2 shape ',np.array(outputs).shape)#(28,)是一维列表
    results = tf.matmul(outputs[-1], weights['out']) + biases['out'] #选取最后一个 output
     # outputs[-1].shape 128*128; weights['out']128*10

    return results
pred = RNN(x, weights, biases)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( y,pred))
# train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# #labels为实际值，标签；；；logits是计算出来的预测值。
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# init= tf.initialize_all_variables() # tf 马上就要废弃这种写法
# 替换成下面的写法:
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        print('batch_ys.shape',batch_ys.shape,batch_ys)#batch_ys.shape (128, 10)
        '''
        [[0. 0. 0. ... 0. 0. 1.]
         [0. 0. 0. ... 0. 0. 1.]
         [0. 0. 0. ... 1. 0. 0.]
         ...
         [0. 0. 1. ... 0. 0. 0.]
         [0. 0. 0. ... 0. 1. 0.]
         [0. 0. 0. ... 0. 0. 0.]]
        '''
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
        x: batch_xs,
        y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
            }))
        step += 1