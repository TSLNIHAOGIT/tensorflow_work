import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 加载mnist_inference.py中定义的常量和前向传播的函数
from tensorflow_work.mnist_save import minist_inference
import ssl

#全局取消证书验证
ssl._create_default_https_context = ssl._create_unverified_context


#配置神经网络参数
BATCH_SIZE = 100  # 一个训练 batch 中的训练数据个数。数字越小时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降
LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 50000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

#模型保存的路径和文件名
MODEL_SAVE_PATH='model'
MODEL_NAME='model.ckpt'


# # 训练模型的过程
# def train(mnist):
#     x = tf.placeholder(tf.float32, [None, minist_inference.INPUT_NODE], name='x-input')
#     y_ = tf.placeholder(tf.float32, [None, minist_inference.OUTPUT_NODE], name='y-input')
#
#     regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
#     #直接使用mnist_inference.py中定义的前向传播过程
#     y=minist_inference.inference(x,regularizer)
#
#     # 定义存储训练轮数的变量。这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量（trainable=False）。
#     # 在使用 Tensorflow 训练神经网络时，一般会将代表训练轮数的变量指定为不可训练的参数
#     global_step = tf.Variable(0, trainable=False)
#
#     # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。在第4章中介绍过给定训练轮数的变量可以加快训练早期变量的更新速度
#     variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
#
#     # 在所有代表神经网络参数的变量上使用滑动平均。其他辅助变量（比如 global_step ）就不需要了。
#     # tf.trainable_variables 返回的就是图上集合 GraphKeys.TRAINABLE_VARIABLES 中的元素。
#     # 这个集合的元素就是所有没有指定 trainable=False 的参数
#     variables_averages_op = variable_averages.apply(tf.trainable_variables())
#
#
#     # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数。这里使用了 Tensorflow 中提供的 sparse_softmax_cross_entropy_with_logits
#     # 函数来计算交叉熵。当分类问题只用一个正确答案时，可以使用这个函数来加速交叉熵的计算。 MNIST 问题的图片中只包含 0~9 中的一个数字，
#     # 所以可以使用这个函数来计算交叉熵损失。这个函数的第一个参数是神经网络不包含 softmax 层的前向传播结果，第二个是训练数据的正确答案。
#     # 因为标准答案是一个长度为 10 的一个数组，而该函数需要提供的是一个正确答案的数字，所以需要使用 tf.argmax 函数来得到正确答案对应的类别编号。
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
#     # 计算在当前 batch 中所有样例的交叉熵品均值
#     cross_entropy_mean = tf.reduce_mean(cross_entropy)
#     loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
#
#     # 设置指数衰减的学习率
#     learning_rate = tf.train.exponential_decay(
#         LEARNING_RATE_BASE,  # 基础学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减
#         global_step,  # 当前迭代的轮数
#         mnist.train.num_examples / BATCH_SIZE,  # 过完所有的训练数据需要的迭代次数
#         LEARNING_RATE_DECAY)  # 学习率衰减速度
#
#     # 使用 tf.train.GradientDescentOptimizer 优化算法来优化损失函数。注意这里损失函数包含了交叉熵损失和L2正则化损失
#     train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#
#     # 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，又要更新每一个参数的轮滑平均值。
#     # 为了一次完成多个操作，Tensorflow 提供了 tf.control_dependencies 和 tf.group 两种机制。下面两行程序和
#     # tf.group(train_step, variables_averages_op) 是等价的
#     with tf.control_dependencies([train_step, variables_averages_op]):
#         train_op = tf.no_op(name='train')
#
#
#     #初始化tensorflow持久类
#     saver=tf.train.Saver()
#     with tf.Session() as sess:
#         tf.initialize_all_variables().run()
#         #在训练过程中不在测试模型在验证数据上表现，验证和测试过程会有一个独立程序完成
#         for i in range(TRAINING_STEPS):
#             xs, ys = mnist.train.next_batch(BATCH_SIZE)
#             _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
#             #每1000轮保存一次模型
#             if i % 1000 == 0:
#                 #输出当前的训练情况。这里只输出了模型在当前batch上的损失函数大小。通过损失函数大小大概可以了解训练情况。
#                 #在你验证集上的正确率信息会有一个单独程序来生成。
#                 print('after {} training stet(s) ,loss on training batch is {}'.format(step,loss_value))
#                 #保存当前模型，注意这里给出了global_step参数，这样，可以让每个被保存模型的文件末尾加上训练轮数，如model.ckpt-1000,表示训练了1000轮之后的模型
#                 saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)





# 2. 定义训练过程。

def train(mnist):
    # 定义输入输出placeholder。
    x = tf.placeholder(tf.float32, [None, minist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, minist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    print('regularizer',regularizer)
    y = minist_inference.inference(x, regularizer)#(?, 10)
    print('y.shape',y.shape)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            print('xs',xs.shape)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)



def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    train(mnist)
if __name__=='__main__':
    tf.app.run()