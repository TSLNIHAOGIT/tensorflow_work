import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow_work.mnist_cnn_other import mnist_inference_other
import numpy as np
import ssl
# #全局取消证书验证
ssl._create_default_https_context = ssl._create_unverified_context



# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE =0.8 #原来是0.001
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001 #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000#原来是10000
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率

#模型保存的路径和文件名
MODEL_SAVE_PATH = 'model'
MODEL_SAVE_NAME = 'mnist_cnn_model.ckpt'

def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_inference_other.IMAGE_SIZE, mnist_inference_other.IMAGE_SIZE,
                                    mnist_inference_other.NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference_other.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    #直接使用mnist_inference_other.py中定义的前向传播结果
    y = mnist_inference_other.inference(x, train=False, regularizer=regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 生成一个滑动平均的类，并在所有变量上使用滑动平均
    variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variables_averages.apply(tf.trainable_variables())

    # 计算交叉熵及当前batch中的所有样例的交叉熵平均值,并求出损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 定义指数衰减式的学习率以及训练过程
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    #初始化TF持久化类
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, [BATCH_SIZE, mnist_inference_other.IMAGE_SIZE, mnist_inference_other.IMAGE_SIZE,
                                    mnist_inference_other.NUM_CHANNELS])
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i%1000 == 0:
                print('After %d training steps, loss on training batch is %g' % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_SAVE_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("../mnist_cnn/MNIST_data", one_hot=True)
    train(mnist)

if __name__ =='__main__':
    tf.app.run()