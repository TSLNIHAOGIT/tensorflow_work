#coding:utf-8
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 加载mnist_inference.py中定义的常量和前向传播的函数
from tensorflow_work.mnist_cnn import minist_inference
import numpy as np
import ssl
#全局取消证书验证
ssl._create_default_https_context = ssl._create_unverified_context


#配置神经网络参数
BATCH_SIZE = 100  # 一个训练 batch 中的训练数据个数。数字越小时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降
LEARNING_RATE_BASE = 0.01 # 基础的学习率,原来0.8
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARAZTION_RATE = 0.0001#正则化的权重
TRAIN_STEP = 30000
MOVING_AVERAGE_DECAY = 0.99#滑动平均率


#模型保存的路径和文件名
MODEL_SAVE_PATH='model'
MODEL_NAME='model.ckpt'


def train(mnist):
    x = tf.placeholder(tf.float32,[
    BATCH_SIZE,
    minist_inference.IMAGE_SIZE,
    minist_inference.IMAGE_SIZE,
    minist_inference.NUM_CHANNELS],
    name="x-input")

    y_ = tf.placeholder(tf.float32,[None,minist_inference.OUTPUT_NIDE],name="y-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)


    y = minist_inference.inference(x,True,regularizer)
    print('y.shape',y.shape)

    global_step = tf.Variable(0,trainable=False)#设置global_step为不可训练数值，在训练过程中它不进行相应的更新

    #对w,b进行滑动平均操作
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)#对滑动平均函数进行输入滑动平均率以及步数
    variable_average_op = variable_average.apply(tf.trainable_variables())#对所有可训练的参数进行滑动平均操作，不断对新变量进行移动平均，同时要使用上一次的移动平均值

    #计算损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_,logits = y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean+tf.add_n(tf.get_collection("losses"))#这里计算collection里的所有的和。之前把w正则化的值放在了collection里

    #对 学习率 进行指数衰减
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)

    #定义训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)#每当进行一次训练global_step会加1

    #一次进行多个操作,既进行反向传播更新神经网络中的参数，又更新每一个参数的滑动平均值(滑动平均是影子操作)
    with tf.control_dependencies([train_step,variable_average_op]):
        train_op = tf.no_op(name="train")


    #保存操作
    saver = tf.train.Saver()

    #启动程序

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAIN_STEP):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            reshapeed_xs = np.reshape(xs,(BATCH_SIZE,
            minist_inference.IMAGE_SIZE,
            minist_inference.IMAGE_SIZE,
            minist_inference.NUM_CHANNELS
            ))
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:reshapeed_xs,y_:ys})

            #每1000轮保存一次模型
            if i%1000 ==0:
                print ("step ",step,"   ","loss  ",loss_value)
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step = global_step)



def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    train(mnist)
if __name__=='__main__':
    #处理flag解析，然后执行main函数
    tf.app.run()