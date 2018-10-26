import tensorflow as tf
#全局取消证书验证
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np





#配置神经网络参数
BATCH_SIZE = 100  # 一个训练 batch 中的训练数据个数。数字越小时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降
LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARAZTION_RATE = 0.0001#正则化的权重
TRAIN_STEP = 30000
MOVING_AVERAGE_DECAY = 0.99#滑动平均率


# def inference(input_tensor, train, regularizer):
#     # 输入：28×28×1，输出：28×28×32
#     with tf.variable_scope('layer1-conv1'):
#         conv1_weights = tf.get_variable('weights', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
#                                                    initializer = tf.truncated_normal_initializer(stddev=0.1))
#         conv1_biases = tf.get_variable('biases', [CONV1_DEEP], initializer = tf.constant_initializer(0.0))
#         # 使用尺寸为5，深度为32的过滤器，步长为1，使用全0填充
#         conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
#         relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
#
#     # 输入：28×28×32，输出：14×14×32
#     with tf.name_scope('layer2-pool1'):
#         pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
#     # 输入：14×14×32，输出：14×14×64
#     with tf.variable_scope('layer3-conv2'):
#         conv2_weights = tf.get_variable('weights', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
#                                                    initializer = tf.truncated_normal_initializer(stddev=0.1))
#         conv2_biases = tf.get_variable('biases', [CONV2_DEEP], initializer = tf.constant_initializer(0.0))
#
#         # 使用尺寸为5，深度为64的过滤器，步长为1，使用全0填充
#         conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
#         relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
#
#     # 输入：14×14×64，输出：7×7×64
#     with tf.name_scope('layer4-pool2'):
#         pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#         # 将7×7×64的矩阵转换成一个向量，因为每一层神经网络的输入输出都为一个batch矩阵，所以这里得到的维度
#         # 也包含了一个batch中数据的个数（batch×7×7×64 --> batch×vector)
#         pool_shape = pool2.get_shape().as_list()
#         # pool_shape[0]为一个batch中数据的个数
#         nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
#         # 通过tf.reshape函数将第四层的输出变成一个batch的向量
#         reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
#
#     with tf.variable_scope('layer5-fc1'):
#         fc1_weights = tf.get_variable('weights', [nodes, FC_SIZE], initializer = tf.truncated_normal_initializer(
#             stddev=0.1))
#         # 只有全连接层的权重需要加入正则化
#         if regularizer != None:
#             tf.add_to_collection('losses', regularizer(fc1_weights))
#             fc1_biases = tf.get_variable('biases', [FC_SIZE], initializer = tf.constant_initializer(0.1))
#             fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
#             if train:
#                 fc1 = tf.nn.dropout(fc1, 0.5)
#
#     with tf.variable_scope('layer6-fc2'):
#         fc2_weights = tf.get_variable('weights', [FC_SIZE,
#                                                   NUM_LABELS], initializer = tf.truncated_normal_initializer(
#             stddev=0.1))
#         if regularizer != None:
#             tf.add_to_collection('losses', regularizer(fc2_weights))
#             fc2_biases = tf.get_variable('biases', [NUM_LABELS], initializer = tf.constant_initializer(0.1))
#             logit = tf.matmul(fc1, fc2_weights) + fc2_biases
#
#     return logit


#定义神经网络结构相关的参数
INPURT_NODE = 784
OUTPUT_NIDE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一层卷积层的尺寸和深度
CONV1_SIZE = 5
CONV1_DEEP = 32

#第二层的卷积层的尺寸和深度
CONV2_SIZE = 5
CONV2_DEEP = 64

#全连接层的节点个数
FC_SIZE = 512
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
'''
开始报错函数不能转为张量是因为中间纬度不对，tf.reshape 改为reshaped就可以了
'''
def inference(input_tensor,train,regularizer):
    with tf.variable_scope("layer1-conv1"):
        conv1_weights = tf.get_variable("weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides = [1,1,1,1],padding = "SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))



    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1,ksize = [1,2,2,1],strides=[1,2,2,1],padding = "SAME")


    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1,conv2_weights,[1,1,1,1],padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))


    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    print('pool2.get_shape()',pool2.get_shape())#(100, 7, 7, 64)

    pool_shape = pool2.get_shape().as_list()#以list的形式返回tensor的shape
    print("pool_shape",len(pool_shape))#4
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    #
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])
    # print('reshaped.shape',reshaped.shape)#(100, 3136)

    with tf.variable_scope("layer5-fc1"):
        fc1_weights = tf.get_variable("weight",[nodes,FC_SIZE],#3136,512
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))



        if regularizer !=None:

            tf.add_to_collection("losses",regularizer(fc1_weights))

        fc1_biases = tf.get_variable("bias",[FC_SIZE],
                                     initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        print('fc1.shape',fc1.shape)#fc1.shape (100, 512)
        t=tf.matmul(reshaped,fc1_weights)+fc1_biases
        print('tf.matmul(reshaped,fc1_weights)+fc1_biases  shape',t.shape)
    #一般只在全连接层进行dropout操作，而不在卷积层或者池化层
        if train:
            fc1= tf.nn.dropout(fc1,0.9)


    with tf.variable_scope("layer6-fc2"):

        fc2_weights = tf.get_variable("weight",[FC_SIZE,NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1))#[FC_SIZE,NUM_LABELS]即【 512，10】

        if regularizer != None:
            tf.add_to_collection("losses",regularizer(fc2_weights))

        fc2_biases=tf.get_variable("bias",[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        print('fc1,fc2_weights,shape',fc1.shape,fc2_weights.shape)

        logit = tf.matmul(fc1,fc2_weights)+fc2_biases
        print('fc2_bias.shape', fc2_biases.shape)

    return logit

def train(mnist):
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS],
                       name="x-input")

    # y_ = tf.placeholder(tf.float32, [None, OUTPUT_NIDE], name="y-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)#此句话有问题不能将函数转为张量
    print('regularizer',regularizer)


    y = inference(x, True,regularizer)
    print('y.shape',y.shape)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        xs, ys = mnist.train.next_batch(BATCH_SIZE)  # xs.shape (100, 784)
        xs = np.reshape(xs,
                        [BATCH_SIZE,
                         IMAGE_SIZE,
                         IMAGE_SIZE,
                         NUM_CHANNELS]
                        )

        print('xs.shape',xs.shape)
        print('xs',xs)

        print('x',x.shape,sess.run(x,feed_dict={x: xs}))



def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    train(mnist)
if __name__=='__main__':
    tf.app.run()