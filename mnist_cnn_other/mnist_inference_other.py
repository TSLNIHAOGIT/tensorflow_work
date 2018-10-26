import tensorflow as tf

#定义神经网络结构相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一层卷积层滤波器的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
#第二层卷积层滤波器的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
#全连接层的节点个数
FC_SIZE = 512

#定义卷积神经网络，train用于区分训练过程和测试过程
def inference(input_tensor, train, regularizer):
    #声明第一层卷积层，输入28*28*1，输出28*28*32
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    #声明第一层池化层，输入28*28*32，输出14*14*32
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #声明第三层卷积层，输入14*14*32，输出14*14*64
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable('weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    #声明第四层池化层，输入14*14*64，输出7*7*64
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #将第四层输出格式（7*7*64)转化为第五层的输入格式一个向量
    pool2_shape = pool2.get_shape().as_list()
    nodes = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]   #7*7*64,pool2_shape[0]为一个bantch中数据的个数
    reshaped = tf.reshape(pool2, [pool2_shape[0], nodes])

    #声明第五层全连接层，输入7*7*64=3136长度的向量，输出512
    #引入dropout概念，会在训练时随机将部分节点的输出改为0，避免过拟合问题，一般用在全连接层
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight', [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_biases = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))
        if regularizer != None:#只有全连接层的权重需要加入正则化
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:fc1 = tf.nn.dropout(fc1, 0.5)

    #声明第6层全连接层，输入512，输出10，通过softmax之后得到最后的分类结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weight', [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable('bias', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        if regularizer != None:#只有全连接层的权重需要加入正则化
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2 = tf.matmul(fc1, fc2_weights) + fc2_biases

    return fc2