import tensorflow as tf
import ssl

from tensorflow.examples.tutorials.mnist import input_data
#全局取消证书验证
ssl._create_default_https_context = ssl._create_unverified_context

#定义神经网络结构相关参数
INPUT_NODE = 784  # 输入层的节点数。对于 MNIST 数据集，就等于图片的像素
OUTPUT_NODE = 10  # 输出层的节点数。这个等于类别的数目。因为在 MNIST 的数据集中需要区分的是 0~9 这 10 个数字，所以这里输出层的节点数为 10
LAYER1_NODE = 500  # 隐藏层节点数。这里使用只有一个隐藏层的网络结构作为样例。这个隐藏层有500个节点

#通过tf.get_variable()函数来或取变量，在训练神经网络时会创建这些变量；在测试时会通过保存的模型加载这些变量。
# 而且更加方便的是，因为可以在变量加载是将滑动平均变量重命名，所以可以直接通过同样的名字在训练是使用变量自身，而在测试是使用变量的滑动平均值。
# 在这个函数中也会将变量的正则化损失加入损失集合
# def get_weight_variable(shape,regularizer):
#     weights=tf.get_variable('weights',shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
#
#     #当给出了正则化生成函数是，将当前变量的正则化损失加入名字为loss的集合。在这里使用了add_to_collection函数将一个张量加入集合
#     #而这个集合的名称为losses。
#     #这是自定义集合，不在Tensorflow自动管理的集合列表中
#     if regularizer!=None:
#         tf.add_to_collection('losses',regularizer(weights))
#     return weights
#
# # 一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果。在这里定义了一个使用 ReLU 激活函数的三层全连接神经网络。
# # 通过加入隐藏层实现了多层网络结构， 通过 ReLU 激活函数实现了去线性化。在这个函数中也支持传入用于计算参数平均值得类，这样方便在
# # 测试时使用滑动平均模型。
# def inference(input_tensor,regularizer):
#     #声明第一层神经网络的变量并完成前向传播过程
#     with tf.variable_scope('layer1'):
#         #这里通过tf.getvariable()或者tf.Variable()没有本质区别，因为在训练或者测试时，没有在同一程序中多次调用这个函数，若多次调用，则应在
#         #第一次调用后将reuse设为true
#
#         weights=get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
#         biases=tf.get_variable('biases',[LAYER1_NODE], initializer=tf.constant_initializer(0.0))
#         layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
#
#        #类似声明第二层神经网络的变量并完成前向传播过程
#     with tf.variable_scope('layer2'):
#         # 这里通过tf.getvariable()或者tf.Variable()没有本质区别，因为在训练或者测试时，没有在同一程序中多次调用这个函数，若多次调用，则应在
#         # 第一次调用后将reuse设为true
#
#         weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
#         biases = tf.get_variable('biases', [OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
#         layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
#     #返回前向传播结果
#     return layer2
#


#2. 通过tf.get_variable函数来获取变量。
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights

#3. 定义神经网络的前向传播过程。使用命名空间方式，不需要把所有的变量都作为变量传递到不同的函数中提高程序的可读性

def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):

        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)


    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2