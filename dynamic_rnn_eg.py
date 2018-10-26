#coding=utf-8
import tensorflow as tf
import numpy as np

tf.set_random_seed(1)   # set random seed
# 创建输入数据
np.random.seed(0)
X = np.random.randn(2, 4, 8)#,batch_size,num_step,dim
print('X',X)



## 第二个example长度为2
# X[1,2:] = 0

print('X2',X)
X_lengths = [4,4]#很重要，第一个example长度为4，第二个为2;2之后的不在进行计算了，state也是保留此时的

cell = tf.contrib.rnn.BasicLSTMCell(num_units=6, state_is_tuple=True)

# outputs, last_states = tf.nn.dynamic_rnn(
#     cell=cell,
#     dtype=tf.float64,
#     sequence_length=X_lengths,#表示的是batch中每行sequence的长度。
#     inputs=X)
#
# result = tf.contrib.learn.run_n(
#     {"outputs": outputs, "last_states": last_states},
#     n=1,
#     feed_dict=None)
#
# print ('''**************''',result[0])
#
#
# X_lengths = [4]#第一个example长度为4，第二个为2
#
# outputs, last_states = tf.nn.dynamic_rnn(
#     cell=cell,
#     dtype=tf.float64,
#     sequence_length=X_lengths,
#     inputs=X)
#
# result = tf.contrib.learn.run_n(
#     {"outputs": outputs, "last_states": last_states},
#     n=1,
#     feed_dict=None)
#
# print ('''**************''',result[0])

outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=None,
    inputs=X)

result = tf.contrib.learn.run_n(
    {"outputs": outputs, "last_states": last_states},
    n=1,
    feed_dict=None)

print ('''**************''',result[0])

# assert result[0]["outputs"].shape == (2, 10, 64)

# 第二个example中的outputs超过6步(7-10步)的值应该为0
# assert (result[0]["outputs"][1,7,:] == np.zeros(cell.output_size)).all()

'''
length=2
{'outputs': array([[[ 0.08725499,  0.10630993, -0.0380687 , -0.06147519,
         -0.04316959, -0.03054062],
        [ 0.19470228,  0.1501232 , -0.03202667, -0.04039893,
          0.12036165,  0.00829583],
        [ 0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        ]]]), 'last_states': LSTMStateTuple(c=array([[ 0.51285631,  0.31464421, -0.05321285, -0.07625828,  0.22931176,
         0.01818816]]), h=array([[ 0.19470228,  0.1501232 , -0.03202667, -0.04039893,  0.12036165,
         0.00829583]]))}
'''