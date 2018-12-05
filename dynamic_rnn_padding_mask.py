#coding=utf-8
import tensorflow as tf
import numpy as np


'''
（1）padding:dynamic处理变长序列时，取最大长度序列，不足的序列补0；
 (2) mask:设置sequence_length，这样输出时补零的time_step部分输出也为0
（3）此时计算cost不是取最后一个time_step的hidden_state而是取最后一个不为零的：cost这里其实是不用mask的，因为label和prdiciton都是0
（4）预测的时候，即训练结束后，放入softmax分类时，不是取最后一个time_step的hidden_state而是取最后一个不为零的

'''


#padding,mask
'''
但是dynamic有个参数：sequence_length，这个参数用来指定每个example的长度，比如上面的例子中，我们令 sequence_length为[20,13]，表示第一个example有效长度为20，第二个example有效长度为13，当我们传入这个参数的时候，对于第二个example，TensorFlow对于13以后的padding就不计算了，其last_states将重复第13步的last_states直至第20步，而outputs中超过13步的结果将会被置零
--------------------- 
作者：luchi007 
来源：CSDN 
原文：https://blog.csdn.net/u010223750/article/details/71079036 
版权声明：本文为博主原创文章，转载请附上博文链接！
'''

tf.set_random_seed(1)   # set random seed
# 创建输入数据
np.random.seed(0)
X = np.random.randn(2, 8, 8)#,batch_size,num_step,dim
print('X',X)



## 第二个example长度为2
X[1,2:] = 0

print('X2',X)
X_lengths = [4,4]#很重要，第一个example长度为4，第二个为2;2之后的不在进行计算了，state也是保留此时的

cell = tf.contrib.rnn.BasicLSTMCell(num_units=6, state_is_tuple=True)

outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,#表示的是batch中每行sequence的长度。
    inputs=X)

result = tf.contrib.learn.run_n(
    {"outputs": outputs, "last_states": last_states},
    n=1,
    feed_dict=None)

print ('''**************''',result[0])


####################################################
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

####################################################
# outputs, last_states = tf.nn.dynamic_rnn(
#     cell=cell,
#     dtype=tf.float64,
#     sequence_length=None,
#     inputs=X)
#
# result = tf.contrib.learn.run_n(
#     {"outputs": outputs, "last_states": last_states},
#     n=1,
#     feed_dict=None)
#
# print ('''**************''',result[0])

# assert result[0]["outputs"].shape == (2, 10, 64)

# 第二个example中的outputs超过6步(7-10步)的值应该为0
# assert (result[0]["outputs"][1,7,:] == np.zeros(cell.output_size)).all()

'''
length=2
{'outputs': array([
        [[ 0.08725499,  0.10630993, -0.0380687 , -0.06147519,
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