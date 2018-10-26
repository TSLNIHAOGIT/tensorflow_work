import tensorflow as tf
import numpy as np

'''
注意单层的rnncell:输入的纬度和以和输出的不一样；
当时多层情形时：由于上层输出作为该层的输入因此，此时每层的输入输出纬度都应该一样

'''

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128) # state_size = 128
print(cell.state_size) # 128

inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
# inputs=tf.constant(value=1,shape=(32,100),dtype=np.float32)
print(0)
h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
print(1)

output, h1 = cell(inputs, h0) #调用call函数,最新版的1.8
# output, h1 = cell.call(inputs, h0) #调用call函数
print(2)
print(h1.shape) # (32, 128)






print('****************************************************')



cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128) # state_size = 128
print(cell.state_size) # 128

inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
# inputs=tf.constant(value=1,shape=(32,100),dtype=np.float32)
print(0)
h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
print(1)

output, h1 = cell(inputs, h0) #调用call函数,最新版的1.8
# output, h1 = cell.call(inputs, h0) #调用call函数
print(2)
print(output.shape,h1[1].shape) # (32, 128),(32, 128)

print('****************************************************')

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=100) # state_size = 128
print(lstm_cell.state_size) # 128

cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*2)

inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
# inputs=tf.constant(value=1,shape=(32,100),dtype=np.float32)
print(0)
h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
print(1)

output, h1 = cell(inputs, h0) #调用call函数,最新版的1.8
# output, h1 = cell.call(inputs, h0) #调用call函数
print(2)
print(output.shape,type(h1),h1[0][0],'\n',h1[0][1]) # 类似（（state1(c,h),state2(c,h））(32, 100),(32, 100)#单层lstm时h1是一个元组，两层lstm时，是两个元组，以此类推,且这些元组都放在一个大元组里