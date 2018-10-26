import tensorflow as tf
#定义一个lstm结构，在tensorflow中通过一句简单命令就可以实现一个完整的lstm结构
#lstm中使用的变量也会在该函数中自动被声明


#LSTM结构在Tnesorflow中可以很简单的实现，以下代码为了展示LSTM的前向传播过程

import tensorflow as tf

lstm_hidden_size=100
batch_size=10

num_steps=10

#定义一个LSTM结构，在Tensorflow中通过一句简单的命令就可以事先一个完整的LSTM结构
#LSTM中使用的变量也会在该函数中自动被声明
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

#将LSTM中的状态初始化为全0的数组。和其他神经网络相似，在优化循环神经网络时，每次也会使用一个batch的训练样本，
#下面中，batch_size给出了一个batch的大小
#BasicLSTMCell类提供了zero_state函数来生成全0的初始状态
init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

#定义损失函数
loss=0.0

def fully_connected(lstm_out):

    return


def cal_loss(final_out,expected_out):
    return



expected_out=""
#在8.1节中介绍，理论上循环神经网络可以处理任意长度的序列，但是在训练是为了避免梯度消散的问题，会规定一个最大的序列长度，
#一下代码中，用num_steps来表示这个长度

for i in range(num_steps):
    #在第一个时刻声明LSTM结构中使用的变量，在之后的时刻都需要复用之前定义好的变量
    if i>0:
        tf.get_variable_scope().reuse_variables()
        #每一步处理时间序列中的一个时刻。将当前输入current_input和前一时刻状态state传入定义的LSTM结构可以得到
        # 当前LSTM结构的输出lstm_output和更新后的状态state

        lstm_out,state=lstm(curren_input,state)

        #将当前时刻LSTM结构的输出传入一个全连接层得到最后的输出
        final_out=fully_connected(lstm_out)

        #计算当前时刻的损失
        loss+=cal_loss(final_out,expected_out)