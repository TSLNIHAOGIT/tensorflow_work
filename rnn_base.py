import numpy as np
X=[1,2]
state=[0.0,0.0]

#分开定义不同输入部分的权重以方便操作
w_cell_state=np.asanyarray([[0.1,0.2],[0.3,0.4]])
w_cell_input=np.asanyarray([0.5,0.6])
b_cell=np.asanyarray([0.1,-0.1])

#定义用于输出全联接层参数
w_output=np.asarray([[1.0],[2.0]])
b_output=0.1

#按照时间顺序执行循环神经网络前向传播过程
length=len(X)
for i in range(length):
    #计算循环体中全联接层神经网络
    before_activation=np.dot(state,w_cell_state)+X[i]*w_cell_input+b_cell
    state=np.tanh(before_activation)

    #根据当前时刻状态计算最终输出
    final_output=np.dot(state,w_output)+b_output

    #输出每个时刻的信息
    print('before_activation',before_activation)
    print('state',state)
    print('final_output',final_output)