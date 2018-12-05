import tensorflow as tf
from numpy.random import RandomState

'''
Loop {
    for i=1 to m,{  
        θj:=θj+α（yi −hθ(xi)）xj (for every j).i是上标
        
} }
'''

def nn_classfier():

    #定义batch大小
    batch_size=8

    #定义神经网络参数
    # 声明变量W1，W2
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    #shape一个维度为None可以方便使用不大batch,训练时数据分成多个小的batch,
    #但测试时可以一次使用全部数据，数据较大时这样测试会造成内存溢出
    x=tf.placeholder(tf.float32,shape=(None,2),name='x_input')#矩阵结构[[0.7,0.9],[3,5]],3行代表3个样例数据
    y_=tf.placeholder(tf.float32,shape=(None,1),name='y_input')

    #前向传播算法计算神经网络的输出
    a=tf.matmul(x,w1)
    y=tf.matmul(a,w2)

    #定义损失函数和反向传播算法
    cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))#计算预测值y和真实值y_之间的交叉熵
    train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)#反向传播算法优化神经网络参数

    #用随机数生成模拟数据集
    rdm=RandomState(1)
    dataset_size=257
    X=rdm.rand(dataset_size,2)
    Y=[[int(x1+x2<1)] for  (x1,x2) in X]

    #创建会话运行程序
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(w1))
        print(sess.run(w2))


        #设定训练的轮数
        STEPS=8000
        for i in range(STEPS):
            #每次选取batch_size个样不进行训练
            start=(i*batch_size)%dataset_size
            end=min(start+batch_size,dataset_size)

            print('i',i,'start',start,'end',end)

            #通过样本训练网络并更新参数
            sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})#从start行标到end行标
            if i%100==0:
                #每个一段时间计算在所有数据上的交叉熵并输出
                total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
                print('after {} tranin steps ,cross entropy on all data is {}'.format(i,total_cross_entropy))

        print(sess.run(w1))
        print(sess.run(w2))

def nn_self_loss_classfier():

    #定义batch大小
    batch_size=8

    #定义神经网络参数
    w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))


    #shape一个维度为None可以方便使用不大batch,训练时数据分成多个小的batch,
    #但测试时可以一次使用全部数据，数据较大时这样测试会造成内存溢出
    #两个输入节点
    x=tf.placeholder(tf.float32,shape=(None,2),name='x_input')#矩阵结构[[0.7,0.9],[3,5]],3行代表3个样例数据
    y_=tf.placeholder(tf.float32,shape=(None,1),name='y_input')

    #前向传播算法计算神经网络的输出
    y=tf.matmul(x,w1)

    # 定义损失函数和反向传播算法

    #定义预测多和少的成本
    loss_less=10
    loss_more=1
    loss=tf.reduce_sum(tf.where(tf.greater(y,y_),
                                (y-y_)*loss_more,
                                (y_-y)*loss_less
                                ))


    train_step=tf.train.AdamOptimizer(0.001).minimize(loss)#反向传播算法优化神经网络参数

    #用随机数生成模拟数据集
    rdm=RandomState(1)
    dataset_size=128
    X=rdm.rand(dataset_size,2)
    Y=[[x1+x2+rdm.rand()/10.0-0.05] for  (x1,x2) in X]#Y中加入了随机噪音

    #创建会话运行程序
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(w1))



        #设定训练的轮数
        STEPS=5000
        for i in range(STEPS):
            #每次选取batch_size个样不进行训练
            start=(i*batch_size)%dataset_size
            end=min(start+batch_size,dataset_size)

            # print('i',i,'start',start,'end',end)

            #通过样本训练网络并更新参数
            sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})#从start行标到end行标
            # if i%100==0:
            #     #每个一段时间计算在所有数据上的损失函数并输出
            #     total_cross_entropy=sess.run(loss,feed_dict={x:X,y_:Y})
            #     print('after {} tranin steps ,self-define loss on all data is {}'.format(i,total_cross_entropy))

        print(sess.run(w1))



def get_weight(shape,lambdas):
    '''
    没有正是运行
    :param shape:
    :param lambdas:
    :return:
    '''

    #或取一层神经网络的权重，并将这个权重发的L2正则化加入名称为loss的集合中
    var=tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    #add_to_collection函数将这个新生成的变量L2正则化损失加入集合
    tf.add_to_collection(
        'losses',
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(lambdas )(var))
    )
    #返回生成的变量
    return var



    #定义batch大小
    batch_size=8
    layer_dimension=[2,10,10,10,1]
    n_layers=len(layer_dimension)

    #这个变量维护前向传播时最深层的节点，开始时就是输入层
    cur_layer=x
    #当前层节点数
    in_dimension=layer_dimension[0]

    #定义神经网络参数
    w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))


    #shape一个维度为None可以方便使用不大batch,训练时数据分成多个小的batch,
    #但测试时可以一次使用全部数据，数据较大时这样测试会造成内存溢出
    #两个输入节点
    x=tf.placeholder(tf.float32,shape=(None,2),name='x_input')#矩阵结构[[0.7,0.9],[3,5]],3行代表3个样例数据
    y_=tf.placeholder(tf.float32,shape=(None,1),name='y_input')


    #通过一个循环生成5层全联接层的神经网络
    for i in range(i,n_layers):
        # layer_dimension[i]#为下一层节点数
        out_dimension=layer_dimension[i]
        #生成当前层权重变量，并将这个变量的L2正则项损失加入计算图中的集合中
        weight=get_weight([in_dimension,out_dimension],0.001)
        bias=tf.Variable(tf.constant(0.1,shape=[out_dimension]))
        #使用relu激活函数
        cur_layer=tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
        #进入下一层之前将下一层的节点个数更新为当前层节点个数
        in_dimension=layer_dimension[i]

    #在定义神经网络前向传播时已经将所有L2正则化损失加入图上的集合中，
    #这里只需要计算刻画模型在训练数据上表现的损失函数
    mse_loss=tf.reduce_mean(tf.square(y_-cur_layer))

    #将均方误差损失函数加入损失集合
    tf.add_to_collection('losses',mse_loss)

    #get_collection返回一个列表，这个列表是所有这个集合中的元素，此例中，
    #这些元素就是损失函数的不同部分，将它们加起来就可以得到最终的损失函数
    loss=tf.add_n(tf.get_collection('losses'))


    # #前向传播算法计算神经网络的输出
    # y=tf.matmul(x,w1)
    #
    # # 定义损失函数和反向传播算法
    #
    # #定义预测多和少的成本
    # loss_less=10
    # loss_more=1
    # loss=tf.reduce_sum(tf.where(tf.greater(y,y_),
    #                             (y-y_)*loss_more,
    #                             (y_-y)*loss_less
    #                             ))
    #
    #
    # train_step=tf.train.AdamOptimizer(0.001).minimize(loss)#反向传播算法优化神经网络参数
    #
    # #用随机数生成模拟数据集
    # rdm=RandomState(1)
    # dataset_size=128
    # X=rdm.rand(dataset_size,2)
    # Y=[[x1+x2+rdm.rand()/10.0-0.05] for  (x1,x2) in X]#Y中加入了随机噪音
    #
    # #创建会话运行程序
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(w1))
    #
    #
    #
    #     #设定训练的轮数
    #     STEPS=5000
    #     for i in range(STEPS):
    #         #每次选取batch_size个样不进行训练
    #         start=(i*batch_size)%dataset_size
    #         end=min(start+batch_size,dataset_size)
    #
    #         # print('i',i,'start',start,'end',end)
    #
    #         #通过样本训练网络并更新参数
    #         sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})#从start行标到end行标
    #         # if i%100==0:
    #         #     #每个一段时间计算在所有数据上的损失函数并输出
    #         #     total_cross_entropy=sess.run(loss,feed_dict={x:X,y_:Y})
    #         #     print('after {} tranin steps ,self-define loss on all data is {}'.format(i,total_cross_entropy))
    #
    #     print(sess.run(w1))
def moving_average():
    #定义变量计算滑动平均，初始值为0，类型必须是实数

    v1=tf.Variable(0,dtype=tf.float32)
    step = tf.Variable(0, trainable=False)

    #定义滑动平均类，衰减率为0.99,和控制衰减率的变量step
    ema=tf.train.ExponentialMovingAverage(0.99,step)

    #定义一个更新滑动平均的操作，给定一个列表，每次执行都要更新列表中的变量
    maintain_average_op=ema.apply([v1])
    with tf.Session() as sess:
        init_op=tf.initialize_all_variables()
        sess.run(init_op)
        
    #或取滑动平均之后变量发的取值，初始化之后变量v1以及v1的滑动平均值都是0
        print(sess.run([v1,ema.average(v1)]))

        #更新变量v1的值到5
        sess.run(tf.assign(v1,5))

        #更新v1的滑动平均值，衰减率为min(0.99,(1+step)/(10+step))=0.1
        #所以v1的滑动平均值为0.1*0+0.9*5=4.5
        sess.run(maintain_average_op)
        print([v1,ema.average(v1)])

        #更新step值为10000
        sess.run(tf.assign(step,10000))
        #更新v1值为10
        sess.run(tf.assign(v1,10))

        #跟新v1滑动平均值
        sess.run(maintain_average_op)
        print(sess.run([v1, ema.average(v1)]))
        #再次跟新v1滑动平均值
        sess.run(maintain_average_op)
        print(sess.run([v1, ema.average(v1)]))

if __name__=='__main__':
    nn_classfier()
    # nn_self_loss_classfier()
    moving_average()


'''
dataset_size=128
i 1 start 8 end 16
i 2 start 16 end 24
i 3 start 24 end 32

i 14 start 112 end 120
i 15 start 120 end 128
i 16 start 0 end 8
i 17 start 8 end 16
i 18 start 16 end 24




'''