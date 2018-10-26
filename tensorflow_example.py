import tensorflow as tf


def tensor_gener_compute():
    #定义常数矩阵a,b
    a=tf.constant([[1.0,2.0]],name='a')
    b=tf.constant([[2.0],[3.0]],name='b')

    # 使用默认计算图,结果为True
    print(a.graph is tf.get_default_graph())

    #矩阵相乘
    product=tf.matmul(a,b)

    #乘法结果加上常数
    result=tf.add(product,tf.constant(2.0),name='add')

    #载入会话并运行
    with tf.Session() as sess:
        result = sess.run(result)
        print(result)


def tensor_gener_compute2():
    #定义常数矩阵a,b
    a=tf.constant([[1.0,2.0]],name='a')
    b=tf.constant([[2.0],[3.0]],name='b')
    print('b1',b[0],b[1])

    # 使用默认计算图,结果为True
    print(a.graph is tf.get_default_graph())

    #矩阵相乘
    product=tf.matmul(a,b)

    #乘法结果加上常数
    for i in range(3):
        result1=tf.add(product,tf.constant(2.0),name='add')
        print('result1',result1)
        result2=result1+result1
        print('result2', result2)


    sess=tf.InteractiveSession()
    #tf.InteractiveSession可以放在开始，也可以放在后面，使用更少的代码就可以输出结果
    print('result.eval',result1.eval())
    print('sess.run(result',sess.run(result1))
    sess.close()

def self_def_graph():
    g1=tf.Graph()#生成新的计算图，不使用默认计算图
    # g1.device('/cpu:20')
    with g1.as_default():#设置g1为默认计算图
        #在计算图g1中定义变量'v',并设定初始值为0
        v1=tf.get_variable('v',shape=[1],initializer=tf.zeros_initializer())#shape=[1]

    g2 = tf.Graph()  # 生成新的计算图，不使用默认计算图
    with g2.as_default():  # 设置g2为默认计算图
        # 在计算图g2中定义变量'v',并设定初始值为1
        v2 = tf.get_variable('v',shape=[1], initializer=tf.ones_initializer())#shape=[1]

    with tf.Session(graph=g1) as sess:
        tf.initialize_all_variables().run()# Initialize the variables we defined above on g1 graph,所有变量必须初始化之后才可以使用
        '''先通过tf.variable_scope生成一个上下文管理器，并指明需求的变量在这个上下文管理器中，
        就可以直接通过tf.get_variable获取已经生成的变量'''
        with tf.variable_scope('',reuse=True):
            #reuse为True的时候表示用tf.get_variable 得到的变量可以在别的地方重复使用,将参数reuse设置为True是，tf.variable_scope将只能获取已经创建过的变量。
            #在tensorflow中，为了 节约变量存储空间 ，我们常常需要通过共享 变量作用域(variable_scope) 来实现 共享变量
            # 若tf.variable_scope函数使用参数reuse=None或者reuse=False创建上下文管理器，
            # #tf.get_variable操作将创建新的变量。#如果同名的变量已经存在，则tf.get_variable函数将报错
            vv=tf.get_variable('v')
            print(sess.run(vv))

    with tf.Session(graph=g2) as sess:
        tf.initialize_all_variables().run()
        with tf.variable_scope('',reuse=True):
            v_f=tf.get_variable('v')
            print(sess.run(v_f))

def create_variable_tensor():
    #由于tf.Variable() 每次都在创建新对象，所有reuse=True 和它并没有什么关系。
    # 对于get_variable()，来说，如果已经创建的变量对象，就把那个对象返回，如果没有创建变量对象的话，就创建一个新的,共享变量时使用。
    x=tf.constant([0.9,0.7])
    weights=tf.Variable(tf.random_normal([2,3],stddev=2))
    biases=tf.Variable(tf.zeros([3]))#生成初始值全为0，长度为3的变量
    w2 = tf.Variable(weights.initialized_value())#w2的初始权值与weights相同，Returns the Tensor used as the initial value for the variable.
    w3 = tf.Variable(weights.initialized_value()*2.0)
    with tf.Session() as sess:
        # tf.initialize_all_variables().run()#这种初始化页可以
        # tf.global_variables_initializer().run()
        # sess.run(tf.initialize_all_variables())#Tensorflow中，所有变量都必须初始化才能使用
        sess.run(tf.global_variables_initializer())
        print(sess.run(weights),'w2',sess.run(w2))

def forward_propagation():

    #声明变量W1，W2
    w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
    w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

    #暂时将输入的特征向量定义为常量
    x=tf.constant([[0.7,0.9]])#矩阵结构[[0.7,0.9],[3,5]]

    #前向传播算法计算神经网络的输出
    a=tf.matmul(x,w1)
    y=tf.matmul(a,w2)

    with tf.Session() as sess:
        #tensorflow所有变量必须初始化后才能使用
        sess.run(tf.global_variables_initializer())
        # sess.run(w1.initializer)
        # sess.run(w2.initializer)

        print(sess.run(y))
def forward_propagation_placeholder():

    #声明变量W1，W2
    w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
    w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

    #暂时将输入的特征向量定义为常量
    x=tf.placeholder(tf.float32,shape=(3,2),name='input')#矩阵结构[[0.7,0.9],[3,5]],3行代表3个样例数据
    x2=tf.placeholder(tf.float32,shape=(3,2),name='input2')
    x3=x+x2
    #前向传播算法计算神经网络的输出
    a=tf.matmul(x3,w1)
    y=tf.matmul(a,w2)

    with tf.Session() as sess:
        #tensorflow所有变量必须初始化后才能使用
        sess.run(tf.global_variables_initializer())
        # sess.run(w1.initializer)
        # sess.run(w2.initializer)

        print(sess.run(x,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]],x2:[[0.79,0.9],[0.71,0.4],[0.75,0.8]]}))
        print(sess.run(x2, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]],
                                     x2: [[0.79, 0.9], [0.71, 0.4], [0.75, 0.8]]}))
        print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]],
                                      x2: [[0.79, 0.9], [0.71, 0.4], [0.75, 0.8]]}))

if __name__=='__main__':
    tensor_gener_compute2()
    # self_def_graph()
    # create_variable_tensor()
    # forward_propagation()
    # forward_propagation_placeholder()

