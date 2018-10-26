import tensorflow as tf;
import numpy as np;
'''tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
tf.nn.embedding_lookup（tensor, id）:tensor就是输入张量，id就是张量对应的索引，其他的参数不介绍'''
c = np.random.random([10,5])
print (c,'\n**************')
b1 = tf.nn.embedding_lookup(c, [1, 3,2,4])#输出为张量的第一和第三个元素,返回类型和原来相同
b2 = tf.nn.embedding_lookup(c, [[1,3,4],[2,4,4]])#仍是取行返回的是张量

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    b1,b2=sess.run([b1,b2])
    print(b1,'\n**********','\n',b2)

import tensorflow as tf
import numpy as np

# 定义一个未知变量input_ids用于存储索引
input_ids = tf.placeholder(dtype=tf.int32, shape=[None])

# 定义一个已知变量embedding，是一个5*5的对角矩阵
# embedding = tf.Variable(np.identity(5, dtype=np.int32))

# 或者随机一个矩阵
embedding = a = np.asarray([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3], [4.1, 4.2, 4.3]])
print('embedding\n',embedding)

# 根据input_ids中的id，查找embedding中对应的元素
input_embedding = tf.nn.embedding_lookup(embedding, input_ids)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# print(embedding.eval())
print('input_embedding',sess.run(input_embedding, feed_dict={input_ids: [1, 2, 3, 0, 3, 2, 1]}))

#############

import tensorflow as tf

embedding = tf.get_variable("embedding", initializer=tf.ones(shape=[10, 5]))
look_uop = tf.nn.embedding_lookup(embedding, [1, 2, 3, 4])
# embedding_lookup就像是给 其它行的变量加上了stop_gradient
w1 = tf.get_variable("w", shape=[5, 1])

z = tf.matmul(look_uop, w1)

opt = tf.train.GradientDescentOptimizer(0.1)

#梯度的计算和更新依旧和之前一样，没有需要注意的
gradients = tf.gradients(z, xs=[embedding])
train = opt.apply_gradients([(gradients[0],embedding)])

#print(gradients[4])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(train))
    print('embedding',sess.run(embedding))
    print('w1 ',sess.run(w1 ))
    print('look_up',sess.run(look_uop))