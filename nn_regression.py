import numpy as np

#产生100条数据
num_puntos = 100
conjunto_puntos = []
for i in range(num_puntos):
    x1= np.random.normal(0.0, 0.9)
    y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.05)
    conjunto_puntos.append([x1, y1])


x_data = [v[0] for v in conjunto_puntos]
y_data = [v[1] for v in conjunto_puntos]

import matplotlib.pyplot as plt
#
# #Graphic display
# plt.plot(x_data, y_data, 'ro')
# plt.legend()
# plt.show()

import tensorflow as tf

#定义W和b，这里的W是一个数，取值范围为-1到1，b为一个数
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
#模型
y = W * x_data + b
# y=tf.matmul(W,x_data)+b

#损失函数
loss = tf.reduce_mean(tf.square(y - y_data))
#优化，基于梯度下降法的优化，步长为0.5
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
#对损失函数优化
train = optimizer.minimize(loss)

#初始化
init = tf.global_variables_initializer()

#启动图计算
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train)
    if step % 10 == 0:
        W_,b_,loss_=sess.run([W,b,loss])
        print(step,W_,b_,loss_ )

plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.legend()
plt.show()