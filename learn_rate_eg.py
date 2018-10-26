import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;

all_examples_num=500
batch_size=10
decay_steps = 50#all_examples_num/bactch_size，50次迭代就会将所有数据训练一遍，此时epoch=1;

learning_rate = 0.1
decay_rate = 0.96

global_steps = 100#即若为100次迭代，两个epoch即epoch=2,epoch：迭代次数，1个epoch等于使用训练集中的全部样本训练一次；
epoch=global_steps/decay_steps
print('epoch',epoch)

global_ = tf.Variable(tf.constant(0))
c = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=True)
d = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=False)

T_C = []
F_D = []

with tf.Session() as sess:
    for i in range(global_steps):#当前迭代步数，1，2，3，4，5,......
        # print('i',i)
        T_c = sess.run(c, feed_dict={global_: i})
        T_C.append(T_c)
        F_d = sess.run(d, feed_dict={global_: i})
        F_D.append(F_d)

plt.figure(1)
plt.plot(range(global_steps), F_D, 'r-')
plt.plot(range(global_steps), T_C, 'b-')

plt.show()