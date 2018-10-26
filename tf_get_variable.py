import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;

a1 = tf.get_variable(name='a1', shape=[2, 3])
a2 = tf.get_variable(name='a2', shape=[1], initializer=tf.constant_initializer(1))
a3 = tf.get_variable(name='a3', shape=[2, 3], initializer=tf.ones_initializer())
embedding = tf.get_variable("embedding", [2,3])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print('a1',sess.run(a1))
    print(sess.run(a2))
    print(sess.run(a3)  )
    print('embedding',sess.run(embedding))