import tensorflow as tf
tf.reset_default_graph()#清空default graph以及nodes
with tf.variable_scope('Space_a'):
    a = tf.constant([1,2,3])
with tf.variable_scope('Space_b'):
    b = tf.constant([4,5,6])
with tf.variable_scope('Space_c'):
    c = a + b
d = a + b
with tf.Session()as sess:
    print(a)
    print(b)
    print(c)
    print(d)
    print(sess.run(c))
    print(sess.run(d))
