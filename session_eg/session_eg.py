import tensorflow as tf

a = tf.constant(4)
b = tf.constant(7)
c = a + b
# sess = tf.Session()#单独使用这个会报错
#以下三个session是等价的，可以使用任何一个
# sess = tf.InteractiveSession()
# with tf.Session() as sess:
sess = tf.Session()
with sess.as_default():
    print(c.eval())