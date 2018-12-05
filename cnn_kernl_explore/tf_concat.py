import tensorflow as tf
import  numpy as np
tt=[]
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tt.append(t1)
tt.append(t2)
print('type t1 one',type(t1))
t1=tf.convert_to_tensor(t1,dtype=tf.float32)
print('type(t1)',type(t1))

aa=tf.placeholder(tf.float32, shape=(None,3))
print('aa',aa)
a=tf.concat([t1, aa],0)# == > [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
b=tf.concat([t1, t2],1)## [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
print('type(a)',type(a))#type(a) <class 'tensorflow.python.framework.ops.Tensor'>
print('tt\n',tt)

t=np.array([t1])
print('t',t)

with tf.Session() as sess:
    tf.global_variables_initializer()

    feed_dict = {aa: t2};

    a=sess.run(a,feed_dict=feed_dict)
    print(a)