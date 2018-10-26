import tensorflow as tf
import  numpy as np
tt=[]
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tt.append(t1)
tt.append(t2)
a=tf.concat([t1, t2],0)# == > [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
b=tf.concat([t1, t2],1)## [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
print('tt\n',tt)

t=np.array([t1])
print('t',t)

with tf.Session() as sess:
    tf.global_variables_initializer()
    b=sess.run(b)
    print(b)