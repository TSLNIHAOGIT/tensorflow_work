import tensorflow as tf
with tf.Session() as sess:
    f01 = tf.constant([1,1],dtype=tf.float32)
    f02 = tf.constant([1,2],dtype=tf.float32)
    f11 = tf.constant(sess.run(tf.reshape(f01,[2,1])))
    f12 = tf.constant(sess.run(tf.reshape(f02,[2,1])))
    f13 = tf.tensordot(f11,f12,2)
    f1,f2=sess.run([f11,f12])
    print(f1,'\n\n',f2)
    '''
    [[1.]
     [1.]] 
    
    [[1.]
     [2.]]
    '''
    print('******')
    print(sess.run(f13))