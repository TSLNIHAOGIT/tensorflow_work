import tensorflow as tf

# A = tf.random_normal([5, 4], dtype=tf.float32,seed=0)
A=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0,],[1.0,2.0,3.0]])
B = tf.constant([0, 2, 2], dtype=tf.int32)#B中是A.shape=(a,b),B中元素是b的序号
w = tf.ones([3], dtype=tf.float32)


# D = tf.nn.seq2seq.sequence_loss_by_example([A], [B], [w])
D=tf.contrib.legacy_seq2seq.sequence_loss_by_example([A], [B], [w])
res=tf.reduce_sum(D)
'''
拿一个word举例来说：假设总共有500个词汇，则该word后一个单词可能为500个中的任意一个，算出属于500个中任意一个的概率：p1,p2,...,p500
假设该单词，是500个中的第一70个（从0开始），则此位置标为1,然后计算交叉熵；最后对所有词汇的交叉熵进行求和
'''

print('A.shape',A.shape)
with tf.Session() as sess:
    A,B,w,D,res=sess.run([A,B,w,D,res])
    print('A', A)
    print('B', B)
    print('w',w)

    print('D',D)
    print('res',res)


import tensorflow as tf

#our NN's output
logits=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])
#step1:do softmax
y=tf.nn.softmax(logits)
#true label
y_=tf.constant([[1.0,0.0,0.0],[0.0,0.0,1.0],[0.0,0.0,1.0]])
#step2:do cross_entropy
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#do cross_entropy just one step
cross_entropy2=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits( labels=y_,
    logits=logits ))#dont forget tf.reduce_sum()!!

with tf.Session() as sess:
    softmax=sess.run(y)
    c_e = sess.run(cross_entropy)
    c_e2 = sess.run(cross_entropy2)

    cross_each=sess.run(-y_*tf.log(y))
    print('cross_each',cross_each)

    print("step1:softmax result=")
    print(softmax)
    print("step2:cross_entropy result=")
    print(c_e)
    print("Function(softmax_cross_entropy_with_logits) result=")
    print(c_e2)
