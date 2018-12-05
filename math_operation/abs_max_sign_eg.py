import tensorflow as tf
x = tf.constant(
    [
     [
        [1,1,3],
        [3,2,0],
        [5,2,0],
        [0,0,0],

     ],
    [
        [3,1,6],
        [8,0,0],
        [0,0,0],
        [0,0,0],
     ],


    ]
)

sess = tf.Session()
print('tf.abs(x).shape',tf.abs(x).shape)
print('abs(x)',sess.run(tf.abs(x)))#每个元素都取绝对值

print('reduce_max(x)',sess.run(tf.reduce_max(tf.abs(x), 2)))#每个元素都取绝对值
print('x.shape',x.shape)


def length(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length

print('x.shape',x.shape)
print('used',sess.run(tf.sign(tf.reduce_max(tf.abs(x), 2))))

print(sess.run(length(x)))
#[3 2]
'''
tf.abs(x).shape (2, 4, 3)
abs(x) [[[1 1 3]
  [3 2 0]
  [5 2 0]
  [0 0 0]]

 [[3 1 6]
  [8 0 0]
  [0 0 0]
  [0 0 0]]]
reduce_max(x) [[3 3 5 0]
 [6 8 0 0]]
x.shape (2, 4, 3)
x.shape (2, 4, 3)
used [[1 1 1 0]
 [1 1 0 0]]
[3 2]
'''