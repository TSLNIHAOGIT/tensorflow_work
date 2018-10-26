import tensorflow as tf
import numpy as np

# input0=tf.Variable(tf.ones([1,3,4,5]))
#
input1 = tf.constant(
    [

[
    [[1],[2],[ 3]],
    [[4],[5],[6 ]],
    [[ 7],[8],[9]],
]

            ],dtype=tf.float32)
# print('input.shape',input.shape)#(1, 3, 3, 1)#3X3,通道数为1


input2 = tf.constant(
    [

[
    [[1,2],[2,4],[ 3,5]],
    [[4,1],[5,3],[6 ,2]],
    [[ 7,2],[8,3],[9,0]],
]

            ],dtype=tf.float32)

filter1 = tf.constant(
[
  [ [[2]],[[1]] ],
 #
  [ [[ 3]],[[4]] ]

],dtype=tf.float32)


filter2 = tf.constant(
[
  [  [[2],[3]],[[1],[5]]  ],
 #
  [  [[3],[1]],[[4],[2]]  ]
],dtype=tf.float32)

filter3 = np.array(
[
  [  [[2,1],[3,5]],[[1,6],[5,3]]  ],
 #
  [  [[3,3],[1,4]],[[4,2],[2,5]]  ]
])

filter3=tf.constant(filter3,dtype=tf.float32)

input=input2
filter=filter3
print('filter.shape',filter.shape)#(2, 2, 2, 1)
op = tf.nn.conv2d(input,filter,strides = [1,1,1,1],padding ='VALID')
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # sess.run(init)
    result = sess.run(op)

    print('input',input.shape,'\n',sess.run(input))
    print('filter',filter.shape,'\n',sess.run(filter))
    print('output', result.shape, '\n', result)

# input = tf.Variable(tf.ones([1,3,3,2]))
# filter = tf.Variable(tf.ones([1,1,2,2]))
# #
# op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
#
#
# if  __name__=='__main__':
#     with tf.Session() as sess:
#         sess.run(tf.initialize_all_variables())
#         # sess.run(init)
#
#         print('input',input.shape,'\n',sess.run(input))
#         print('filter',filter.shape, '\n', sess.run(filter))
#
#         result = sess.run(op2)
#         print('output',result.shape, '\n', np.array(result))