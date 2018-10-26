'''
将category变量输入到神经网络中时需要进行embedding表示，
最著名的莫过于word2vec，相应的tensorflow官方教程为：https://www.tensorflow.org/versions/r0.9/tutorials/word2vec/index.html#vector-representations-of-words

本文参考以上教程中的embedding代码，，重点练习两方面：
1、category变量的embedding表示
2、多变量embedding表示的串联


输入：两个整数，，  取值范围 [0,5]
输出：两个整数的和（one-hot 表示），即输出共有11个单元，对应于 0~11
样本来源：每次随机产生一组输入（10*2个  0~5之间的整数 及其 和的one-hot 表示）

'''
'''
Created on Jul 28, 2016

@author: colinliang
'''
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy as np
import tensorflow as tf

minv = np.int32(0);
maxv = np.int32(5);
totalLength = (maxv - minv + 1) * 2 - 1;


def generate_batch(batch_size):
    x1 = np.random.randint(minv, maxv + 1, batch_size).astype(np.int32)
    x2 = np.random.randint(minv, maxv + 1, batch_size).astype(np.int32)
    y = x1 + x2;

    yDistributed = np.zeros(shape=(batch_size, totalLength), dtype=np.float32)
    for i in range(batch_size):
        yDistributed[i, y[i]] = 1.0
    yDistributed = yDistributed + 1e-20;
    yDistributed /= np.sum(yDistributed, axis=1, keepdims=True)


    # print('x1,x2,yd\n', x1, '\n', x2, '\n', yDistributed)

    # print('totalLength',totalLength)
    '''
    （1）每个特征分别做embedding,x1,x2中有相同数值是可以的
    （2）所有特征的所有取值类别放在一起做embedding时就要将所以的进行排序
    
    
    x1:[3 4 5 0 0 3 3 4 1 0] 
    x2:[0 2 1 2 1 5 3 0 3 3] 
    totalLength:11
    
    x1取值最多0-5总共6种，x2取值也最多0-5总共6种；所以单纯的对x1embeding是时，是6X
    
    '''
    return x1, x2, yDistributed


x1, x2, yD = generate_batch(10)
# print(x1)
# print(x2)
# print(yD)


batch_size = 10;
graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs1 = tf.placeholder(tf.int32, shape=[batch_size])
    train_inputs2 = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.float32, shape=[batch_size, totalLength])

    # embedding layer
    embedding_size = 8;
    # 此处totallength为11,相当于每个维度是11，两个是22维降维16维
    #因为用的查表，并不是one-hot左乘，因此多一行参数也没关系（这一行参数可能和某一行是相同的）
    embeddings = tf.Variable(tf.random_uniform([totalLength, embedding_size], -1.0, 1.0))#每个embedding_size的大小也可不同，更每个特征取值类别确定，例如一个8一个5等等
    embed1 = tf.nn.embedding_lookup(embeddings, train_inputs1);#
    embed2 = tf.nn.embedding_lookup(embeddings, train_inputs2)


    embed = tf.concat( values=[embed1, embed2],axis=1)  # concat two matrix

    print('shape of embed1 : \t', str(embed1.get_shape()))

    print('shape of embed2: \t', str(embed2.get_shape()))

    print('shape of embed : \t', str(embed.get_shape()))

    '''
    shape of embed1 : 	 (10, 8)
    shape of embed2: 	 (10, 8)
    shape of embed : 	 (10, 16)
    w1 shape:  (16, 100)
    b1 shape:  (100,)
    yo shape:  (10, 11)
    train_labels shape:  (10, 11)
    '''

    # layer 1
    nh1 = 100;
    w1 = tf.Variable(tf.random_uniform([embedding_size * 2, nh1], -1.0, 1.0));
    print('w1 shape: ', w1.get_shape())
    b1 = tf.Variable(tf.zeros([nh1]))
    print('b1 shape: ', b1.get_shape())

    y1 = tf.matmul(embed, w1) + b1;

    z1 = tf.nn.relu(y1);

    # layer 2
    nh2 = 100;
    w2 = tf.Variable(tf.random_uniform([nh1, nh2], -1, 1))
    b2 = tf.Variable(tf.zeros([nh2]))

    y2 = tf.matmul(z1, w2) + b2;
    z2 = tf.nn.relu(y2);

    # layer 3-- output layer
    wo = tf.Variable(tf.random_uniform([nh2, totalLength], -1., 1.))
    bo = tf.Variable(tf.zeros([totalLength]))
    yo = tf.matmul(z2, wo) + bo;
    print('yo shape: ', yo.get_shape())

    print('train_labels shape: ', train_labels.get_shape())
    loss = tf.nn.softmax_cross_entropy_with_logits( labels=train_labels,logits=yo, );
    optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(loss);
    #
    init = tf.initialize_all_variables()
    num_steps = 10000;
    with tf.Session(graph=graph) as session:
        init.run();
        print('inited')
        average_loss = 0

        for epoch in range(1):
            total_loss = 0.0;
            avg_loss = 0.0
            nstep = 1;
            for step in range(nstep):
                x1, x2, yd = generate_batch(batch_size=batch_size)
                feed_dict = {train_inputs1: x1, train_inputs2: x2, train_labels: yd};
                _, loss_val,embed1_ss = session.run([optimizer, loss,embed1], feed_dict=feed_dict)
                total_loss += np.mean(loss_val)
                print('embed1\n', embed1_ss)
            # print(total_loss)
            avg_loss = total_loss / float(nstep);
            #                 print(avg_loss)


            print('epoch=%d,       avg_loss: %f' % (epoch, avg_loss))

        # use add to add two number
        for step in range(5):
            feed_dict = {train_inputs1: x1, train_inputs2: x2, train_labels: yd};
            yo.eval(feed_dict);
            outputs = session.run(yo, feed_dict=feed_dict)
            sums = np.argmax(outputs, axis=1)
            for i in range(outputs.shape[0]):
                print(str(x1[i]), '+', str(x2[i]), '=', str(sums[i]), ';\tis Correct? ', str(x1[i] + x2[i] == sums[i]))

