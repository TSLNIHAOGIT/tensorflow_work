'''
deepfm中将所有特征降维后的维度都设为相同了:  将所有特征（类别变量）取值的种类加起来作为最终的feat_size输入，embedd_size固定

embeding层维度：shape=[config.feature_size, config.embedding_size]

'''


import pandas as pd

import numpy as np
import tensorflow as tf

import datetime
from sklearn.feature_selection import chi2,mutual_info_classif,f_classif,SelectPercentile
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import lightgbm as lgb
import warnings
import time
import pandas as pd
import numpy as np
import os
import re
import pandas as pd
import urllib.parse
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import gc
from sklearn.feature_selection import RFE


# X_train0 = train.drop(
#     labels=['instance_id', 'click',
# 'devtype',
# 'creative_has_deeplink',

# 'os', 'creative_is_jump',
# 'creative_is_js','creative_is_voicead', 'app_paid'], axis=1)  # 默认删除行，添加axis = 1删除列
col=['instance_id','click','adid', 'advert_id', 'advert_industry_inner', 'advert_name',
        'app_cate_id', 'app_id', 'campaign_id', 'carrier', 'city',]

print('开始y_k_v3_2')
path='/Users/ozintel/Downloads/Tsl_python_progect/local_ml/kaggle_competition/kaggle_competition_datas/kdxf/kdxf_data'
train_file_name='round1_iflyad_train.txt'
train1=pd.read_csv(os.path.join(path,train_file_name),delimiter='\t',
                   usecols=col,
                   nrows=100000,
                   )#dtype='category'#lightgbm能够直接处理'category'类型数据


path='/Users/ozintel/Downloads/Tsl_python_progect/local_ml/kaggle_competition/kaggle_competition_datas/kdxf/kdxf_data2'
test_file_name='round2_iflyad_test_feature.txt'
train_file_name='round2_iflyad_train.txt'
train=pd.read_csv(os.path.join(path,train_file_name),delimiter='\t',
                  nrows=300000,
                  usecols=col,
                  )#dtype='category'#lightgbm能够直接处理'category'类型数据
test=pd.read_csv(os.path.join(path,test_file_name),delimiter='\t',
                 nrows=1000,
                 )
train=train.append(train1)
data=train

print(test.shape)

del train1
gc.collect()

print('click',data['click'].unique())

data=data.fillna(-1)

encoder=['adid', 'advert_id', 'advert_industry_inner', 'advert_name',
        'app_cate_id', 'app_id', 'campaign_id', 'carrier', 'city',]
col_encoder = LabelEncoder()
for feat in encoder:
    col_encoder.fit(data[feat])
    data[feat] = col_encoder.transform(data[feat])





minv = np.int32(0);
maxv = np.int32(5);
totalLength = 1#两个类别：根据损失函数类型决定是否，要把做成one-hot形式




batch_size = 5000;
graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs1 = tf.placeholder(tf.int32, shape=[batch_size])
    train_inputs2 = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.float32, shape=[batch_size, ])

    # embedding layer
    # 'adid'
    num_adid=(data['adid'].nunique()+1)//2
    if num_adid<=50:
        embedding_size1 =num_adid
    else:
        embedding_size1=50

    # city
    num_city=    (data['city'].nunique()+1) // 2
    if num_city<=50:
        embedding_size2 =num_city
    else:
        embedding_size2 =50


    # 此处totallength为11,相当于每个维度是11，两个是22维降维16维
    # 因为用的查表，并不是one-hot左乘，因此多一行参数也没关系（这一行参数可能和某一行是相同的）

    #其实也有问题，也不知道是不是问题：一个batchsize里面没有这么多类别。
    embeddings1 = tf.Variable(
        tf.random_uniform([data['adid'].nunique(), embedding_size1], -1.0, 1.0))  # 每个embedding_size的大小也可不同，更每个特征取值类别确定，例如一个8一个5等等

    embeddings2 = tf.Variable(
        tf.random_uniform([data['city'].nunique(), embedding_size2], -1.0,
                          1.0))  # 每个embedding_size的大小也可不同，更每个特征取值类别确定，例如一个8一个5等等

    embed1 = tf.nn.embedding_lookup(embeddings1, train_inputs1);  #
    embed2 = tf.nn.embedding_lookup(embeddings2, train_inputs2)

    embed = tf.concat(values=[embed1, embed2], axis=1)  # concat two matrix

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
    w1 = tf.Variable(tf.random_uniform([embedding_size1+embedding_size2, nh1], -1.0, 1.0));
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
    # loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=yo, )#需要one-hot形式标签
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels, logits=yo)#有错
    loss = -tf.reduce_mean(train_labels * tf.log(tf.clip_by_value(yo, 1e-10, 1.0)))
    optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(loss);
    #
    init = tf.initialize_all_variables()
    num_steps = 10000;
    with tf.Session(graph=graph) as session:
        init.run();
        print('inited')
        average_loss = 0



        for epoch in range(100):
            total_loss = 0.0;
            avg_loss = 0.0
            nstep = 80;

            for step in range(nstep):
                # x1, x2, yd = generate_batch(batch_size=batch_size)
                # if count%batch_size==0:
                    x1, x2, yd = data['adid'][step*batch_size:(step+1)*batch_size],data['city'][step*batch_size:(step+1)*batch_size],data['click'][step*batch_size:(step+1)*batch_size]
                    feed_dict = {train_inputs1: x1, train_inputs2: x2, train_labels: yd};
                    _, loss_val, embed1_ss = session.run([optimizer, loss, embed1], feed_dict=feed_dict)
                    total_loss += np.mean(loss_val)
                    # print('embed1\n', embed1_ss)
                    # print(step,total_loss)


            avg_loss = total_loss / float(nstep);
            #                 print(avg_loss)


            print('epoch=%d,       avg_loss: %f' % (epoch, avg_loss))

        # # # use add to add two number
        # for step in range(5):
        #     feed_dict = {train_inputs1: x1, train_inputs2: x2, train_labels: yd};
        #     yo.eval(feed_dict);
        #     outputs = session.run(yo, feed_dict=feed_dict)
        #     sums = np.argmax(outputs, axis=1)
        #     for i in range(outputs.shape[0]):
        #         print(str(x1[i]), '+', str(x2[i]), '=', str(sums[i]), ';\tis Correct? ', str(x1[i] + x2[i] == sums[i]))

