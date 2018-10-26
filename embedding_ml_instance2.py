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


path='/Users/ozintel/Downloads/Tsl_python_progect/local_ml/kaggle_competition/kaggle_competition_datas/kdxf/kdxf_data'
train_file_name='round1_iflyad_train.txt'
train1=pd.read_csv(os.path.join(path,train_file_name),delimiter='\t',
                   usecols=col,
                   nrows=1000000,
                   )#dtype='category'#lightgbm能够直接处理'category'类型数据


path='/Users/ozintel/Downloads/Tsl_python_progect/local_ml/kaggle_competition/kaggle_competition_datas/kdxf/kdxf_data2'
test_file_name='round2_iflyad_test_feature.txt'
train_file_name='round2_iflyad_train.txt'
train=pd.read_csv(os.path.join(path,train_file_name),delimiter='\t',
                  nrows=2000000,
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




batch_size = 10000;
graph = tf.Graph()

IGNORE_FEATURES = [
     'instance_id','click'
]
CATEGORITAL_FEATURES = [
    'adid', 'advert_id', 'advert_industry_inner', 'advert_name',
    'app_cate_id', 'app_id', 'campaign_id', 'carrier', 'city'
]
NUMERIC_FEATURES = [
    'feat_num_1', 'feat_num_2'
]




def embedding_self(train_inputs0):
    col_names=CATEGORITAL_FEATURES
    embed_list_all=[]
    embeding_size_all=0
    for index,each_col in enumerate(col_names):
        num_nunique=data[each_col].nunique()
        num=(num_nunique+1)//2
        if num<=50:
            embedding_size=num
        else:
            embedding_size=50
        embeding_size_all=embeding_size_all+embedding_size
        embeddings = tf.Variable(
            tf.random_uniform([num_nunique, embedding_size], -1.0, 1.0))#feat_size,embedding_size

        # col_j=index + 2
        embed = tf.nn.embedding_lookup(embeddings, train_inputs0[:,index])#一个batch的data[each_col]
        embed_list_all.append(embed)
    embed_all=tf.concat(values=embed_list_all, axis=1)  # concat two matrix
    return embed_all,embeding_size_all




with graph.as_default():
    # Input data.

    train_labels = tf.placeholder(tf.float32, shape=[batch_size, ])
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size,None])
    print('train_inputs.shape',train_inputs.shape)


    # embedding layer
    # # 'adid'
    # num_adid=data['adid'].nunique()//2
    # if num_adid<=50:
    #     embedding_size1 =num_adid
    # else:
    #     embedding_size1=50
    #
    # # city
    # num_city=    data['city'].nunique() // 2
    # if num_city<=50:
    #     embedding_size2 =num_city
    # else:
    #     embedding_size2 =50
    #
    #
    # # 此处totallength为11,相当于每个维度是11，两个是22维降维16维
    # # 因为用的查表，并不是one-hot左乘，因此多一行参数也没关系（这一行参数可能和某一行是相同的）
    # embeddings1 = tf.Variable(
    #     tf.random_uniform([data['adid'].nunique(), embedding_size1], -1.0, 1.0))  # 每个embedding_size的大小也可不同，更每个特征取值类别确定，例如一个8一个5等等
    #
    # embeddings2 = tf.Variable(
    #     tf.random_uniform([data['city'].nunique(), embedding_size2], -1.0,
    #                       1.0))  # 每个embedding_size的大小也可不同，更每个特征取值类别确定，例如一个8一个5等等
    #
    # embed1 = tf.nn.embedding_lookup(embeddings1, train_inputs1);  #
    # embed2 = tf.nn.embedding_lookup(embeddings2, train_inputs2)
    #
    # embed = tf.concat(values=[embed1, embed2], axis=1)  # concat two matrix
    #
    # print('shape of embed1 : \t', str(embed1.get_shape()))
    #
    # print('shape of embed2: \t', str(embed2.get_shape()))
    #
    # print('shape of embed : \t', str(embed.get_shape()))
    #
    # '''
    # shape of embed1 : 	 (10, 8)
    # shape of embed2: 	 (10, 8)
    # shape of embed : 	 (10, 16)
    # w1 shape:  (16, 100)
    # b1 shape:  (100,)
    # yo shape:  (10, 11)
    # train_labels shape:  (10, 11)
    # '''
    embed, embeding_size_all=embedding_self(train_inputs)
    print('shape of embed : \t', str(embed.get_shape()))
    # layer 1
    nh1 = 100;
    w1 = tf.Variable(tf.random_uniform([embeding_size_all, nh1], -1.0, 1.0));
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
    yo_logits = tf.matmul(z2, wo) + bo; #yo即logits值

    y_prob=tf.sigmoid(yo_logits)
    print('yo shape: ', yo_logits.get_shape())

    y_prob=tf.clip_by_value(y_prob, 1e-10, 1.0)##有问题的
    y_prob=tf.reshape(y_prob,shape=[batch_size,])

    print('train_labels shape: ', train_labels.get_shape()) #train_labels shape:  (5000,)

    print('pred_y.shape',y_prob.shape) #pred_y.shape (5000, 1)
    # loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=yo_logits, )#需要one-hot形式标签
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels, logits=yo_logits)#有错
    # loss = -tf.reduce_mean(train_labels * tf.log(tf.clip_by_value(yo, 1e-10, 1.0)))

    loss=tf.losses.log_loss(labels=train_labels, predictions=y_prob,)
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
                    x, yd = data[encoder][step*batch_size:(step+1)*batch_size].values,data['click'][step*batch_size:(step+1)*batch_size].values
                    # print('xx shape',x.shape)

                    feed_dict = {train_inputs: x, train_labels: yd};
                    # train_inputs_ss=session.run(train_inputs,feed_dict=feed_dict)
                    # print('train_inputs_ss ',train_inputs_ss)


                    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                    total_loss += np.mean(loss_val)

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

