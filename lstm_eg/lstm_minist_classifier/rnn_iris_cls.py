import numpy as np
from random import shuffle
import tensorflow as tf
from sklearn.datasets import load_iris
iris=load_iris()

# train_path = 'C:/Users/user/Documents/irris_train.txt'
# test_path = 'C:/Users/user/Documents/irris_test.txt'
# logs_path = 'C:/Users/user/Desktop/log'


#第一个样例
print(iris.data[0])#[ 5.1  3.5  1.4  0.2]
print(iris.data.shape)#(150, 4)
print(iris.target.shape)#(150,)
print(iris.target)
"""
　　[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
　　0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
　　1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
　　2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
　　2 2]


"""


train=iris.data
label=iris.target

def get_files(file_dir):
    irris_zero = []
    label_irris_zero = []
    irris_one = []
    label_irris_one = []

    file = open(file_dir)
    for line in file:
        line = line.replace('\n', '')
        wordList = line.split('\t')
        temp = []
        for word in wordList[:-1]:
            temp.append(float(word))
        if int(wordList[-1]) == 0:
            irris_zero.append(temp)
            label_irris_zero.append(int(0))
        else:
            irris_one.append(temp)
            label_irris_one.append(int(1))

    image_list = np.array(irris_zero + irris_one)
    label_list = np.array(label_irris_zero + label_irris_one)

    return image_list, label_list


def net_input(train_input):
    ti = []
    for i in train_input:
        temp_list = []
        for j in i:
            temp_list.append([j])
        ti.append(np.array(temp_list))

    train_input = ti
    return train_input


def net_input_trans_output(train_input, train_label):
    train_output = []
    for i in range(len(train_input)):
        count = 0
        if int(train_label[i]) == 1:
            temp_list = ([0] * 2)
            temp_list[-1] = 1
            train_output.append(temp_list)
        else:
            temp_list = ([0] * 2)
            temp_list[0] = 1
            train_output.append(temp_list)
    return train_output


def trainning(loss, learning_rate):
    with tf.name_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    return train_op


def cross_entropy_fun(logits, labels):
    with tf.variable_scope("loss") as scope:
        # cross_entropy = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(logits,1e-10,1.0)))
        cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cross_entropy


def initialize_weight_bias(in_size, out_size):
    weight = tf.truncated_normal(shape=(in_size, out_size), stddev=0.01, mean=0.0)
    bias = tf.constant(0.1, shape=[out_size])
    return tf.Variable(weight), tf.Variable(bias)


def model(data, target, dropout, num_hidden, num_layers):
    cells = list()
    for _ in range(num_layers):
        cell = tf.nn.rnn_cell.GRUCell(num_units=num_hidden)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=1 - dropout)  # 0.7=1-dropout
        cells.append(cell)
    network = tf.nn.rnn_cell.MultiRNNCell(cells=cells)
    val, _ = tf.nn.dynamic_rnn(cell=network, inputs=data, dtype=tf.float32)

    # get last output
    val = tf.transpose(val, [1, 0, 2])
    last_output = tf.gather(val, int(val.get_shape()[0]) - 1)

    # add softmax layer
    in_size = num_hidden
    out_size = int(target.get_shape()[1])
    weight, bias = initialize_weight_bias(in_size, out_size)

    # prediction output
    prediction = tf.nn.softmax(tf.matmul(last_output, weight) + bias)
    return prediction


def main():
    learning_rate = 0.001
    num_hidden = 24
    num_layers = 5
    batch_size = 10
    epoch = 500
    default_dropout = 0.2

    # train_input, train_label = get_files(train_path)
    # test_input, test_label = get_files(test_path)
    train_input,train_label=train,label
    test_input, test_label=train,label

    train_input = [map(float, i) for i in train_input]
    test_input = [map(float, i) for i in test_input]

    train_input = net_input(train_input)
    test_input = net_input(test_input)

    test_output = net_input_trans_output(test_input, test_label)
    train_output = net_input_trans_output(train_input, train_label)

    data = tf.placeholder(tf.float32, [None, 4, 1])
    target = tf.placeholder(tf.float32, [None, 2])
    dropout = tf.placeholder(tf.float32)


    ################
    prediction = model(data, target, dropout, num_hidden, num_layers)

    loss = cross_entropy_fun(prediction, target)
    minimize = trainning(loss, learning_rate)

    mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    tf.summary.scalar('error_rate', error)
    tf.summary.scalar('loss', loss)
    merge_summary_op = tf.summary.merge_all()
    with tf.Session() as session:
        summary_writer = tf.summary.FileWriter(logdir='.', graph=tf.get_default_graph())
        init_op = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init_op)

        # train
        no_of_batches = int(len(train_input)) // batch_size

        for i in range(epoch):
            ptr = 0
            for j in range(no_of_batches):
                inp, out = train_input[ptr:ptr + batch_size], train_output[ptr:ptr + batch_size]
                ptr += batch_size
                _, va, summary = sess.run([minimize, error, merge_summary_op],
                                          {data: inp, target: out, dropout: default_dropout})
                summary_writer.add_summary(summary, global_step=i * no_of_batches + j)
            print("Epoch ", str(i), va)
        summary_writer.close()

        # test
        incorrect = sess.run(error, {data: test_input, target: test_output, dropout: default_dropout})
        print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))


if __name__ == '__main__':
    main()
