import tensorflow as tf
import numpy as np
from tensorflow_work.lstm_eg import reader

DATA_PATH = "/Users/ozintel/Downloads/simple-examples/simple-examples/data"
HIDDEN_SIZE = 200#隐藏层节点数为200，即词向量纬度为200
NUM_LAYERS = 2
VOCAB_SIZE = 10000
LEARNING_RATE = 1.0
TRAIN_BATCH_SIZE = 20#batch大小为20
TRAIN_NUM_STEP = 35#截断长度为35

EVAL_BATCH_SIZE = 1  # 学习速率
EVAL_NUM_STEP = 1
NUM_EPOCH = 2  # 训练轮数
KEEP_PROB = 0.5  # 不dropout的概率
MAX_GRAD_NORM = 5  # 控制梯度膨胀的参数


class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps

        # 初始化输入数据的维度
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 多层LSTM，设置dropout
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)#HIDDEN_SIZE；return new_h, new_state
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=KEEP_PROB)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)

        # 初始化
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # 将id转换为词向量，从 batch_size * num_steps  到 batch_size * num_steps * HIDDEN_SIZE
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
        #If initializer is None (the default), the default initializer passed in the variable scope will be used.
        # If that one is None too, a glorot_uniform_initializer will be used.
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        print('inputs_shape',inputs.shape)#inputs_shape (20, 35, 200)

        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)

        outputs = []

        # 训练
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                print('time_step',time_step,inputs[:, time_step, :].shape)#inputs[:, time_step, :].shape，(20, 200)


                cell_output, state = cell(inputs[:, time_step, :], state)#和单层的类似return new_h, new_state
                print('cell_output',cell_output.shape)#cell_output (20, 200)
                outputs.append(cell_output)
                # print('outputs',len(outputs))#列表中添加了多个张量，每个张量的纬度都意义，例如把张量换成数量，向量等等进行理解
        print('tf.concat(outputs, 1)',tf.concat(outputs, 1).shape)#(20,200X35)
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        # print('output',output.shape)# (20*35, 200)output是所有词汇是一列，每个词汇是200纬度；相当于全连接层的一个batch有20*35=700条数据，每个数据是200纬度的向量

        # 一个全连接得到最后的结果
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias
        print('logits',logits.shape,'\n',logits)#'len(logits)',len(logits),
        print('bias.shape',bias.shape)
        print('self.targets.shape',self.targets.shape,'tf.reshape(self.targets, [-1])',tf.reshape(self.targets, [-1]))

        # 计算交叉熵作为loss
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        # loss=tf.nn.seq2seq.sequence_loss_by_example(
            [logits],#（700，10000）
            [tf.reshape(self.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)]
        )
        print('loss',loss.shape)#loss (700,)

        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        if not is_training:
            return

        # 通过clip_by_global_norm控制梯度大小 避免膨胀
        # tf.trainable_variables()返回所有训练的变量
        # tf.gradients（）计算梯度 然后处理梯度
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(
                self.cost, trainable_variables
            ),
            MAX_GRAD_NORM
        )

        # 定义优化方法
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        # 定义训练步骤
        self.train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables)
        )


# 训练过程函数
def run_epoch(session, model, data, train_op, output_log):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    # step = 0
    # [x,y] = reader.ptb_producer( data, model.batch_size, model.num_steps )
    # coord = tf.train.Coordinator()
    # tf.train.start_queue_runners(session, coord=coord)
    for step, (x, y) in enumerate(
            reader.ptb_iterator(data, model.batch_size, model.num_steps)
    ):
        # print('step, \n',step,'\n',x,'\n',y)
        '''X,Yshape都是(20，35)
        X:[[9970 9971 9972 9974 9975 9976 9980 9981 9982 9983 9984 9986 9987 9988
  9989 9991 9992 9993 9994 9995 9996 9997 9998 9999    2 9256    1    3
    72  393   33 2133    0  146   19]
        Y:[[9971 9972 9974 9975 9976 9980 9981 9982 9983 9984 9986 9987 9988 9989
  9991 9992 9993 9994 9995 9996 9997 9998 9999    2 9256    1    3   72
   393   33 2133    0  146   19    6]
        
        '''
        # break
        # [a,b] = session.run([x,y])
        # if a.size != model.batch_size * model.num_steps :
        #     break

        cost, state, _ = session.run(
            [model.cost, model.final_state, train_op],
            {
                model.input_data: x,
                model.targets: y,
                model.initial_state: state
            }
        )
        total_costs += cost
        iters += model.num_steps
        step += 1
        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (step, np.exp(total_costs / iters)))

    return np.exp(total_costs / iters)


def main(_):
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # variable_scope为变量空间，当reuse=true时共享变量。
    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        # 训练，每次训练后用valid数据测试
        for i in range(NUM_EPOCH):
            print("In iteration : %d " % (i + 1))
            run_epoch(session, train_model, train_data, train_model.train_op, True)

            valid_perplexity = run_epoch(session, eval_model, valid_data, tf.no_op(), False)
            print("Epoch: %d Validation Perplexity : %.3f" % (i + 1, valid_perplexity))
        # 在最终测试集上进行测试
        test_perplexity = run_epoch(session, eval_model, test_data, tf.no_op(), False)
        print("test Perplexity : %.3f" % (test_perplexity))


if __name__ == "__main__":
    tf.app.run()
    # session = tf.Session()
    # coord = tf.train.Coordinator()
    # tf.train.start_queue_runners(session, coord=coord)
    # [a,b] = session.run([x,y])
    # print(a)
    # [a,b] = session.run([x,y])
    # print(a)
    # # print("444")