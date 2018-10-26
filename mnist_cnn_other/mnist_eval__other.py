import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow_work.mnist_cnn_other import mnist_inference_other
from tensorflow_work.mnist_cnn_other import mnist_train_other
import numpy as np
import ssl
#全局取消证书验证
ssl._create_default_https_context = ssl._create_unverified_context

#1. 每10秒加载一次最新的模型

# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            5000,#验证集数据个数
            mnist_inference_other.IMAGE_SIZE,
            mnist_inference_other.IMAGE_SIZE,
            mnist_inference_other.NUM_CHANNELS],
                           name="x-input")

        y_ = tf.placeholder(tf.float32, [None, mnist_inference_other.OUTPUT_NODE], name='y-input')
        validate_feed = {x: np.reshape(mnist.validation.images,[-1,28,28,1]), y_: mnist.validation.labels}

        y = mnist_inference_other.inference(x,False, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train_other.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train_other.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("../mnist-cnn/MNIST_data/", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    main()