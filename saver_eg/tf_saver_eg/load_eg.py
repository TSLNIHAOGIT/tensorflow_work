import tensorflow as tf
# tf.reset_default_graph()

def load_build_var():

  # Create some variables.
  v1 = tf.get_variable("v1", shape=[3])
  v2 = tf.get_variable("v2", shape=[5])

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()

  # Later, launch the model, use the saver to restore variables from disk, and
  # do some work with the model.
  with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "model/model.ckpt")
    print("Model restored.")
    # Check the values of the variables
    print("v1 : %s" % v1.eval())
    print("v2 : %s" % v2.eval())


def load_meta_graph():
  saver=tf.train.import_meta_graph("model/model.ckpt.meta")
  with tf.Session() as sess:
    saver.restore(sess,'model/model.ckpt')
    print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0')))


if __name__=='__main__':
  load_build_var()
  print('************************')
  # load_meta_graph()