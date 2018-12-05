import tensorflow as tf
from tensorflow.python.framework import graph_util


if __name__ == "__main__":
    a = tf.Variable(tf.constant(5., shape=[1]), name="a")
    b = tf.Variable(tf.constant(6., shape=[1]), name="b")
    c = a + b
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    # 导出当前计算图的GraphDef部分
    graph_def = tf.get_default_graph().as_graph_def()

    # 保存指定的节点，并将节点值保存为常数
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])

    #所有通过tf.Variable()明确定义的变量
    for each in tf.all_variables():
        print(each)
    '''
    <tf.Variable 'a:0' shape=(1,) dtype=float32_ref>
    <tf.Variable 'b:0' shape=(1,) dtype=float32_ref>
    '''

    # 将计算图写入到模型文件中
    model_f = tf.gfile.GFile("model.pb", "wb")
    model_f.write(output_graph_def.SerializeToString())