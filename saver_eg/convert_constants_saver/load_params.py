import tensorflow as tf
from tensorflow.python.platform import gfile
# sess = tf.Session()
# # 将保存的模型文件解析为GraphDef
# model_f = gfile.FastGFile("model.pb", 'rb')
# graph_def = tf.GraphDef()
# graph_def.ParseFromString(model_f.read())
# c = tf.import_graph_def(graph_def, return_elements=["add:0"])
# print(c)
#
# for each in tf.global_variables():
#     print('each',each)
# print(sess.run(c))
# # [array([ 11.], dtype=float32)]


with tf.Session() as sess:
	print("load graph")
	with gfile.FastGFile("model.pb",'rb') as f:
		graph_def = tf.GraphDef()
    # Note: one of the following two lines work if required libraries are available
		#text_format.Merge(f.read(), graph_def)
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name='')
		for i,n in enumerate(graph_def.node):
			print("Name of the node - %s" % n.name)
