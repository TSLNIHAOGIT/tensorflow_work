import tensorflow as tf

def save_eg():
    v=tf.Variable(0,dtype=tf.float32,name='v')
    v1=tf.Variable(0,dtype=tf.float32,name='v1')
    # result=v + v1
    # v2=2.3*v
    for variables in tf.all_variables():#all_variables,只会显示tf.Variable明确定义的变量，
        print('variables.name0',variables.name)
    ema=tf.train.ExponentialMovingAverage(0.99)
    maintain_average_op=ema.apply(tf.all_variables())#对所有变量做滑动平均
    for variables in tf.all_variables():
        print('variables.name1',variables.name)
    saver=tf.train.Saver()
    with tf.Session( )  as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        sess.run(tf.assign(v,10))
        sess.run(tf.assign(v1,20))
        sess.run(maintain_average_op)
        saver.save(sess,'average_model.ckpt')
        print(sess.run([v,ema.average(v)]))#明确调用ema的average获取影子变量的取值
    # return result

def load_rename():
    # result=save_eg()
    v1 = tf.Variable(0, dtype=tf.float32, name='v1')
    v = tf.Variable(0, dtype=tf.float32, name='v')
    # r=result+v3+v4
    ema = tf.train.ExponentialMovingAverage(0.99)
    print(ema.variables_to_restore())
    saver=tf.train.Saver(ema.variables_to_restore())#所有包含影子变量重命名的字典，将影子变量名称变为原始变量名称
    with tf.Session() as sess:
        saver.restore(sess,'average_model.ckpt')
        print(sess.run([v,v1]))

# def to_constants():

if __name__=='__main__':
    # save_eg()
    load_rename()
