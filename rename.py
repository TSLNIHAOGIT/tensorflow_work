import tensorflow as tf

def save_eg():
    vv=tf.Variable(0,dtype=tf.float32,name='v')
    vv1=tf.Variable(0,dtype=tf.float32,name='v1')
    # result=v + v1
    # v2=2.3*v
    for variables in tf.all_variables():#all_variables,只会显示tf.Variable明确定义的变量，
        print('variables.name0',variables.name)
        '''
        variables.name1 v:0
        variables.name1 v1:0
        '''
    ema=tf.train.ExponentialMovingAverage(0.99)
    maintain_average_op=ema.apply(tf.all_variables())#对所有变量做滑动平均
    for variables in tf.all_variables():
        print('variables.name1',variables.name)
        '''
        variables.name1 v:0
        variables.name1 v1:0
        variables.name1 v/ExponentialMovingAverage:0
        variables.name1 v1/ExponentialMovingAverage:0
        '''

    saver=tf.train.Saver()
    with tf.Session( )  as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        sess.run(tf.assign(vv,10))
        sess.run(tf.assign(vv1,20))

        sess.run(maintain_average_op)
        saver.save(sess,'/Users/ozintel/Downloads/Tsl_python_progect/tensor_project/tensorflow_work/average_model.ckpt')
        print(sess.run([vv,ema.average(vv)]))#明确调用ema的average获取影子变量的取值
    # return result
#变量加载时变量重命名有两种方式键为影子变量名称，值可以新变量或者"新变量名称:输出序号",此时新变量名称要和生成影子变量的原始变量名称相同
def load_rename():
    # result=save_eg()
    vv1 = tf.Variable(0, dtype=tf.float32, name='v1')
    vv2 = tf.Variable(0, dtype=tf.float32, name='v')
    for variables in tf.all_variables():#all_variables,只会显示tf.Variable明确定义的变量，
        print('variables.name0',variables.name)
    # r=result+v3+v4
    ema = tf.train.ExponentialMovingAverage(0.99)
    print(ema.variables_to_restore())#加载时v\v1都会生成相应的字典，用滑动平均值替代原来的变量值
    # 名称v1/ExponentialMovingAverage的变量加载到'v1:0'中，用滑动平均值替代原来的变量值
    #{'v1/ExponentialMovingAverage': <tf.Variable 'v1:0' shape=() dtype=float32_ref>,
    # 'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}

    # rename_dict={'v1/ExponentialMovingAverage':vv1,'v/ExponentialMovingAverage':vv2}
    for variables in tf.all_variables():#all_variables,只会显示tf.Variable明确定义的变量，
        print('variables.name1',variables.name)
    saver=tf.train.Saver(
        # rename_dict
        ema.variables_to_restore()
    )#所有包含影子变量重命名的字典，将影子变量名称变为原始变量名称
    with tf.Session() as sess:
        saver.restore(sess,'average_model.ckpt')
        print(sess.run([vv1,vv2]))

# def to_constants():

if __name__=='__main__':
    # save_eg()
    load_rename()
