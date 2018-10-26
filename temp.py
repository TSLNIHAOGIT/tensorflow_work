import numpy as np
# x=[[1,2,3],[3,5,6],[6,5,4]]
# y=np.array(x)
# print(y)
# print(y[0:2])

import numpy as np
# np.truncated_normal([2, 5], stddev=0.1)


# x=np.array([1,3,2,4,3,4,5,9,45])
# print(x.reshape(-1,3))
#
# print(list(range(1,5)))
# print(np.reshape(x,(-1,3)))
#
# D = {'one': 1, 'two': 2}
# print(D)
# D.update({'three': 3, 'four': 4})  # 传一个字典
# print(D)
# D.update({'one': 6, 'two': 9})  # 传一个字典
# print(D)

a=np.array([[9,2,8],[4,5,6]])

b=np.array([[11,21,31],[7,8,9]])
c=np.array([[11,21,31],[7,8,9]])
print(a,type(a))
print('max','\n',np.max(a,axis=1))#Returns the indices of the maximum values along an axis,因为向量都是列向量，所以第o纬是对列操作，1维是对行操作
print(b[b>1])
# print(np.concatenate((a,b,c),axis=0))
#
# all_softmax_res=()
# print(tuple(a))
# print('gg',all_softmax_res)

y=np.array([[0.9,0.1],[0.2,0.8],[0.3,0.7]])
print(y)
label=[0,1,1]
for index, each in enumerate(y):
    print(each[label[index]])



print('jj',y[:2])


b=['a']*5
print(b)

