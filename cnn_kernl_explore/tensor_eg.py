import numpy as np

# filter2 = np.array(
# [
#   [  [[2,1],[3,5]],[[1,6],[5,3]]  ],
#  #
#   [  [[3,3],[1,4]],[[4,2],[2,5]]  ]
# ])
#
# filter=filter2
# print('filter.shape',filter.shape,'\n',filter)#(2, 2, 2, 1)
# y = np.zeros((2, 2, 2,2))
# print(y.shape)
# print(y)

# x0=[
#   [[0 ,0 ,0 ,0],
#    [0 ,0 ,0 ,0],
#    [0 ,0 ,0 ,0]]
# ,
#   [[0 ,0 ,0 ,0],
#    [0 ,0 ,0 ,0],
#    [0 ,0 ,0 ,0]]
#     ]
# x0=np.array(x0)
# print(x0.shape)#(2, 3, 4)
#
# b=np.array([x0,x0])
# print(b.shape,'\n',b)#(2, 2, 3, 4)

x=[
   [
  [[0 ,0 ,0 ,0],
   [0 ,0 ,0 ,0],
   [0 ,0 ,0 ,0]]
,
  [[0 ,0 ,0 ,0],
   [0 ,0 ,0 ,0],
   [0 ,0 ,0 ,0]]
    ]

]
# x=np.array(x)
# print(x.shape)#(1,2, 3, 4)
# print(x)
# (2, 2, 3, 4)

# [[
#   [[0 0 0 0]
#    [0 0 0 0]
#    [0 0 0 0]]
#
#   [[0 0 0 0]
#    [0 0 0 0]
#    [0 0 0 0]]]
#
#
#  [[[0 0 0 0]
#    [0 0 0 0]
#    [0 0 0 0]]
#
#   [[0 0 0 0]
#    [0 0 0 0]
#    [0 0 0 0]]]]



# x=\
# np.array([
#
#     [[1,2,3,4],[2,3,4,5],[3,4,5,6]],
#    [[12,2,3,1],[2,5,2,2],[3,4,5,3]],
#    [[13,4,5,6],[2,6,4,3],[3,5,5,2]],
#
# ]
# )
# print(x.shape)#(3, 3, 4)
# print(x[:,1,:])
# '''[[2 3 4 5]
#  [2 5 2 2]
#  [2 6 4 3]]'''

Z=np.array(
[[2 ,3 ,4 ,5],
 [2, 5, 2 ,2],
 [2, 6, 4 ,3]
 ])

print(Z.shape)
print(Z[0,0:2])
print(Z[0:2,1:3])#以逗号分割，分别选取第一个纬度，第二个纬度，且每个纬度可有变化范围的
#
X = np.random.randn(2, 3, 4)
# mask=np.array()
print(X)
#
# X[1,1:,1:3]=0
#
# print('**\n',X)
# X[1,1:]=0
# print('**\n',X)
print('**\n')
print(X[:,0,:])