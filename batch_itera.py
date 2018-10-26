batch_size=7
dataset_size=128

# [(step*batch_size):((step+1)*batch_size)]
for step in range(30):
    start=(step*batch_size)%dataset_size #小于dataset_size时，就是前面的值；大于batch_size时看余数是多少
    end=min(start+batch_size,dataset_size)
    print(step,(start,end))


'''
0 7
7 14
14 21
21 28
28 35
35 42

119 126
126 128
***
#超出之后就不是从0开始了
5 12
12 19
19 26
***
40 47
47 54
54 61
61 68
68 75
75 82


'''
