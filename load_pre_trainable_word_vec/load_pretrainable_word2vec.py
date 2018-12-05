import numpy as np
import tensorflow as tf
def loadWord2Vec(filename):
    vocab = []
    embd = []
    cnt = 0
    fr = open(filename,'r')
    line = fr.readline().decode('utf-8').strip()
    #print line
    word_dim = int(line.split(' ')[1])
    vocab.append("unk")
    embd.append([0]*word_dim)
    for line in fr :
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print ("loaded word2vec")
    fr.close()
    return vocab,embd
filename=''
vocab,embd = loadWord2Vec()
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd)

W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                trainable=False, name="W")
embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
embedding_init = W.assign(embedding_placeholder)
