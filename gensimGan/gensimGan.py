# Mnist GAN

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import os
from tqdm import tqdm

import sys
sys.path.insert(0, '../gloveLoader/')
sys.path.insert(0, '../dataLoader/')

from loadGlove import loadGloveModel, getKeyedVect
import load_data 


print('Downloading data and turning into embeddings')
KEYED_VECTOR = '../gloveloader/glove_50d_keyed.txt'
TRAIN_FILE = '../dataLoader/newsfiles/technology/'
sent_embeddings,keyed_vect = load_data.turn_sents_into_embeddings(KEYED_VECTOR,TRAIN_FILE,num=1000)
# Each piece is 15x50
x_train = sent_embeddings
print('Finished downloading...')



r = 0
def next_batch(data,size):
    global r
    if r*size+size > len(data):
        r=0
    x_train_batch = data[size*r:r*size+size, :]
    r = r+1
    return x_train_batch

def find_sim_sent(sent_mat,keyed_vect):
    zero_vect = np.zeros([1,50])
    for ind,n in enumerate(sent_mat):
        if(n==zero_vect):
            break
        sent_mat[ind] = keyed_vect.similar_by_vector(n)[0][0]
    return sent_mat

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))

def init_bias(shape):
    return tf.Variable(tf.constant(0.2, shape=shape))

class Generator:
    def __init__(self):
        with tf.variable_scope('g'):
            self.gW1 = init_weights([100,256])
            self.gb1 = init_bias([256])
            self.gW2 = init_weights([256,750])
            self.gb2 = init_bias([750])

    def forward(self, z, training=True):
        fc1 = tf.matmul(z, self.gW1) + self.gb1
        fc1 = tf.layers.batch_normalization(fc1, training = training)
        fc1 = tf.nn.leaky_relu(fc1)
        fc2 = (tf.matmul(fc1, self.gW2)+self.gb2)
        return fc2

class Discriminator:
    def __init__(self):
        with tf.variable_scope('d'):
            self.dW1 = init_weights([5,5,1,16])
            self.db1 = init_bias([16])
            self.dW2 = init_weights([3,3,16,32])
            self.db2 = init_bias([32])

            self.W3 = init_weights([1500,128])
            self.b3 = init_bias([128])
            self.W4 = init_weights([128,1])
            self.b4 = init_bias([1])

    def forward(self, X):
        self.X = tf.reshape(X, shape=[-1,15,50,1])
        conv1 = tf.nn.leaky_relu(tf.nn.conv2d(self.X, self.dW1, strides=[1,1,1,1], padding='SAME') + self.db1)
        conv1 = tf.contrib.layers.batch_norm(conv1,trainable=True)
        conv2 = tf.nn.leaky_relu(tf.nn.conv2d(conv1, self.dW2, strides=[1,1,1,1], padding='SAME')+self.db2)
        conv2 = tf.contrib.layers.batch_norm(conv2, trainable=True)
        conv2 = tf.reshape(conv2, shape=[-1,15*50*2])

        fc1 = tf.nn.leaky_relu(tf.matmul(conv2,self.W3)+self.b3)
        logits = tf.matmul(fc1, self.W4)+self.b4
        fc2 = tf.nn.sigmoid(logits)

        return fc2, logits

def cost(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

d = Discriminator()
g = Generator()

phX = tf.placeholder(tf.float32, [None,750])
phZ = tf.placeholder(tf.float32, [None, 100])

G_out = g.forward(phZ)
G_out_sample = g.forward(phZ, False)


D_out_real, D_logits_real = d.forward(phX)
D_out_fake, D_logits_fake = d.forward(G_out)

D_real_loss = cost(D_logits_real, tf.ones_like(D_logits_real))
D_fake_loss = cost(D_logits_fake, tf.zeros_like(D_logits_fake))

D_loss = D_real_loss + D_fake_loss
G_loss = cost(D_logits_fake, tf.ones_like(D_logits_fake))

# Init learning rate
lr = 0.001



# Pretraining epochs out of total epochs [After pretraining images are generated]
pretrain_epochs = 1000
batch_size = 50

# Epochs per label
epochs = pretrain_epochs+100

train_vars = tf.trainable_variables()

dvars = [var for var in train_vars if 'd' in var.name]
gvars = [var for var in train_vars if 'g' in var.name]

D_train = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=dvars)
G_train = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=gvars)

init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    
    sess.run(init)

    data = np.array(x_train).reshape(-1,750)
    # Get data for specific label
    
    g_cost = []
    d_cost = []

    k=0
    for j in tqdm(range(epochs)):
        batch_x = next_batch(data,batch_size)

        batch_z = np.random.randn(batch_size,100)

        # Training discriminator
        _, d_loss = sess.run([D_train, D_loss], feed_dict={phX:batch_x,phZ: batch_z})

        # Training generator
        _, g_loss = sess.run([G_train, G_loss], feed_dict={phZ:batch_z})

        # Applied loss for later plotting
        d_cost.append(d_loss)
        g_cost.append(g_loss)

        # Generating images
        if j%10 ==0 and j>=pretrain_epochs:
            final_sent = []
            sample_z = np.random.randn(1,100)*10
            #sample_z = np.zeros([1,100])

            gen_sample = sess.run(G_out_sample, feed_dict={phZ:sample_z})

            # Print iteration and costs
            print(f'Iteration {j}. G_loss {g_loss}. D_loss {d_loss}')

            # save image
            newSentence = gen_sample.reshape(15,50)
            for n in newSentence:
                print(keyed_vect.similar_by_vector(n)[0][0],end=' ')
                final_sent.append(n)
            print()
    plt.plot(d_cost)
    plt.plot(g_cost)
    #plt.xlim([0,10])
    #plt.ylim([-1,5])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Generator and Discriminator loss with no search during training')
    plt.show()

print('done')

#print('but',keyed_vect['but'])#keyed_vect.similar_by_vector('hello')[0])

#for i in final_sent:
#    print(keyed_vect.similar_by_vector(i)[0][0],' ',i)