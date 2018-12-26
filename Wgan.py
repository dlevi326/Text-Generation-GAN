from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import os
from tqdm import tqdm

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

import sys
sys.path.insert(0, '../gloveLoader/')
sys.path.insert(0, '../dataLoader/')

from loadGlove import loadGloveModel, getKeyedVect
import load_data 

g_cost = []
d_cost = []

print('Downloading data and turning into embeddings')
KEYED_VECTOR = '../gloveloader/glove_50d_keyed.txt'
TRAIN_FILE = '../dataLoader/newsfiles/newsfiles/'
sent_embeddings,keyed_vect = load_data.turn_sents_into_embeddings(KEYED_VECTOR,TRAIN_FILE,num=100)
# Each piece is 15x50
x_train = sent_embeddings
print('Finished downloading...')

class RandomWeightedAverage(_Merge):

    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self):
        self.img_rows = 15
        self.img_cols = 50
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)
        # alternates = tf.map_fn(lambda x: keyed_vect.similar_by_vector(x), fake_img)
        # print(alternates)
        # # for z in alternates:
        # #     print(z)
        # print(fake_img)
        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        interpolated_img = RandomWeightedAverage()([real_img, fake_img])

        validity_interpolated = self.critic(interpolated_img)

        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])

        self.critic.trainable = False
        self.generator.trainable = True


        z_gen = Input(shape=(100,))

        img = self.generator(z_gen)

        valid = self.critic(img)

        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

    def conversion(self, inputs):
        print(tf.map_fn(lambda x: x, inputs),end=' ')

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):

        gradients = K.gradients(y_pred, averaged_samples)[0]

        gradients_sqr = K.square(gradients)

        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))

        gradient_l2_norm = K.sqrt(gradients_sqr_sum)

        gradient_penalty = K.square(1 - gradient_l2_norm)

        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def find_sim_sent(self,sent_mat,keyed_vect):
        zero_vect = np.zeros([1,50])
        for ind,n in enumerate(sent_mat):
            if(n==zero_vect):
                break
            sent_mat[ind] = keyed_vect.similar_by_vector(n)[0][0]
        return sent_mat

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 15 * 50, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((15, 50, 128)))
        #model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        #model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=1, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        
        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=1, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=1, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        # print(img)
        # for x in img:
        #     print(keyed_vect.similar_by_vector(x)[0][0],end=' ')

        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size, sample_interval=50):

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()

        #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = X_train.astype(np.float32)
        # X_train = np.expand_dims(X_train, axis=3)

        #X_train = np.array(x_train).reshape(-1,750)
        X_train = np.expand_dims(x_train, axis = 3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # for x in range(len(noise)):
                #     noise[x][:50] = keyed_vect[keyed_vect.similar_by_vector(noise[x][:50])[0][0]]
                #     noise[x][50:] = keyed_vect[keyed_vect.similar_by_vector(noise[x][50:])[0][0]]

                #noise = self.find_sim_sent(tf.reshape(noise, [-1,64,50]), keyed_vect)
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            g_loss = self.generator_model.train_on_batch(noise, valid)

            d_cost.append(d_loss)
            g_cost.append(g_loss)
            # for x in valid:
            #     print(keyed_vect.similar_by_vector(x[0])[0][0], end=' ')

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
    #         if epoch % sample_interval == 0:
    #             self.sample_images(epoch)
            r, c = 1, 1
            noise = np.random.normal(0, 1, (r * c, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            newSentence = gen_imgs.reshape(15,50)
            for n in newSentence:
                print(keyed_vect.similar_by_vector(n)[0][0],end=' ')


if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=1000, batch_size=32, sample_interval=100)
    plt.plot(d_cost)
    plt.plot(g_cost)
    #plt.xlim([0,10])
    plt.ylim([-1,5])
    plt.xlabel('Loss')
    plt.ylabel('Epochs')
    plt.title('Generator and Discriminator loss with no search during training')
    plt.show()