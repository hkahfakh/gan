# coding:utf-8
'''
DCGAN on MNIST using Keras
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''

import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self, sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time))



def pcaDimensionreduction():
    pass

class DCGAN(object):
    def __init__(self, input_rows=1, input_cols=28):

        self.input_rows = input_rows
        self.input_cols = input_cols
        self.D = None  # discriminator
        self.G = None  # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model
        self.GM = None  # generator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential(name='discriminator')  # 创建一个构造模型

        # In: 1 x 28 x 1, depth = 1    输入通道数
        # Out: 14 x 14 x 1, depth=64   输出通道数
        input_shape = (self.input_rows, self.input_cols)

        self.D.add(Dense(64, input_shape=(16,)))  # 输入 (*, 16) 的数组    输出(*, 32)
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(Activation('relu'))

        self.D.add(Dense(32))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(Activation('relu'))

        self.D.add(Dense(16))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(Activation('relu'))

        self.D.summary()
        return self.D

    def generator(self):
        if self.G:  # 判断是否已经创建了
            return self.G
        self.G = Sequential(name='generator')

        self.G.add(Dense(64, input_shape=(self.input_rows, self.input_cols)))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(Dense(32))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Dense(16))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.summary()  # 打印模型概述信息
        return self.G

    # 损失函数是    没整明白
    def discriminator_model(self):
        if self.DM:
            return self.DM

        optimizer = Adam(lr=0.001, decay=0.9)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    # 损失函数是   类损失+域损失
    def generator_model(self):
        if self.GM:
            return self.GM

        optimizer = Adam(lr=0.001, decay=0.999)
        self.GM = Sequential()
        self.GM.add(self.generator())
        self.GM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.GM

    # 不能简单的将两个模型接在了一起
    # generator-class classier
    #         \discriminator_model=(discriminator + domain classier)
    def adversarial_model(self):
        if self.AM:
            return self.AM
        # optimizer = RMSprop(lr=0.0001, decay=3e-8)
        optimizer = Adam(lr=0.001, decay=0.999)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # 交叉熵就是KL散度
        return self.AM


class MNIST_DCGAN(object):
    def __init__(self):
        self.input_rows = 1
        self.input_cols = 28

        self.x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
        self.x_train = self.x_train.reshape(-1, self.input_rows, self.input_cols, 1).astype(np.float32)

        self.DCGAN = DCGAN()
        self.discriminator = self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval > 0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval > 0:
                if (i + 1) % save_interval == 0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],
                                     noise=noise_input, step=(i + 1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.input_rows, self.input_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()


if __name__ == '__main__':
    # mnist_dcgan = MNIST_DCGAN()
    # mnist_dcgan.train(train_steps=10, batch_size=256, save_interval=500)
    # mnist_dcgan.plot_images(fake=True)
    # mnist_dcgan.plot_images(fake=False, save2file=True)
