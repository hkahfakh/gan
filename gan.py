# coding:utf-8
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape, Input
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from getData import get_data
from pca import dimensionReduction
from tqdm import tqdm

from keras.utils.vis_utils import plot_model


class GAN(object):
    # 输出的col看降维的维数
    def __init__(self, input_rows=1, input_cols=16):
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.D = None  # discriminator
        self.G = None  # generator
        self.DM = None  # discriminator model
        self.GM = None  # generator model
        self.AM = None  # adversarial model
        self.CM = None  # classifer model

    # 这意思应该是生成16位的特征   供辨别器分辨
    def generator(self):
        if self.G:  # 判断是否已经创建了
            return self.G
        self.G = Sequential(name='generator')

        self.G.add(Dense(64, input_shape=(16,)))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Dense(32))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Dense(16))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        return self.G

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D

        self.D = Sequential(name='discriminator')  # 创建一个构造模型

        self.D.add(Dense(64, input_shape=(16,)))  # 输入 (*, 16) 的数组    输出(*, 32)
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(Activation('relu'))

        self.D.add(Dense(32))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(Activation('relu'))

        self.D.add(Dense(16))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(Activation('relu'))

        return self.D

    def domain_loss(self):
        pass

    def class_loss(self):
        pass

    # 输出的是正常或恶意数据的概率
    def generator_model(self):
        if self.GM:
            return self.GM

        optimizer = Adam(lr=0.001, decay=0.999)
        self.GM = Sequential(name="generator_model")
        self.GM.add(self.generator())
        self.GM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.GM

    # 损失函数是    没整明白
    def discriminator_model(self):
        if self.DM:
            return self.DM

        optimizer = Adam(lr=0.001, decay=0.9)
        self.DM = Sequential(name="discriminator_model")
        self.DM.add(self.discriminator())
        self.DM.add(Dense(1, activation='sigmoid'))  # 二分类所以加了这个
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # 交叉熵就是KL散度
        return self.DM

    # 不能简单的将两个模型接在了一起
    # generator-class classier
    # generator-discriminator_model=(discriminator + domain classier)
    def adversarial_model(self):
        if self.AM:
            return self.AM

        self.DM.trainable = False
        GAN_input = Input(shape=(16,))
        x = self.GM(GAN_input)
        GAN_output = self.DM(x)
        self.AM = Model(inputs=GAN_input, outputs=GAN_output)
        self.AM.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
        return self.AM

    def classifer_model(self):
        if self.CM:
            return self.CM

        optimizer = Adam(lr=0.001, decay=0.999)
        self.CM = Sequential(name="classifer_model")
        self.CM.add(self.generator())
        self.CM.add(Dense(1, activation='sigmoid'))  # 二分类所以加了这个
        self.CM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.CM


def train_GAN(gan, epochs=300, batch_size=32, loadWeights=1):
    # Loading the data
    UNSW_data = get_data("./dataSet/UNSW_finally.npy")
    X, y = UNSW_data[:, :-1], UNSW_data[:, -1]
    X_pca = dimensionReduction(X, 16)
    UNSW_X_train, UNSW_X_test, UNSW_y_train, UNSW_y_test = train_test_split(X_pca, y,
                                                                            test_size=0.2)

    KDD_data = get_data("./dataSet/KDD_finally.npy")
    X, y = KDD_data[:, :-1], KDD_data[:, -1]
    X_pca = dimensionReduction(X, 16)
    KDD_X_train, KDD_X_test, KDD_y_train, KDD_y_test = train_test_split(X_pca, y, test_size=0.2)

    # Creating GAN
    generator = gan.generator_model()
    classification = gan.classifer_model()
    discriminator = gan.discriminator_model()
    GAN = gan.adversarial_model()

    if loadWeights == 1:
        GAN.load_weights("./output/gan_weight.h5", by_name=True)
        classification.load_weights('./output/class_classifer_weight.h5')
    else:
        for i in range(1, epochs + 1):
            print("Epoch %d" % i)

            for _ in tqdm(range(batch_size)):
                # Generate fake images from random noiset
                target_data = UNSW_X_train[np.random.randint(0, UNSW_X_train.shape[0], batch_size)]
                target_data = generator.predict(target_data)
                # Select a random batch of real images from MNIST
                temp = np.random.randint(0, KDD_X_train.shape[0], batch_size)
                class_X_train, class_y_train = KDD_X_train[temp], KDD_y_train[temp]
                source_data = KDD_X_train[temp]
                source_data = generator.predict(source_data)

                # 假的图片为标签一串0   真的图片标签一串1
                label_target = np.zeros(batch_size)
                label_source = np.ones(batch_size)

                # Concatenate fake and real images
                X = np.concatenate([target_data, source_data])
                y = np.concatenate([label_target, label_source])

                # 训练辨别器
                generator.trainable = False
                discriminator.trainable = True
                discriminator.train_on_batch(X, y)

                # 训练生成器(此时冻结辨别器权重)
                generator.trainable = True
                discriminator.trainable = False
                GAN.train_on_batch(target_data, label_source)  # 就是训练生成器尽量使辨别器输出为1，也就是让辨别器一位他是源数据
                classification.train_on_batch(class_X_train, class_y_train)

        classification.save_weights('./output/class_classifer_weight.h5')
        GAN.save_weights('./output/gan_weight.h5')

    tt = np.random.randint(0, KDD_X_train.shape[0], 1)
    t = KDD_X_train[tt]
    t = classification.predict(t)
    print("为攻击数据的概率", t)
    print("是否为攻击数据", KDD_y_train[tt])

    target_data = UNSW_X_train[np.random.randint(0, UNSW_X_train.shape[0], 1)]
    target_data = generator.predict(target_data)
    print("为源数据集的概率", discriminator.predict(target_data))


# NSL-KDD作为源数据集，UNSW-NB15作为目标数据集
# 两个数据集都转换到一个特征空间
if __name__ == '__main__':
    train_GAN(GAN())
