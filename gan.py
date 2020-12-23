# coding:utf-8
import numpy as np
from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.layers import BatchNormalization
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from getData import get_data
from pca import dimensionReduction

from keras.utils.vis_utils import plot_model
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt


def flatten_list(l):
    return list(np.array(l).flatten())


# 计算期望
def calMean(data):
    return np.sum(data) / data.shape[0]


def gen_Data(path, dim=10, test_size=0.1):
    UNSW_data = get_data(path)
    X, y = UNSW_data[:, :-1], UNSW_data[:, -1]
    X_pca = dimensionReduction(X, dim)
    return train_test_split(X_pca, y, test_size=test_size)


def testClassClassifer(X, y, network):
    tt = np.random.randint(0, X.shape[0], 1)
    t = X[tt]
    t = network.predict(t)
    print("为攻击数据的概率", t)
    print("是否为攻击数据", y[tt])


def testDomainClassifer(X, dataSetName, gen, disc):
    target_data = X[np.random.randint(0, X.shape[0], 1)]
    target_data = gen.predict(target_data)
    print("为源数据集的概率", disc.predict(target_data))
    print("数据集为", dataSetName)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


class GAN(object):
    # 输出的col看降维的维数
    def __init__(self, input_rows=1, input_cols=16):
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.DGxt = None
        self.D = None  # discriminator
        self.G = None  # generator
        self.DM = None  # discriminator model
        self.GM = None  # generator model
        self.AM = None  # adversarial model
        self.CM = None  # classifer model
        self.LL = None

    # 这意思应该是生成16位的特征   供辨别器分辨
    def generator(self):
        if self.G:  # 判断是否已经创建了
            return self.G
        self.G = Sequential(name='generator')
        # self.G.add(Input())
        self.G.add(Dense(64, input_shape=(10,)))
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

    # 目标 + 源
    def discriminator_loss(self, y_true, y_pred):
        a = -K.mean(K.log(y_pred)[16:])
        b = K.log(1 - y_pred)[:16]
        return a + b

    # 这个是类分类 的loss    所以真实值预测值   是是否为恶意攻击的概率
    def generator_loss(self, y_true, y_pred):

        L_class = -K.mean(y_true * (K.log(y_pred))) + (1 - y_true) * (K.log(1 - y_pred))
        L_domain = -K.mean(K.log(self.DGxt))  # 这个地方不能用y_pred
        return (L_class + L_domain) / 2

    # 输出的是正常或恶意数据的概率
    def generator_model(self):
        if self.GM:
            return self.GM

        self.GM = Sequential(name="generator_model")
        self.GM.add(self.generator())
        return self.GM

    def classifer_model(self):
        if self.CM:
            return self.CM

        optimizer = Adam(lr=0.001, decay=0.999)
        self.CM = Sequential(name="classifer_model")
        self.CM.add(self.generator())
        self.CM.add(Dense(1, activation='sigmoid'))  # 二分类所以加了这个
        self.CM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # plot_model(self.CM, to_file='output/classifer.png', show_shapes=True)
        return self.CM

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

    def adversarial_model(self):
        if self.AM:
            return self.AM

        optimizer = Adam(lr=0.001, decay=0.999)
        self.AM = Sequential(name="adversarial_model")
        self.AM.add(self.generator())
        self.AM.add(self.discriminator_model())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # 交叉熵就是KL散度
        return self.AM


def train_GAN(gan, source_dataSet, target_dataSet, epochs=100, batch_size=32, loadWeights=1):
    # Loading the data
    UNSW_X_train, UNSW_X_test, UNSW_y_train, UNSW_y_test = target_dataSet
    KDD_X_train, KDD_X_test, KDD_y_train, KDD_y_test = source_dataSet

    generator = gan.generator_model()
    discriminator = gan.discriminator_model()
    gan_model = gan.adversarial_model()

    his_loss1 = list()
    his_acc1 = list()
    his_loss2 = list()
    his_acc2 = list()
    if loadWeights == 1:  # genertor是共享权重的 权重相同 可以打印权重确认
        gan_model.load_weights('./recovery/gan_weight.h5')
    else:
        for e in range(0, epochs):
            lower_bound = int(e * batch_size / 2)
            upper_bound = int((e + 1) * batch_size / 2)
            # 从UNSW中获取这次训练的数据集  为批的一半  predict是为了转化为不变域上的特征  送给辨别器
            target_data = UNSW_X_train[lower_bound:upper_bound]
            source_data = KDD_X_train[lower_bound:upper_bound]
            X = np.concatenate([target_data, source_data])

            label_target = np.zeros(int(batch_size / 2))  # 目标域是0
            label_source = np.ones(int(batch_size / 2))
            y = np.concatenate([label_target, label_source])

            # 训练辨别器
            X_d = generator.predict(X)
            generator.trainable = False
            discriminator.trainable = True
            h1 = discriminator.fit(X_d, y,
                                   initial_epoch=e,
                                   epochs=e + 1,
                                   verbose=1)  # 损失nan
            his_acc1.append(h1.history['accuracy'])
            his_loss1.append(h1.history['loss'])

            # 训练生成器(此时冻结辨别器权重)
            generator.trainable = True
            discriminator.trainable = False
            h2 = gan_model.fit(X, np.ones(int(batch_size)),
                               initial_epoch=e,
                               epochs=e + 1,
                               verbose=1, )
            # validation_data=(UNSW_X_test, np.zeros(UNSW_X_test.shape[0])))
            his_acc2.append(h2.history['accuracy'])
            his_loss2.append(h2.history['loss'])

        plot_cruse(flatten_list(his_acc1), flatten_list(his_loss1))
        plot_cruse(flatten_list(his_acc2), flatten_list(his_loss2))
        gan_model.save_weights('./output/gan_weight.h5')


    a = np.concatenate([UNSW_X_test[:3200], KDD_X_test[:3200]])
    b = np.concatenate([np.zeros(3200), np.ones(3200)])
    score = gan_model.evaluate(a, b, batch_size=32)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def plot_cruse(his_acc, his_loss):
    epochs = range(1, len(his_acc) + 1)
    plt.title('Accuracy and Loss')
    plt.plot(epochs, his_acc, 'red', label='Training acc')
    plt.plot(epochs, his_loss, 'blue', label='Validation loss')
    plt.legend()
    plt.show()


# NSL-KDD作为源数据集，UNSW-NB15作为目标数据集
# 两个数据集都转换到一个特征空间
if __name__ == '__main__':
    source_dataSet = gen_Data("./dataSet/KDD_finally.npy")
    target_dataSet = gen_Data("./dataSet/UNSW_finally.npy")

    gan = GAN()
    train_GAN(gan, source_dataSet, target_dataSet)
