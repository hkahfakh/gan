# coding:utf-8
import numpy as np

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


class GAN(object):
    # 输出的col看降维的维数
    def __init__(self, input_rows=1, input_cols=16):
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.E_source_mean = 0
        self.E_target_mean = 0
        self.E_attack_mean = 0
        self.E_benign_mean = 0
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

    def class_loss(self, y_true, y_pred):
        E_attack = np.float32(self.E_attack_mean)
        E_benign = np.float32(self.E_benign_mean)
        E_attack = tf.convert_to_tensor(E_attack)
        E_benign = tf.convert_to_tensor(E_benign)
        a = -E_attack * (K.log(y_pred)) - E_benign * (K.log(1 - y_pred))
        return a

    def domain_loss(self, y_true, y_pred):
        E_source = self.E_source_mean
        E_target = self.E_target_mean
        E_source = tf.convert_to_tensor(E_source)
        E_target = tf.convert_to_tensor(E_target)
        return -E_source * (K.log(y_pred)) - E_target * (K.log(1 - y_pred))

    def binary_crossentropy(self, y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

    # 输出的是正常或恶意数据的概率
    def generator_model(self):
        if self.GM:
            return self.GM

        optimizer = Adam(lr=0.001, decay=0.999)
        self.GM = Sequential(name="generator_model")
        self.GM.add(self.generator())
        self.GM.compile(loss=self.class_loss,
                        optimizer=optimizer,
                        metrics=['accuracy'])
        return self.GM

    # 损失函数是    没整明白
    def discriminator_model(self):
        if self.DM:
            return self.DM

        optimizer = Adam(lr=0.001, decay=0.9)
        self.DM = Sequential(name="discriminator_model")
        self.DM.add(self.discriminator())
        self.DM.add(Dense(1, activation='sigmoid'))  # 二分类所以加了这个
        self.DM.compile(loss=self.domain_loss, optimizer=optimizer, metrics=['accuracy'])  # 交叉熵就是KL散度
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
        plot_model(self.CM, to_file='output/classifer.png', show_shapes=True)
        return self.CM


# 计算期望
def calMean(data):
    return np.sum(data) / data.shape[0]


def gen_Data(path, dim=16, test_size=0.2):
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


def train_GAN(gan, source_dataSet, target_dataSet, epochs=300, batch_size=32, loadWeights=1):
    # Loading the data
    UNSW_X_train, UNSW_X_test, UNSW_y_train, UNSW_y_test = target_dataSet
    KDD_X_train, KDD_X_test, KDD_y_train, KDD_y_test = source_dataSet

    # Creating GAN
    generator = gan.generator_model()
    classification = gan.classifer_model()
    discriminator = gan.discriminator_model()
    GAN = gan.adversarial_model()

    if loadWeights == 1:  # genertor是共享权重的 权重相同 可以打印权重确认
        GAN.load_weights("./output/gan_weight.h5", by_name=True)
        classification.load_weights('./output/class_classifer_weight.h5')
    else:
        for e in range(0, epochs):
            lower_bound = int(e * batch_size / 2)
            upper_bound = int((e + 1) * batch_size / 2)
            # 从UNSW中获取这次训练的数据集  为批的一半  predict是为了转化为不变域上的特征  送给辨别器
            target_data = UNSW_X_train[lower_bound:upper_bound]
            target_data = generator.predict(target_data)

            # 从KDD中获取这次训练的数据集  为批的一半
            source_data = KDD_X_train[lower_bound:upper_bound]
            source_data = generator.predict(source_data)

            # 这个是训练classifer的
            class_X_train, class_y_train = KDD_X_train[
                                           lower_bound:upper_bound], KDD_y_train[lower_bound:upper_bound]
            # 假的图片为标签一串0   真的图片标签一串1
            label_target = np.zeros(int(batch_size / 2))
            label_source = np.ones(int(batch_size / 2))

            # Concatenate fake and real images
            X = np.concatenate([target_data, source_data])
            y = np.concatenate([label_target, label_source])

            # 训练辨别器
            generator.trainable = False
            discriminator.trainable = True
            discriminator.train_on_batch(X, y)
            discriminator.fit(X, y, initial_epoch=e, epochs=e + 1, verbose=1)

            # 训练生成器(此时冻结辨别器权重)
            generator.trainable = True
            discriminator.trainable = False
            GAN.fit(target_data, label_source, initial_epoch=e, epochs=e + 1, verbose=1)
            classification.fit(class_X_train, class_y_train, initial_epoch=e, epochs=e + 1, verbose=1)

            # if epochs % 10 == 0:
            #     classification.save_weights('./output/class_classifer_weight.h5')
            #     GAN.save_weights('./output/gan_weight.h5')

        classification.save_weights('./output/class_classifer_weight.h5')
        GAN.save_weights('./output/gan_weight.h5')

    testClassClassifer(KDD_X_train, KDD_y_train, classification)
    testDomainClassifer(UNSW_X_train, "UNSW_X_train", generator, discriminator)


# NSL-KDD作为源数据集，UNSW-NB15作为目标数据集
# 两个数据集都转换到一个特征空间
if __name__ == '__main__':
    source_dataSet = gen_Data("./dataSet/KDD_finally.npy")
    target_dataSet = gen_Data("./dataSet/UNSW_finally.npy")

    gan = GAN()
    gan.E_source_mean = 1 / 2
    gan.E_target_mean = 1 / 2
    a = np.concatenate([source_dataSet[2], target_dataSet[2]])
    gan.E_attack_mean = calMean(a)
    a = 1 - a
    gan.E_benign_mean = calMean(a)
    train_GAN(gan, source_dataSet, target_dataSet)
