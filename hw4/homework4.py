import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import optimizers

def load_cifar10():
    from keras.utils import to_categorical
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test

    # normalize data
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xtrain /= 255
    xtest /= 255


    # encode data to 1hot
    ytrain_1hot = to_categorical(np.array(ytrain))
    ytest_1hot = to_categorical(np.array(ytest))

    return xtrain, ytrain_1hot, xtest, ytest_1hot


def build_multilayer_nn():
    nn = Sequential()
    nn.add(Flatten(input_shape=(32,32,3)))
    nn.add(Dense(units=100, activation="relu", input_shape=(3072,)))
    nn.add(Dense(units=10, activation="softmax"))
    return nn

def train_multilayer_nn(model, xtrain, ytrain):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain_1hot, epochs=30, batch_size=32)


def build_convolution_nn():
    pass


def train_convolution_nn():
    pass


def get_binary_cifar10():
    pass


def build_binary_classifier():
    pass


def train_binary_classifier():
    pass


if __name__ == "__main__":
    xtrain, ytrain_1hot, xtest, ytest_1hot = load_cifar10()
    nn = build_multilayer_nn()
    nn.summary()
    train_multilayer_nn(nn, xtrain, ytrain_1hot)
    nn.evaluate(xtest, ytest_1hot)
