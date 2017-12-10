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

'''
output of evaluate
[loss, accuracy]
[1.5132857789993286, 0.4768]
'''
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

'''
output of evaluate
[loss, accuracy]
[0.84072636938095091, 0.70709999999999995]
'''
def build_convolution_nn():
    nn = Sequential()

    # 2 Convolution 32 filter size 3x3
    nn.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

    # 1 pooling 16x16
    nn.add(MaxPooling2D(pool_size=(2, 2)))

    # 1 dropout .25
    nn.add(Dropout(0.25))

    # 2 Convolution 32 filter size 3x3
    nn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

    # 1 pooling 8x8 output shape 8x8x32
    nn.add(MaxPooling2D(pool_size=(2, 2)))

    # 1 dropout .5
    nn.add(Dropout(0.5))

    # regular multilayer network
    # two hidden layers 250, 100
    nn.add(Flatten())
    nn.add(Dense(units=250, activation='relu'))
    nn.add(Dense(units=100, activation='relu'))

    # output layer 10
    nn.add(Dense(units=10, activation="softmax"))

    return nn

def train_convolution_nn(model, xtrain, ytrain):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # training on 30 epochs allowed the CNN to reach 70% accuracy
    model.fit(xtrain, ytrain_1hot, epochs=30, batch_size=32)

def get_binary_cifar10():
    pass


def build_binary_classifier():
    pass


def train_binary_classifier():
    pass


if __name__ == "__main__":
    xtrain, ytrain_1hot, xtest, ytest_1hot = load_cifar10()

    nn = build_multilayer_nn()
    #nn = build_convolution_nn()

    nn.summary()
    train_multilayer_nn(nn, xtrain, ytrain_1hot)
    print(nn.evaluate(xtest, ytest_1hot))

