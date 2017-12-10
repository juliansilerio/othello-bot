'''
Julian Silerio
jjs2245
12/10/2017

Programming 4
'''

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
    model.fit(xtrain, ytrain, epochs=30, batch_size=32)

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
    model.fit(xtrain, ytrain, epochs=30, batch_size=32)

def get_binary_cifar10():
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test

    # normalize data
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xtrain /= 255
    xtest /= 255

    ytrain_binary = []
    ytest_binary = []

    for label in ytrain:
        if label < 2 or label > 7:
            ytrain_binary.append(0)
        else:
            ytrain_binary.append(1)

    for label in ytest:
        if label < 2 or label > 7:
            ytest_binary.append(0)
        else:
            ytest_binary.append(1)

    return xtrain, ytrain_binary, xtest, ytest_binary

'''
output
[loss, accuracy]
[0.19410956881046296, 0.92200000000000004]

SHORT ANSWER LINE 174-5
'''
def build_binary_classifier():
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
    nn.add(Dense(units=1, activation="sigmoid"))

    return nn


def train_binary_classifier(model, xtrain, ytrain):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=20, batch_size=32)

'''
I felt that binary classification was significantly easier than categorical simply because there weren't as many label choices for thhis task. Because there were only two labels, each label had a higher probability of being correct, and the relative accuracy of the categorical neural network at ~70% suggests that the images are already feasibly classifiably into 2 categories, so naturally a reduction in the number of labels would mean that the pictures could be even better sorted.
Maybe the fact that lots of animals don't look the same (frog and bird??) could affect how well binary classification works, but I think this effect is negligble because the difference between animals overall and vehicles is pretty noticeable. The grays and muted color palette of vehicles pales in comparison to the natural qualities of animals, and accordingly binary classification more accurately evaluates the test data than the categorical model.
'''

if __name__ == "__main__":
    #xtrain, ytrain, xtest, ytest = load_cifar10()
    xtrain, ytrain, xtest, ytest = get_binary_cifar10()

    #nn = build_multilayer_nn()
    #nn = build_convolution_nn()
    nn = build_binary_classifier()

    nn.summary()
    #train_multilayer_nn(nn, xtrain, ytrain)
    #train_convolution_nn(nn, xtrain, ytrain)
    train_binary_classifier(nn, xtrain, ytrain)

    print(nn.evaluate(xtest, ytest))

