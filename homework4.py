import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import optimizers


def load_cifar10():
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test

    #return xtrain, ytrain_1hot, xtest, ytest_1hot


def build_multilayer_nn():
    pass


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

    # Write any code for testing and evaluation in this main section.


