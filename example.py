#import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from keras.datasets import mnist
from keras.datasets import cifar10
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import openml as oml
from keras.models import load_model
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
import openml as oml
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
oml.config.apikey = 'b15b073c6fea6dc55b08f051f5e1abf9'
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from keras.layers import Reshape

task = oml.tasks.get_task(3573)
model = Sequential()
model.add(Reshape((28, 28, 1), input_shape=(784,)))

# first set of CONV => RELU => POOL
model.add(Conv2D(20, (5, 5), padding="same",
input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# second set of CONV => RELU => POOL
model.add(Conv2D(50, (5, 5), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# set of FC => RELU layers
model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
model.add
# softmax classifier
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print(model.summary())
run = oml.runs.run_task(task, model)
