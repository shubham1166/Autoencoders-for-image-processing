'''
Created on May 20, 2019

This is an example of COnvolutional autoencoder. We have used Keras library and Keras functional API

@author: T01130
'''

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dropout,concatenate,Conv2DTranspose
from keras.models import Model
from keras import backend as K
import keras

## tensboard



# Tensor Board
from os import makedirs
from os.path import exists, join
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np

from FirstPackage.TrainValTensorBoard import TrainValTensorBoard

from keras.datasets import cifar10
import numpy as np
import os

from keras.callbacks import TensorBoard
from time import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"
(x_train, _), (x_test, _) = cifar10.load_data()
x_test = x_test#[:2000]
x_train=x_train#[:10000]



epochs=50;


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# tensorBoard
log_dir = './logs/'
if not exists(log_dir):
    makedirs(log_dir)
# x_train = x_train[1:10000]


#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)

inputs = Input((32, 32, 3))
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
conv10 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv9)






# checkpoint
path = "E:/DATA_DEEPLEARNING_UNET/"
# filepath=path+"test.hdf5"



#compile
autoencoder = Model(inputs=[inputs], outputs=[conv10])
# load pretrained weights
#autoencoder.load_weights(path+'AutoEncoder_Cifar10_Deep_weights.11-0.01-0.01.hdf5')


#adgrd=keras.optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
autoencoder.compile(optimizer='adam', loss='mean_squared_error',metrics=['acc'])

filepath=path+'AutoEncoder_Cifar10_Deep_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

 
 



tensorboard = TensorBoard(log_dir="logs/{}".format(time()))



Listcallbacks = [TrainValTensorBoard(write_graph=True), checkpoint]

autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=80,
                shuffle=True,
                validation_data=(x_test, x_test)
                ,callbacks= Listcallbacks
                )



decoded_imgs = autoencoder.predict(x_test)

print(autoencoder.summary())

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(32, 32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n +1)
    plt.imshow(decoded_imgs[i].reshape(32, 32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


