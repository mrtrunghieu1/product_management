import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten,MaxPooling2D
import numpy as np
import random
from keras.optimizers import Adam

def vgg16_model(shape_1, num_classes):
    model = Sequential()
    model.add(Conv2D(input_shape=(shape_1,num_classes,1),filters=64,kernel_size=(1,1),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(1,1),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(1, 1), strides=(1,1)))
    model.add(Conv2D(filters=128, kernel_size=(1,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(1,1), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(1, 1), strides=(1,1)))
    model.add(Conv2D(filters=256, kernel_size=(1,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(1,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(1,1), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(1, 1), strides=(1,1)))
    model.add(Conv2D(filters=512, kernel_size=(1,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(1,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(1,1), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(1, 1), strides=(1,1)))
    model.add(Conv2D(filters=512, kernel_size=(1,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(1,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(1,1), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(1, 1), strides=(1,1)))
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=num_classes, activation="softmax"))
    
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    return model