import os
import h5py
import numpy as np
np.random.seed(148)

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.applications.vgg19 import VGG19
from keras import backend as K
from keras.optimizers import SGD, RMSprop
import matplotlib.pyplot as plt


# path to the model weights file.
img_width, img_height = 224, 224 
train_data_dir = 'train'
validation_data_dir = 'validation'


### Parameters
batch_size = 32
epochs1 = 50
epochs2 = 15
num_classes = 200
tensorflow = True
train_size = 5096
test_size = 512


def fitData(tensorflow, batch_size, epochs, model, generator_train, generator_test, train_size, test_size):
    history = None
    if tensorflow:
        tbCB = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False, write_images=False)

        history = model.fit_generator(generator_train,
            steps_per_epoch=train_size / batch_size,
            epochs=epochs,
            validation_data=generator_test,
            validation_steps=test_size / batch_size,
            callbacks=[tbCB])
    else:
        history = model.fit_generator(generator_train,
            steps_per_epoch=train_size / batch_size,
            epochs=epochs,
            validation_data=generator_test,
            validation_steps=test_size / batch_size)

    return history


# Data
datagen = ImageDataGenerator(rescale=1./255, 
    horizontal_flip=True, 
    rotation_range=10,
    shear_range=0.2,
    zoom_range=0.2)

testgen = ImageDataGenerator(rescale=1./255)

generator_train = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

generator_test = testgen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)


base_model = VGG19(weights='imagenet', input_shape=(img_width, img_height, 3), pooling='max', include_top=False)

# Top classifier
x = base_model.output
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Combined model w/ classifier
model = Model(input=base_model.input, output=predictions)
model.load_weights('total_weights.h5')

# Train top classifer only
for i, layer in enumerate(base_model.layers):
    layer.trainable = False
    print(i, layer.name)

model.summary()
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
history1 = fitData(tensorflow, batch_size, epochs1, model, generator_train, generator_test, train_size, test_size)
model.save_weights('top_weights.h5')


# Train last convolution block too. 
for layer in model.layers[:17]:
    print(layer.name)
    layer.trainable = False
for layer in model.layers[17:]:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
history2 = fitData(tensorflow, batch_size, epochs2, model, generator_train, generator_test, train_size, test_size)

model.save_weights('total_weights.h5')
