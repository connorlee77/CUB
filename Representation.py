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
img_width, img_height = 150, 150 
train_data_dir = 'train'
validation_data_dir = 'validation'


### Parameters
batch_size = 32
epochs = 1
num_classes = 20
tensorflow = False
train_size = 600
test_size = 515




# Data
datagen = ImageDataGenerator(rescale=1./255)
generator_train = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

generator_test = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)


base_model = VGG19(weights='imagenet', input_shape=(img_width, img_height, 3), pooling='max', include_top=False)

# Top classifier
x = base_model.output
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Combined model w/ classifier
model = Model(input=base_model.input, output=predictions)

model.summary()
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
bottleneck_features_train = model.predict_generator(generator_train, 5994)
np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

bottleneck_features_test = model.predict_generator(generator_test, 5794)
np.save(open('bottleneck_features_test.npy', 'w'), bottleneck_features_test)
