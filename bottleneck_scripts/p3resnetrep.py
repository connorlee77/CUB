import os
import h5py
import numpy as np
import pandas as pd 
np.random.seed(148)

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras import backend as K
from keras.optimizers import SGD, RMSprop


def preprocess_input_res(x):
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]

# path to the model weights file.
img_width, img_height = 224, 224 
train_data_dir = 'warp_train'
validation_data_dir = 'warp_validation'

### Parameters
batch_size = 32
num_classes = 200
total_train = 5994
total_test = 5794
train_steps = int(total_train/batch_size)
test_steps = int(total_test/batch_size)

DIR = './'
train_CSV = pd.read_csv(DIR + 'train.csv')
train_labels = train_CSV['label'].values[0:train_steps * batch_size]
np.save(open('p3_train_labels.npy', 'w'), train_labels)

test_CSV = pd.read_csv(DIR + 'test.csv')
test_labels = test_CSV['label'].values[0:test_steps * batch_size]
np.save(open('p3_test_labels.npy', 'w'), test_labels)


datagen = ImageDataGenerator(preprocessing_function=preprocess_input_res)
testgen = ImageDataGenerator(preprocessing_function=preprocess_input_res)

generator_train = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

generator_test = testgen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)


base_model = ResNet50(weights='imagenet', input_shape=(img_width, img_height, 3), include_top=False)

# Top classifier
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Combined model w/ classifier
model = Model(input=base_model.input, output=predictions)
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('p3resnet_total.h5')
model.summary()

intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-4].output)

train_predictions = intermediate_layer_model.predict_generator(generator_train, train_steps, verbose=1)
np.save(open('p3_bottleneck_features_train.npy', 'w'), train_predictions)

test_predictions = intermediate_layer_model.predict_generator(generator_test, test_steps, verbose=1)
np.save(open('p3_bottleneck_features_test.npy', 'w'), test_predictions)


