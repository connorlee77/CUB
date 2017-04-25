import os
import h5py
import numpy as np
import pandas as pd 
np.random.seed(148)

import sklearn.metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.applications.vgg19 import VGG19
from keras import backend as K
from keras.optimizers import SGD, RMSprop

from scipy import ndimage


def plot_confusion_matrix(cm):
    
    plt.imshow(cm, interpolation='nearest', cmap='hot')
    plt.colorbar()

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusionmatrix1')



# path to the model weights file.
img_width, img_height = 224, 224 
train_data_dir = 'train'
validation_data_dir = 'validation'


### Parameters
batch_size = 32
num_classes = 200
total_test = 5794
steps = int(128/batch_size)

DIR = './'
test_CSV = pd.read_csv(DIR + 'test.csv')
true_labels = test_CSV['label'].values[0:steps * batch_size]

testgen = ImageDataGenerator(rescale=1./255)

generator_test = testgen.flow_from_directory(
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
model.load_weights('total_weights.h5')

predictions = model.predict_generator(generator_test, steps, verbose=1)
pred_labels = np.array(map(np.argmax, predictions)) + 1

acc = sklearn.metrics.accuracy_score(true_labels, pred_labels)
cm = sklearn.metrics.confusion_matrix(true_labels, pred_labels, labels=[i for i in range(1, 201)])
plot_confusion_matrix(cm)
