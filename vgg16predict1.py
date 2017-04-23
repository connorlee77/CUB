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

from keras.applications.vgg19 import VGG19
from keras import backend as K
from keras.optimizers import SGD, RMSprop

from scipy import ndimage

def getTestData():

	DIR = './'
	test_CSV = pd.read_csv(DIR + 'test.csv')
	
	X = []
	Y = []
	for i, row in test_CSV.iterrows():
		path = row['path']
		label = row['label']
		
		Y.append(label)
		img = ndimage.imread(DIR + 'validation/' +path) / 255.0
		X.append(img)
		print i
	return X, np.array(Y)

x, y = getTestData()

# path to the model weights file.
img_width, img_height = 224, 224 
train_data_dir = 'train'
validation_data_dir = 'validation'


### Parameters
batch_size = 32
num_classes = 200

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

p = model.predict_classes(x, batch_size=batch_size, verbose=1)
print p

