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
datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=10)
generator_train = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

generator_test = datagen.flow_from_directory(
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


# Train top classifer only
for i, layer in enumerate(base_model.layers):
    layer.trainable = False
    print(i, layer.name)

model.summary()
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
history1 = fitData(tensorflow, batch_size, epochs, model, generator_train, generator_test, train_size, test_size)
model.save_weights('top_weights.h5')


# Train last convolution block too. 
for layer in model.layers[:17]:
    print(layer.name)
    layer.trainable = False
for layer in model.layers[17:]:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.95), loss='categorical_crossentropy', metrics=['accuracy'])
history2 = fitData(tensorflow, batch_size, epochs, model, generator_train, generator_test, train_size, test_size)

model.save_weights('total_weights.h5')



# Plotting 
acc = history1.history['acc'] + history2.history['acc']
val_acc = history1.history['val_acc'] + history2.history['val_acc']

loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

fig = plt.figure()
plt.plot(acc, label='Train')
plt.plot(val_acc, label='Test')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right', prop={'size':'small'})
fig.savefig('acc.png')

fig = plt.figure()
plt.plot(loss, label='Train')
plt.plot(val_loss, label='Test')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right', prop={'size':'small'})
fig.savefig('loss.png')
