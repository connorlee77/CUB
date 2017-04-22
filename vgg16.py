import os
import h5py
import numpy as np
np.random.seed(148)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras.optimizers import SGD
import matplotlib.pyplot as plt

# path to the model weights file.
img_width, img_height = 150, 150
train_data_dir = 'train'
validation_data_dir = 'validation'
batch_size = 32
epochs = 500

num_classes = 20

datagen = ImageDataGenerator(rescale=1./255)
base_model = VGG19(weights='imagenet', input_shape=(img_width, img_height, 3), pooling='max', include_top=False)

x = base_model.output
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

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
        shuffle=False)

for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

for layer in model.layers[:17]:
    print(layer.name)
    layer.trainable = False
for layer in model.layers[17:]:
   layer.trainable = True

model.summary()
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(generator_train,
    steps_per_epoch=600 / batch_size,
    epochs=epochs,
    validation_data=generator_test,
    validation_steps=515 / batch_size)

model.save_weights('weights.h5')

fig = plt.figure()
plt.plot(history.history['acc'], label='Train')
plt.plot(history.history['val_acc'], label='Test')
plt.title('Train')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='best', prop={'size':'small'})
fig.savefig('acc.png')

fig = plt.figure()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.title('Train')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='best', prop={'size':'small'})
fig.savefig('loss.png')
