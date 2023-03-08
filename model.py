#buiding the CNN
#importing keras libraries and packages
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json

batch_size = 32

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#rescaling of images
train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(200,200),
    batch_size = batch_size,
    classes =["covid","normal"],
    class_mode='categorical')

import tensorflow as tf

model = tf.keras.models.Sequential([
    #first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200,200,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    #second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #flatten the resulys to feed into dense layer
    tf.keras.layers.Flatten(),
    #128 neuron
    tf.keras.layers.Dense(128, activation='relu'),
    #5 output neuron
    tf.keras.layers.Dense(2, activation='softmax')
])

model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

total_sample = train_generator.n

n_epochs = 30

history = model.fit_generator(
    train_generator,
    steps_per_epoch=int(total_sample/batch_size),
    epochs=n_epochs,
    verbose=1)

model.save('covid_model.h5')
