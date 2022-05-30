#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Performs predictions for the Arthisto 1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.03.30
'''

#%% HEADER

# Modules
import itertools
import numpy as np
import tensorflow

from arthisto1960_model import binary_unet
from arthisto1960_utilities import *
from keras import callbacks, layers, models, preprocessing
from numpy import random
from os import path

# TensorFlow
print('TensorFlow version:', tensorflow.__version__)
print('GPU Available:', len(tensorflow.config.experimental.list_physical_devices('GPU')))

#%% FORMATS DATA

# Training tiles
training = identifiers(search_data(paths['labels']), regex=True)

# Loads images as blocks (including shifted)
images = search_data(paths['images'], training)
images = np.array([read_raster(file, dtype=int) for file in images])
images = np.concatenate((
    images_to_blocks(images, blocksize=(256, 256), shift=True,  mode='constant', constant_values=255),
    images_to_blocks(images, blocksize=(256, 256), shift=False, mode='constant', constant_values=255)
))

# Loads labels as blocks (including shifted)
labels = search_data(paths['labels'], pattern=training)
labels = np.array([read_raster(file, dtype=int) for file in labels])
labels = np.concatenate((
    images_to_blocks(labels, blocksize=(256, 256), shift=True,  mode='constant', constant_values=0),
    images_to_blocks(labels, blocksize=(256, 256), shift=False, mode='constant', constant_values=0)
))

# Drops empty blocks
keep   = list(map(not_empty, images))
images = images[keep]
labels = labels[keep]
del keep

# Checks data
# for i in random.choice(range(len(images)), 5):
#     compare([images[i], labels[i]], ['Image', 'Label'])

#%%  SPLITS SAMPLES

samples_size = dict(train=0.70, valid=0.15, test=0.15)
images_train, images_valid, images_test = sample_split(images=images, sizes=samples_size, seed=1)
labels_train, labels_valid, labels_test = sample_split(images=labels, sizes=samples_size, seed=1)
samples_size = dict(train=len(images_train), valid=len(images_valid), test=len(images_test))
del images, labels

#%% AUGMENTS TRAINING DATA

# Augmentation parameters
image_augmentation = dict(
    rescale=1./255,
    horizontal_flip=True, 
    vertical_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.9,1.1],
    zoom_range=[0.9, 1.1],
    fill_mode='constant',
    cval=255
)
label_augmentation = image_augmentation.copy()
label_augmentation.update(cval=0)

# Initialises training data generator
data_generator   = preprocessing.image.ImageDataGenerator(**image_augmentation)
images_generator = data_generator.flow(images_train, batch_size=32, shuffle=True, seed=1)
data_generator   = preprocessing.image.ImageDataGenerator(**label_augmentation)
labels_generator = data_generator.flow(labels_train, batch_size=32, shuffle=True, seed=1)
train_generator  = zip(images_generator, labels_generator)
del image_augmentation, label_augmentation, data_generator, images_generator, labels_generator, images_train, labels_train

# Rescales validation data
images_valid = layers.Rescaling(1./255)(images_valid)

# Checks data
# images, labels = next(train_generator)
# for i in random.choice(range(len(images)), 5):
#     compare(images=[images[i], labels[i]], titles=['Image', 'Label'])
# del images, labels

#%% ESTIMATES PARAMETERS

# Initialises model
model = binary_unet(input_shape=(256, 256, 3), filters=64)
model.compile(optimizer='adam', loss='binary_focal_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])
model.summary()

# Callbacks
train_callbacks = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True),
    callbacks.ModelCheckpoint(filepath=path.join(paths['models'], 'unet64_{epoch:03d}.h5'), monitor='val_accuracy', save_best_only=True),
    callbacks.BackupAndRestore(backup_dir=paths['models'])
]

# Training
training = model.fit(
    train_generator,
    steps_per_epoch=samples_size['train'] // 32,
    validation_data=(images_valid, labels_valid),
    epochs=1000,
    verbose=1,
    callbacks=train_callbacks
)
del train_callbacks

# Saves model and training history
# models.save_model(model, path.join(paths['models'], 'unet64_baseline.h5'))
# np.save(path.join(paths['models'], 'unet64_history.npy'), training.history)

# Displays history
# history = np.load(path.join(paths['models'], 'unet64_history.npy'), allow_pickle=True).item()
# display_history(history)
# del history

#%% EVALUATES MODEL

# Loads model
model = models.load_model(path.join(paths['models'], 'unet64_old.h5'))

# Compute statistics
performance = layers.Rescaling(1./255)(images_test)
performance = model.evaluate(performance, labels_test)
print('Test loss: {:.4f}\nTest accuracy: {:.4f}\nTest recall: {:.4f}\nTest precision: {:.4f}'.format(*performance))
del performance

# Saves test data for statistics
np.save(path.join(paths['statistics'], 'images_test.npy'), images_test)
np.save(path.join(paths['statistics'], 'labels_test.npy'), labels_test)