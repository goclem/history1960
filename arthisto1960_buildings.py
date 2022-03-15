#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Workflow for the semantic segmentation example
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.03.15
'''

#%% HEADER

# Modules
import itertools
import numpy as np
import tensorflow

from arthisto1960_utilities import *
from numpy import random
from os import path
from tensorflow.keras import callbacks, layers, models, preprocessing

# TensorFlow
print('TensorFlow version:', tensorflow.__version__)
print('GPU Available:', len(tensorflow.config.experimental.list_physical_devices('GPU')))

# Paths
paths = dict(
    images_raw=  '../data_1960/raw/images',
    labels_raw=  '../data_1960/raw/labels',
    images_train='../data_1960/training/images',
    labels_train='../data_1960/training/labels',
    images_valid='../data_1960/validation/images',
    labels_valid='../data_1960/validation/labels',
    images_test= '../data_1960/testing/images',
    labels_test= '../data_1960/testing/labels',
    models=      '../data_1960/models',
    figures=     '../data_1960/figures',
    temporary=   '/Users/clementgorin/Temporary'
)

#%% FUNCTIONS

def standardise_image(image:np.ndarray) -> np.ndarray:
    bandmeans   = np.mean(image, axis=(0, 1), keepdims=True)
    bandstds    = np.std(image,  axis=(0, 1), keepdims=True)
    standarised = (image - bandmeans) / bandstds
    return standarised

def images_to_blocks(images:np.ndarray, imagesize:tuple, blocksize:tuple=(256, 256), shift:bool=False, mode:str='symmetric') -> np.ndarray:
    # Defines quantities
    nimages, imagewidth, imageheight, nbands = imagesize
    blockwidth, blockheight = blocksize
    nblockswidth  = (imagewidth  // blockwidth  + 1 + shift)
    nblocksheight = (imageheight // blockheight + 1 + shift)
    # Defines padding
    padwidth  = int(((nblockswidth)  * blockwidth  - imagewidth)  / 2)
    padheight = int(((nblocksheight) * blockheight - imageheight) / 2)
    # Maps images to blocks
    images = np.pad(images, ((0, 0), (padwidth, padwidth), (padheight, padheight), (0, 0)), mode=mode)
    blocks = images.reshape(nimages, nblockswidth, blockwidth, nblocksheight, blockheight, nbands, ).swapaxes(2, 3)
    blocks = blocks.reshape(-1, blockwidth, blockheight, nbands)
    return blocks

def blocks_to_images(blocks:np.ndarray, imagesize:tuple, blocksize:tuple=(256, 256), shift:bool=False) ->  np.ndarray:
    # Defines quantities
    nimages, imagewidth, imageheight, nbands = imagesize
    blockwidth, blockheight = blocksize
    nblockswidth  = (imagewidth  // blockwidth  + 1 + shift)
    nblocksheight = (imageheight // blockheight + 1 + shift)
    # Defines padding
    padwidth  = int(((nblockswidth)  * blockwidth  - imagewidth)  / 2)
    padheight = int(((nblocksheight) * blockheight - imageheight) / 2)
    # Maps blocks to images
    images = blocks.reshape(-1, nblockswidth, nblocksheight, blockwidth, blockheight, nbands).swapaxes(2, 3)
    images = images.reshape(-1, (imagewidth + (2 * padwidth)), (imageheight + (2 * padheight)), nbands)
    images = images[:, padwidth:imagewidth + padwidth, padheight:imageheight + padheight, :]
    return images

def sample_split(array:np.ndarray, sizes:dict, seed:int=1):
    random.seed(seed)
    samples = list(sizes.keys())
    indexes = random.choice(samples, array.shape[0], p=list(sizes.values()))
    samples = [array[indexes == sample, ...] for sample in samples]
    return samples

def display_statistics(image_test:np.ndarray, label_test:np.ndarray, proba_predict:np.ndarray, label_predict:np.ndarray) -> None:
    # Format
    image_test    = (image_test * 255).astype(int)
    label_test    = label_test.astype(bool)
    label_predict = label_predict.astype(bool)
    # Statistics
    mask_tp = np.logical_and(label_test, label_predict)
    mask_tn = np.logical_and(np.invert(label_test), np.invert(label_predict))
    mask_fp = np.logical_and(np.invert(label_test), label_predict)
    mask_fn = np.logical_and(label_test, np.invert(label_predict))
    # Augmented images
    colour  = (255, 255, 0)
    images  = [np.where(np.tile(mask, (1, 1, 3)), colour, image_test) for mask in [mask_tp, mask_tn, mask_fp, mask_fn]]
    # Figure
    images = [image_test, label_test, proba_predict, label_predict] + images
    titles = ['Test image', 'Test label', 'Predicted probability', 'Predicted label', 'True positive', 'True negative', 'False positive', 'False negative']
    fig, axs = pyplot.subplots(2, 4, figsize=(20, 10))
    for image, title, ax in zip(images, titles, axs.ravel()):
        ax.imshow(image)
        ax.set_title(title, fontsize=20)
        ax.axis('off')
    pyplot.tight_layout(pad=2.0)
    pyplot.show()

#%% PREPARES DATA

"""
Notes:
- Vannes  (0250_6745) is incomplete
- Orleans (0600_6770) is incomplete
"""

# Loads images as blocks
images = search_files(paths['images_raw'], 'tif$')
images = np.array([read_raster(file) for file in images])
images = np.concatenate((
    images_to_blocks(images=images, imagesize=images.shape, blocksize=(256, 256), shift=True),
    images_to_blocks(images=images, imagesize=images.shape, blocksize=(256, 256), shift=False)
))

# Loads labels as blocks
labels = search_files(paths['labels_raw'], 'tiff?$')
labels = np.array([read_raster(file) for file in labels])
labels = np.concatenate((
    images_to_blocks(images=labels, imagesize=labels.shape, blocksize=(256, 256), shift=True),
    images_to_blocks(images=labels, imagesize=labels.shape, blocksize=(256, 256), shift=False)
))

# Drops empty blocks
def is_empty(image:np.ndarray, value:int=255) -> bool:
    test = np.equal(image, np.full(image.shape, value)).all()
    return test

keep   = np.invert([is_empty(image) for image in list(images)])
images = images[keep]
labels = labels[keep]
del is_empty, keep

# Checks data
# for i in random.choice(range(len(images)), 3):
#     compare(images=[images[i], labels[i]], titles=['Image', 'Label'])

#%%  COMPUTES SAMPLES

samples_size = dict(train=0.8, valid=0.1, test=0.1)
images_train, images_valid, images_test = sample_split(array=images, sizes=samples_size, seed=1)
labels_train, labels_valid, labels_test = sample_split(array=labels, sizes=samples_size, seed=1)
samples_size = dict(train=len(images_train), valid=len(images_valid), test=len(images_test))
del images, labels

#%% MODEL

def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = layers.Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    y = layers.Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = layers.concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y

def init_unet(n_classes:int, input_size:tuple, nfilters:int):
    # Input
    inputs  = layers.Input(shape=input_size, name='image_input')
    # Contraction
    conv1   = conv_block(inputs, nfilters=nfilters)
    pool1   = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2   = conv_block(pool1, nfilters=nfilters*2)
    pool2   = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3   = conv_block(pool2, nfilters=nfilters*4)
    pool3   = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4   = conv_block(pool3, nfilters=nfilters*8)
    pool4   = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4   = layers.Dropout(0.5)(pool4)
    conv5   = conv_block(pool4, nfilters=nfilters*16)
    conv5   = layers.Dropout(0.5)(conv5)
    # Extension
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=nfilters*8)
    deconv6 = layers.Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=nfilters*4)
    deconv7 = layers.Dropout(0.5)(deconv7)
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=nfilters*2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=nfilters)
    # Output
    outputs = layers.Conv2D(n_classes, kernel_size=(1, 1), activation='sigmoid')(deconv9)
    # Model
    model   = models.Model(inputs=inputs, outputs=outputs, name='Unet')
    return model

unet = init_unet(n_classes=1, input_size=(256, 256, 3), nfilters=16)
unet.summary()
unet.compile(optimizer='adam', loss='binary_focal_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])
del conv_block, deconv_block

#%% PRE-PROCESSING

'''
Notes:
- The data generator must be fit to the training data to estimate the featurewise_center and featurewise_std_normalization
- The estimated parameters can be extracted using ImageDataGenerator.mean and ImageDataGenerator.std
- The batch size is only defined in the generator, the steps_per_epoch must be defined during training
- Fit a validation data generator only with the standardisation parameters
'''

# Build image generator
def generator(images:np.ndarray, labels:np.ndarray, args:dict, batch_size:int=32, shuffle:bool=True, seed:int=1) -> zip:
    # Images generator
    images_datagen   = preprocessing.image.ImageDataGenerator(**args)
    images_generator = images_datagen.flow(images, batch_size=batch_size, shuffle=shuffle, seed=seed)
    # Labels generator
    labels_datagen   = preprocessing.image.ImageDataGenerator(**args)
    labels_generator = labels_datagen.flow(labels, batch_size=batch_size, shuffle=shuffle, seed=seed)
    # Combines generators
    generator = zip(images_generator, labels_generator)
    return generator

# Standardisation parameters
standardisation = dict(
    rescale=1./255
)

# Augmentation parameters
augmentation = dict(
    horizontal_flip=True, 
    vertical_flip=True,
    rotation_range=45, 
    width_shift_range=0.25,
    height_shift_range=0.25,
    brightness_range=[0.75,1.25],
    zoom_range=[0.75, 1.25],
    fill_mode='reflect'
)

train_generator  = generator(images_train, labels_train, dict(**standardisation, **augmentation))
valid_generator  = generator(images_valid, labels_valid, standardisation)
test_generator   = generator(images_test,  labels_test,  standardisation, shuffle=False)
del generator, standardisation, augmentation, images_train, labels_train, images_valid, labels_valid, images_test, labels_test

# Check
# images, labels = next(train_generator)
# for i in random.choice(range(len(images)), 5):
#     compare(images=[images[i], labels[i]], titles=['Image', 'Label'])
# del(images, labels)

#%% ESTIMATES PARAMETERS
'''
Notes:
- Fitting the generator requires steps_per_epoch for the training samples and validation_steps for the validation sample
'''

# Callbacks
train_callbacks = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    callbacks.ModelCheckpoint(filepath='model/unet_{epoch:02d}_{val_accuracy:.4f}.h5', monitor='val_accuracy', save_best_only=True),
    callbacks.BackupAndRestore(backup_dir='model')
]

training = unet.fit(
    train_generator,
    steps_per_epoch=samples_size['train'] // 32,
    #validation_data=valid_generator,
    #validation_steps=samples_size['valid'] // 32,
    epochs=2,
    verbose=1,
    callbacks=train_callbacks)

training.history

#%% EVALUATES MODEL

# Compute statistics
performance = unet.evaluate(test_generator, steps=samples_size['test'] // 32) # 81%
print('Test loss: {:.4f}\nTest accuracy: {:.4f}'.format(*performance))

# Displays statistics
images_test, labels_test = next(test_generator)
probas_predict = unet.predict(images_test, verbose=1)
labels_predict = probas_predict >= 0.5
for i in random.choice(range(len(images)), 5):
    display_statistics(image_test=images_test[i], label_test=labels_test[i], proba_predict=probas_predict[i], label_predict=labels_predict[i])
del images_test, labels_test, probas_predict, labels_predict