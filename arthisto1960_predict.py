#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Performs predictions for the Arthisto 1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.03.23
'''

#%% HEADER

# Modules
import itertools
import numpy as np
import tensorflow

from arthisto1960_model import binary_unet
from arthisto1960_utilities import *
from numpy import random
from pandas import DataFrame
from os import path
from tensorflow.keras import callbacks, layers, models, preprocessing, utils

# TensorFlow
print('TensorFlow version:', tensorflow.__version__)
print('GPU Available:', len(tensorflow.config.experimental.list_physical_devices('GPU')))

# Paths
paths = dict(
    data='../data_1960',
    images='../data_1960/images',
    labels='../data_1960/labels',
    models='../data_1960/models',
    predictions='../data_1960/predictions',
    statistics='../data_1960/statistics'
)

# Clears QGIS auxiliary files
# [os.remove(file) for file in search_files(directory=paths['data'], pattern='.tif.aux.xml')]

#%% FUNCTIONS

# Converts images to blocks of a given size
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

# Converts blocks to images of a given size
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

# Splits the data multiple samples
def sample_split(images:np.ndarray, sizes:dict, seed:int=1) -> list:
    random.seed(seed)
    samples = list(sizes.keys())
    indexes = random.choice(samples, images.shape[0], p=list(sizes.values()))
    samples = [images[indexes == sample, ...] for sample in samples]
    return samples

# Displays training history
def display_history(history:dict, stats:list=['accuracy', 'loss']) -> None:
    fig, axs = pyplot.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for ax, stat in zip(axs.ravel(), stats):
        ax.plot(training.history[stat])
        ax.plot(training.history[f'val_{stat}'])
        ax.set_title(f'Training {stat}', fontsize=15)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(['Training sample', 'Validation sample'], frameon=False)
    pyplot.tight_layout(pad=2.0)
    pyplot.show()

#%% PREPARES DATA

# Training tiles
pattern = identifiers(search_files(paths['labels'], 'tif$'), regex=True)

# Loads images as blocks (including shifted)
images = search_files(directory=paths['images'], pattern=pattern)
images = np.array([read_raster(file) for file in images])
images = np.concatenate((
    images_to_blocks(images=images, imagesize=images.shape, blocksize=(256, 256), shift=True),
    images_to_blocks(images=images, imagesize=images.shape, blocksize=(256, 256), shift=False)
))

# Loads labels as blocks (including shifted)
labels = search_files(directory=paths['labels'], pattern=pattern)
labels = np.array([read_raster(file) for file in labels])
labels = np.concatenate((
    images_to_blocks(images=labels, imagesize=labels.shape, blocksize=(256, 256), shift=True),
    images_to_blocks(images=labels, imagesize=labels.shape, blocksize=(256, 256), shift=False)
))

# Drops empty blocks
def is_empty(image:np.ndarray, value:int=255) -> bool:
    empty = np.equal(image, np.full(image.shape, value)).all()
    return empty

keep   = np.invert([is_empty(image) for image in list(images)])
images = images[keep]
labels = labels[keep]
del is_empty, keep

# Checks data
# for i in random.choice(range(len(images)), 5):
#     compare(images=[images[i], labels[i]], titles=['Image', 'Label'])

#%%  COMPUTES SAMPLES
samples_size = dict(train=0.8, valid=0.1, test=0.1)
images_train, images_valid, images_test = sample_split(images=images, sizes=samples_size, seed=1)
labels_train, labels_valid, labels_test = sample_split(images=labels, sizes=samples_size, seed=1)
samples_size = dict(train=len(images_train), valid=len(images_valid), test=len(images_test))
del images, labels

#%% AUGMENTATION

# Rescales data
images_valid = layers.Rescaling(1./255)(images_valid)
images_test  = layers.Rescaling(1./255)(images_test)

# Augmentation parameters
augmentation = dict(
    rescale=1./255,
    horizontal_flip=True, 
    vertical_flip=True,
    rotation_range=45, 
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.9,1.1],
    zoom_range=[0.9, 1.1],
    fill_mode='reflect'
)

# Initialises training data generator
data_generator   = preprocessing.image.ImageDataGenerator(**augmentation)
images_generator = data_generator.flow(images_train, batch_size=32, shuffle=True, seed=1)
labels_generator = data_generator.flow(labels_train, batch_size=32, shuffle=True, seed=1)
train_generator  = zip(images_generator, labels_generator)
del augmentation, data_generator, images_generator, labels_generator, images_train, labels_train

# Check
# images, labels = next(train_generator)
# for i in random.choice(range(len(images)), 5):
#     compare(images=[images[i], labels[i]], titles=['Image', 'Label'])
# del images, labels

#%% ESTIMATES PARAMETERS

# Initialises model
model = binary_unet(input_shape=(256, 256, 3), filters=32)
model.compile(optimizer='adam', loss='binary_focal_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])
# model = models.load_model('../data_1960/models/unet32_epoch012.h5')

# Summary
# model.summary()
# summary = DataFrame([dict(Name=layer.name, Type=layer.__class__.__name__, Shape=layer.output_shape, Params=layer.count_params()) for layer in model.layers])
# summary.style.to_html(path.join(paths['models'], 'unet32_structure.html'), index=False) 
# del summary
# utils.plot_model(model, to_file=path.join(paths['models'], 'unet32_structure.pdf'), show_shapes=True)

# Callbacks
train_callbacks = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    callbacks.ModelCheckpoint(filepath='../data_1960/models/unet32_{epoch:03d}.h5', monitor='val_accuracy', save_best_only=True),
    callbacks.BackupAndRestore(backup_dir='../data_1960/models')
]

# Training
training = model.fit(
    train_generator,
    steps_per_epoch=samples_size['train'] // 32,
    validation_data=(images_valid, labels_valid),
    epochs=100,
    verbose=1,
    callbacks=train_callbacks
)

# Saves model and training history
# models.save_model(model, '../data_1960/models/unet32_baseline.h5')
# np.save('../data_1960/models/unet32_history.npy', training.history)

# Displays history
# history = np.load('../data_1960/models/history_baseline.npy',allow_pickle=True).item()
# display_history(history)
# del history

#%% EVALUATES MODEL

# Loads model if estimated previously
# unet = models.load_model('../data_1960/models/unet32_baseline.h5')

# Compute statistics
# performance = unet.evaluate(images_test, labels_test)
# print('Test loss: {:.4f}\nTest accuracy: {:.4f}\nTest recall: {:.4f}\nTest precision: {:.4f}'.format(*performance))
# del performance

# Displays statistics
probas_pred = model.predict(images_test, verbose=1)
labels_pred = probas_pred >= 0.5

# Saves test data for statistics
for data in ['images_test', 'labels_test', 'probas_pred', 'labels_pred']:
    np.save(path.join(paths['statistics'], data + '.npy'), globals()[data])

# Displays prediction statistics
for i in random.choice(range(len(images_test)), 5):
    display_statistics(image_test=images_test[i], label_test=labels_test[i], proba_pred=probas_pred[i], label_pred=labels_pred[i])
del images_test, labels_test, probas_pred, labels_pred

#%% PREDICTS NEW TILES

# Loads model
# tensorflow.compat.v1.logging.get_verbosity()
# tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
model = models.load_model('../data_1960/models/unet32_baseline.h5')

# Lists batches
batch_size = 3
batches = search_files(paths['images'], pattern='tif$')
batches = filter_identifiers(batches, search_files(paths['predictions'], pattern='tif$'))
batches = [batches[i:i + batch_size] for i in range(0, len(batches), batch_size)]
del batch_size

# Computes predictions
def predict_tiles(model, files):
    images  = np.array([read_raster(file) for file in files])
    images  = layers.Rescaling(1./255)(images)
    blocks1 = images_to_blocks(images=images, imagesize=images.shape, blocksize=(256, 256), shift=False)
    blocks2 = images_to_blocks(images=images, imagesize=images.shape, blocksize=(256, 256), shift=True)
    probas1 = model.predict(blocks1, verbose=1)
    probas2 = model.predict(blocks2, verbose=1)
    probas1 = blocks_to_images(blocks=probas1, imagesize=images.shape[:3] + (1,), blocksize=(256, 256), shift=False)
    probas2 = blocks_to_images(blocks=probas2, imagesize=images.shape[:3] + (1,), blocksize=(256, 256), shift=True)
    probas  = (probas1 + probas2) / 2
    return probas

# Computes predictions
for i, files in enumerate(batches):
    print('Batch {i:d}/{n:d}'.format(i=i + 1, n=len(batches)))
    probas   = predict_tiles(model=model, files=files)
    outfiles = [path.join(paths['predictions'], path.basename(file).replace('image', 'proba')) for file in files]
    for proba, file, outfile in zip(probas, files, outfiles):
        write_raster(array=proba, source=file, destination=outfile, nodata=None, dtype='float32')
del i, files, file, probas, proba, outfiles, outfile
# %%
