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

# Displays prediction statistics
def display_statistics(image_test:np.ndarray, label_test:np.ndarray, proba_pred:np.ndarray, label_pred:np.ndarray) -> None:
    # Format
    image_test = (np.array(image_test) * 255).astype(int) # Tensor type
    label_test = label_test.astype(bool)
    label_pred = label_pred.astype(bool)
    # Statistics
    mask_tp = np.logical_and(label_test, label_pred)
    mask_tn = np.logical_and(np.invert(label_test), np.invert(label_pred))
    mask_fp = np.logical_and(np.invert(label_test), label_pred)
    mask_fn = np.logical_and(label_test, np.invert(label_pred))
    # Augmented images
    colour  = (255, 255, 0)
    images  = [np.where(np.tile(mask, (1, 1, 3)), colour, image_test) for mask in [mask_tp, mask_tn, mask_fp, mask_fn]]
    # Figure
    images = [image_test, label_test, proba_pred, label_pred] + images
    titles = ['Test image', 'Test label', 'Predicted probability', 'Predicted label', 'True positive', 'True negative', 'False positive', 'False negative']
    fig, axs = pyplot.subplots(2, 4, figsize=(20, 10))
    for image, title, ax in zip(images, titles, axs.ravel()):
        ax.imshow(image)
        ax.set_title(title, fontsize=20)
        ax.axis('off')
    pyplot.tight_layout(pad=2.0)
    pyplot.show()

#%% PREPARES DATA

# Training tiles
pattern = identifiers(search_files(paths['labels'], 'tif$'))
pattern = '({}).tif$'.format('|'.join(pattern))

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
# Declare seed
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
    width_shift_range=0.25,
    height_shift_range=0.25,
    brightness_range=[0.75,1.25],
    zoom_range=[0.75, 1.25],
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

#%% INITIALISES MODEL

'''
Notes:
- Check the number of filters for transpose, we should maintian dimensionality
- Compare the model summary with the original U-net to make sure everything is ok
- May be better https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
'''

def convolutional_block(input, filters:int, dropout:float=0, kernel_size:dict=(3, 3), padding:str='same', initializer:str='he_normal', name:str=''):
    convolution   = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer, use_bias=False, name=f'{name}_convolution1')(input)
    activation    = layers.Activation(activation='relu', name=f'{name}_activation1')(convolution)
    normalisation = layers.BatchNormalization(name=f'{name}_normalisation1')(activation)
    convolution   = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer, use_bias=False, name=f'{name}_convolution2')(normalisation)
    activation    = layers.Activation(activation='relu', name=f'{name}_activation2')(convolution)
    normalisation = layers.BatchNormalization(name=f'{name}_normalisation2')(activation)
    dropout       = layers.SpatialDropout2D(rate=dropout, name=f'{name}_dropout')(normalisation)
    return dropout

def deconvolutional_block(input, skip, filters:int, kernel_size:dict=(3, 3), padding:str='same', strides:dict=(2, 2), dropout:float=0, name:str=''):
    transpose     = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=f'{name}_transpose')(input)
    concatenation = layers.concatenate(inputs=[transpose, skip], axis=3, name=f'{name}_concatenation')
    convblock     = convolutional_block(input=concatenation, filters=filters, dropout=dropout, name=name)
    return convblock

def init_unet(n_classes:int, input_size:tuple, filters:int):
    # Input
    inputs = layers.Input(shape=input_size, name='input')
    # Contraction
    convblock1 = convolutional_block(input=inputs, filters=filters*1, dropout=0.1, name='convblock1')
    maxpool1   = layers.MaxPool2D(pool_size=(2, 2), name='convblock1_maxpool')(convblock1)
    convblock2 = convolutional_block(input=maxpool1, filters=filters*2, dropout=0.1, name='convblock2')
    maxpool2   = layers.MaxPool2D(pool_size=(2, 2), name='convblock2_maxpool')(convblock2)
    convblock3 = convolutional_block(input=maxpool2, filters=filters*4, dropout=0.2, name='convblock3')
    maxpool3   = layers.MaxPool2D(pool_size=(2, 2), name='convblock3_maxpool')(convblock3)
    convblock4 = convolutional_block(input=maxpool3, filters=filters*8, dropout=0.2, name='convblock4')
    maxpool4   = layers.MaxPool2D(pool_size=(2, 2), name='convblock4_maxpool')(convblock4)
    # Bottleneck
    convblock5 = convolutional_block(input=maxpool4, filters=filters*16, dropout=0.3, name='convblock5')
    # Extension
    deconvblock1 = deconvolutional_block(input=convblock5,   skip=convblock4, filters=filters*8, dropout=0.3, name='deconvblock1')
    deconvblock2 = deconvolutional_block(input=deconvblock1, skip=convblock3, filters=filters*4, dropout=0.2, name='deconvblock2')
    deconvblock3 = deconvolutional_block(input=deconvblock2, skip=convblock2, filters=filters*2, dropout=0.2, name='deconvblock3')
    deconvblock4 = deconvolutional_block(input=deconvblock3, skip=convblock1, filters=filters*1, dropout=0.1, name='deconvblock4')
    # Output
    output = layers.Conv2D(n_classes, kernel_size=(1, 1), activation='sigmoid', name='output')(deconvblock4)
    # Model
    model   = models.Model(inputs=inputs, outputs=output, name='Unet')
    return model

unet = init_unet(n_classes=1, input_size=(256, 256, 3), filters=64)
unet.compile(optimizer='adam', loss='binary_focal_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])
del convolutional_block, deconvolutional_block

# Summary
# unet.summary()
summary = DataFrame([dict(Name=layer.name, Type=layer.__class__.__name__, Shape=layer.output_shape, Params=layer.count_params()) for layer in unet.layers])
summary.style.to_html(path.join(paths['models'], 'unet_structure.html'), index=False) 
del summary

utils.plot_model(unet, to_file=path.join(paths['models'], 'unet_structure.pdf'), show_shapes=True)`
#%% ESTIMATES PARAMETERS

# Loads model if estimated previously
# unet = models.load_model('../data_1960/models/unet_baseline.h5')

# Callbacks
train_callbacks = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    callbacks.ModelCheckpoint(filepath='../data_1960/models/unet_{epoch:02d}_{val_accuracy:.4f}.h5', monitor='val_accuracy', save_best_only=True),
    callbacks.BackupAndRestore(backup_dir='../data_1960/models')
]

# Training
training = unet.fit(
    train_generator,
    steps_per_epoch=samples_size['train'] // 32,
    validation_data=(images_valid, labels_valid),
    epochs=100,
    verbose=1,
    callbacks=train_callbacks
)

# Saves model and training history
# models.save_model(unet, '../data_1960/models/unet_baseline.h5')
# np.save('../data_1960/models/history_baseline.npy', training.history)

# Displays history
# history = np.load('../data_1960/models/history_baseline.npy',allow_pickle=True).item()
# display_history(history)
# del history

#%% EVALUATES MODEL

# Compute statistics
# performance = unet.evaluate(images_test, labels_test) # 81%
# print('Test loss: {:.4f}\nTest accuracy: {:.4f}\nTest recall: {:.4f}\nTest precision: {:.4f}'.format(*performance))
# del performance

# Displays statistics
probas_pred = unet.predict(images_test, verbose=1)
labels_pred = probas_pred >= 0.5

# Saves test data for statistics
for data in ['images_test', 'labels_test', 'probas_pred', 'labels_pred']:
    np.save(path.join(paths['statistics'], data + '.npy'), globals()[data])

# Displays prediction statistics
# for i in random.choice(range(len(images_test)), 5):
#     display_statistics(image_test=images_test[i], label_test=labels_test[i], proba_pred=probas_pred[i], label_pred=labels_pred[i])
# del images_test, labels_test, probas_pred, labels_pred

#%% PREDICTS NEW TILES

# Loads model
unet = models.load_model('../data_1960/models/unet_baseline.h5')

# Lists batches
batch_size = 5
batches = search_files(paths['images'], pattern='tif$')
batches = search_files(paths['images'], pattern=training)
batches = [batches[i:i + batch_size] for i in range(0, len(batches), batch_size)]

# Computes predictions
def predict_tiles(model, files):
    images  = np.array([read_raster(file) for file in files])
    images  = layers.Rescaling(1./255)(images)
    blocks1 = images_to_blocks(images=images, imagesize=images.shape, blocksize=(256, 256), shift=False)
    blocks2 = images_to_blocks(images=images, imagesize=images.shape, blocksize=(256, 256), shift=True)
    probas1 = unet.predict(blocks1)
    probas2 = unet.predict(blocks2)
    probas1 = blocks_to_images(probas1, imagesize=images.shape[:3] + (1,), blocksize=(256, 256), shift=False)
    probas2 = blocks_to_images(probas2, imagesize=images.shape[:3] + (1,), blocksize=(256, 256), shift=True)
    probas  = (probas1 + probas2) / 2
    del blocks1, blocks2, probas1, probas2
    return probas

# Computes predictions
for files in batches:
    probas   = predict_tiles(unet, files)
    outfiles = [path.join(paths['predictions'], path.basename(file).replace('image', 'proba')) for file in files]
    for proba, file, outfile in zip(probas, files, outfiles):
        write_raster(array=proba, source=file, destination=outfile, nodata=-1, dtype='float32')

