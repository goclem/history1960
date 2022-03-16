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
    data='../data_1960',
    images='../data_1960/images',
    labels='../data_1960/labels',
    predictions='../data_1960/predictions',
    models='../data_1960/models'
)

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
pattern = '({}).tif'.format('|'.join(pattern))

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

def conv_block(tensor, n_filters, dropout=0, kernel_size=(3, 3), padding='same', initializer='he_normal'):
    tensor = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer)(tensor)
    tensor = layers.Activation('relu')(tensor)
    tensor = layers.BatchNormalization()(tensor)
    tensor = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer)(tensor)
    tensor = layers.Activation('relu')(tensor)
    tensor = layers.BatchNormalization()(tensor)
    tensor = layers.SpatialDropout2D(dropout)(tensor)
    return tensor

def deconv_block(tensor, residual, n_filters, kernel_size=(3, 3), padding='same', strides=(2, 2), dropout=0):
    tensor = layers.Conv2DTranspose(n_filters, kernel_size=kernel_size, strides=strides, padding=padding)(tensor)
    tensor = layers.concatenate([tensor, residual], axis=3)
    tensor = conv_block(tensor=tensor, n_filters=n_filters, dropout=dropout)
    return tensor

def init_unet(n_classes:int, input_size:tuple, n_filters:int):
    # Input
    inputs  = layers.Input(shape=input_size, name='image_input')
    # Contraction
    block1 = conv_block(inputs, n_filters=n_filters*1, dropout=0.1)
    mpool1 = layers.MaxPooling2D(pool_size=(2, 2))(block1)
    block2 = conv_block(mpool1, n_filters=n_filters*2, dropout=0.1)
    mpool2 = layers.MaxPooling2D(pool_size=(2, 2))(block2)
    block3 = conv_block(mpool2, n_filters=n_filters*4, dropout=0.2)
    mpool3 = layers.MaxPooling2D(pool_size=(2, 2))(block3)
    block4 = conv_block(mpool3, n_filters=n_filters*8, dropout=0.2)
    mpool4 = layers.MaxPooling2D(pool_size=(2, 2))(block4)
    # Bottleneck
    block5 = conv_block(mpool4, n_filters=n_filters*16, dropout=0.3)
    # Extension
    block6 = deconv_block(block5, residual=block4, n_filters=n_filters*8, dropout=0.3)
    block7 = deconv_block(block6, residual=block3, n_filters=n_filters*4, dropout=0.2)
    block8 = deconv_block(block7, residual=block2, n_filters=n_filters*2, dropout=0.2)
    block9 = deconv_block(block8, residual=block1, n_filters=n_filters*1, dropout=0.1)
    # Output
    outputs = layers.Conv2D(n_classes, kernel_size=(1, 1), activation='sigmoid')(block9)
    # Model
    model   = models.Model(inputs=inputs, outputs=outputs, name='Unet')
    return model

unet = init_unet(n_classes=1, input_size=(256, 256, 3), n_filters=16)
unet.summary()
unet.compile(optimizer='adam', loss='binary_focal_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])
del conv_block, deconv_block

#%% ESTIMATES PARAMETERS

# Callbacks
train_callbacks = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    callbacks.ModelCheckpoint(filepath='../data_1960/models/unet_{epoch:02d}_{val_accuracy:.4f}.h5', monitor='val_accuracy', save_best_only=True),
    callbacks.BackupAndRestore(backup_dir='../data_1960/models')
]

training = unet.fit(
    train_generator,
    steps_per_epoch=samples_size['train'] // 32,
    validation_data=(images_valid, labels_valid),
    epochs=100,
    verbose=1,
    callbacks=train_callbacks
)

# Saves model and training history
models.save_model(unet, '../data_1960/models/unet_baseline.h5')
np.save('../data_1960/models/history_baseline.npy', training.history)

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
for i in random.choice(range(len(images_test)), 5):
    display_statistics(image_test=images_test[i], label_test=labels_test[i], proba_pred=probas_pred[i], label_pred=labels_pred[i])
del images_test, labels_test, probas_pred, labels_pred

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

#%% FORMATS PREDICTIONS

# Computes labels and vectors
files = search_files(paths['predictions'], pattern='proba.*tif$')
for file in files:
    print(path.basename(file))
    outfile = file.replace('proba', 'label')
    os.system('gdal_calc.py --overwrite -A {} --outfile={} --calc="A>=0.5" --NoDataValue=0 --type=Byte --quiet'.format(file, outfile))
    os.system('gdal_polygonize.py {} {} -q'.format(outfile, outfile.replace('tif', 'gpkg')))
    os.remove(outfile)
del files, outfile

# Aggregates all vectors
pattern = path.join(paths['predictions'], '*.gpkg')
outfile = path.join(paths['data'], 'building1960.gpkg')
os.system('ogrmerge.py -single -overwrite_ds -f GPKG -o {} {}'.format(outfile, pattern))
os.system('find {}/ -name "*.gpkg$" -type f -delete'.format(paths['predictions']))
del pattern, outfile

# Open files for checking
pattern = search_files(directory=paths['predictions'], pattern='proba.*tif$')
pattern = '({}).tif'.format('|'.join(identifiers(pattern)))
files   = search_files(directory=paths['images'], pattern=training)
for file in files:
    os.system('open {}'.format(file))

#%% UTILITIES

training    = '(0350_6695|0400_6445|0550_6295|0575_6295|0700_6520|0700_6545|0700_7070|0875_6245|0875_6270|0900_6245|0900_6270|0900_6470|1025_6320).tif'
legend_1900 = '(0600_6895|0625_6895|0600_6870|0625_6870|0625_6845|0600_6845|0650_6895|0650_6870|0650_6845|0675_6895|0675_6870|0675_6845|0850_6545|0825_6545|0850_6520|0825_6520|0825_6495).tif'
legend_N    = '(0400_6570|0425_6570|0400_6595|0425_6595|0425_6545|0400_6545|0425_6520|0400_6520|0425_6395|0425_6420|0400_6395|0400_6420|0425_6720|0450_6720|0425_6745|0450_6745|0450_6695|0425_6695|0425_6670|0450_6670|0450_6570|0450_6595|0450_6545|0450_6520|0450_6945|0450_6920|0475_6920|0475_6795|0500_6795|0475_6770|0500_6770|0500_6720|0475_6720|0475_6695|0500_6695|0475_6670|0450_6645|0475_6645|0500_6645|0525_6670|0500_6670|0525_6645|0500_6620|0525_6620|0475_6620|0550_6820|0525_6820|0550_6895|0575_6895|0550_6870|0575_6870|0575_6845|0550_6845|0550_6670|0575_6670|0550_6695|0575_6695|0575_6645|0550_6645|0475_6495|0450_6495|0475_6470|0450_6470|0450_6420|0450_6395|0475_6420|0475_6395|0475_6320|0500_6320|0525_6495|0500_6495|0500_6520|0525_6520|0525_6320|0525_6345|0500_6345|0600_6670|0600_6695|0600_6645|0625_6495|0650_6495|0650_6520|0625_6520|0725_6320|0700_6320|0725_6345|0700_6345|0775_6420|0750_6420|0725_6420|0775_6445|0750_6445|0725_6445|0775_6395|0725_6395|0750_6395|0775_6370|0800_6370|0775_6345|0800_6345|1150_6170|1150_6145|1150_6120|1175_6195|1150_6195|1175_6170|1175_6145|1175_6120|1175_6095|1150_6095|1200_6095|1175_6070|1200_6070|1200_6220|1200_6195|1175_6220|1200_6170|1200_6145|1225_6170|1225_6145|1200_6120|1225_6120|1225_6095|1250_6120|1250_6145).tif'