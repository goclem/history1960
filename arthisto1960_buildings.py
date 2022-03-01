#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Workflow for the semantic segmentation example
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.02.15
'''

'''
Setup Keras with GPU:
- conda create -n myenv python=3.7 (only supported version)
- conda activate myenv
- pip install -U plaidml-keras (install keras but not tensorflow)
- conda install tensorflow (should install version compatible with plaidml)
- plaidml-setup (no experimental device, then select the GPU)
'''

#%% HEADER

# Modules
import easydict
import itertools
import matplotlib.pyplot
import numpy as np
import os
import rasterio
import rasterio.plot
import rasterio.windows
import re
import shutil

# Keras
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# import keras
# import keras.preprocessing
# import keras.utils
# import keras.models
# import keras.layers

# Paths
paths = easydict.EasyDict({
    'images_raw':        '../data_1960/raw/images',
    'labels_raw':        '../data_1960/raw/labels',
    'images_training':   '../data_1960/training/images',
    'labels_training':   '../data_1960/training/labels',
    'images_validation': '../data_1960/validation/images',
    'labels_validation': '../data_1960/validation/labels',
    'images_testing':    '../data_1960/testing/images',
    'labels_testing':    '../data_1960/testing/labels',
    'models':            '../data_1960/fh',
    'figures':           '../data_1960/figures',
    'temporary':         '/Users/clementgorin/Temporary'
    })

#%% FUNCITONS

def search_files(directory:str, pattern:str='.') -> list:
    files = []
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    return files

def initialise_directory(directory:str, remove:bool=False):
    if not os.path.exists(folder):
        os.mkdir(folder)
    if os.path.exists(folder) and remove is True:
        shutil.rmtree(folder)
        os.mkdir(folder)

def read_image(file:str) -> np.ndarray:
    raster = rasterio.open(file)
    raster = raster.read()
    image  = raster.transpose([1, 2, 0])
    return image

def write_image(image:np.array, file:str, dtype=rasterio.uint8):
    height, width, bands = image.shape
    args   = dict(fp=file, mode='w', driver='GTiff', height=height, width=width, count=bands, dtype=dtype)
    raster = rasterio.open(**args)
    raster.write(image.transpose([2, 0, 1]))
    raster.close()
    return image

def standardise_image(image:np.ndarray) -> np.ndarray:
    bandmeans   = np.mean(image, axis=(0, 1), keepdims=True)
    bandstds    = np.std(image,  axis=(0, 1), keepdims=True)
    standarised = (image - bandmeans) / bandstds
    return standarised

def images_to_blocks(images:np.ndarray, imagesize:tuple, blocksize:tuple=(256, 256), shift:bool=False) -> np.ndarray:
    # Defines quantities
    nimages, imagewidth, imageheight, nbands = imagesize
    blockwidth, blockheight = blocksize
    nblockswidth  = (imagewidth  // blockwidth  + 1 + shift)
    nblocksheight = (imageheight // blockheight + 1 + shift)
    # Defines padding
    padwidth  = int(((nblockswidth)  * blockwidth  - imagewidth)  / 2)
    padheight = int(((nblocksheight) * blockheight - imageheight) / 2)
    # Maps images to blocks
    images = np.pad(images, ((0, 0), (padwidth, padwidth), (padheight, padheight), (0, 0)))
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

def sample_split(images:np.ndarray, sizes:dict, seed:int=1):
    np.random.seed(seed)
    samples = list(sizes.keys())
    indexes = np.random.choice(samples, images.shape[0], p=list(sizes.values()))
    samples = [images[indexes == sample, ...] for sample in samples]
    return samples

def display_images(i:int):
    fig, axs = matplotlib.pyplot.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(images_test[i,...])
    axs[1].imshow(labels_test[i,...])
    axs[2].imshow(probas_predict[i,...])
    axs[3].imshow(labels_predict[i,...])
    axs[0].set_title('Test image', fontsize=20)
    axs[1].set_title('Test labels', fontsize=20)
    axs[2].set_title('Predicted probability', fontsize=20)
    axs[3].set_title('Predicted labels', fontsize=20)
    for ax in axs.ravel():
        ax.set_axis_off()
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()

def compute_statistics(i, savefig:bool=True):
    # Data
    image_test    = images_test[i,...]
    label_test    = labels_test[i,...].astype(bool).reshape(256, 256)
    proba_predict = probas_predict[i,...].reshape(256, 256)
    label_predict = labels_predict[i,...].reshape(256, 256)
    # Statistics
    label_tp = np.logical_and(label_test, label_predict)
    label_tn = np.logical_and(np.invert(label_test), np.invert(label_predict))
    label_fp = np.logical_and(np.invert(label_test), label_predict)
    label_fn = np.logical_and(label_test, np.invert(label_predict))
    # Descriptive statistics
    tp, tn, fp, fn = {np.sum(label_tp), np.sum(label_tn), 
                      np.sum(label_fp), np.sum(label_fn)}
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    f1        = tp / (tp + 0.5 * (fp + fn))
    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    print((
        'Precision: {:.2%} of building predicted building pixels are actual buildings\n'
        'Recall:    {:.2%} of buildings pixels are correctly predicted as buildings\n'
        'F1:        {:.2%}\n'
        'Accuracy:  {:.2%}').format(precision, recall, f1, accuracy))
    # Figure
    colour   = (255,255,0)
    fig, axs = matplotlib.pyplot.subplots(2, 4, figsize=(20, 10))
    axs[0,0].imshow(images_test[i,...])
    axs[0,1].imshow(labels_test[i,...])
    axs[0,2].imshow(probas_predict[i,...])
    axs[0,3].imshow(labels_predict[i,...])
    axs[1,0].imshow(np.where(np.dstack((label_tp, label_tp, label_tp)), colour, image_test))
    axs[1,1].imshow(np.where(np.dstack((label_tn, label_tn, label_tn)), colour, image_test))
    axs[1,2].imshow(np.where(np.dstack((label_fp, label_fp, label_fp)), colour, image_test))
    axs[1,3].imshow(np.where(np.dstack((label_fn, label_fn, label_fn)), colour, image_test))
    axs[0,0].set_title('Test image',            fontsize=20)
    axs[0,1].set_title('Test labels',           fontsize=20)
    axs[0,2].set_title('Predicted probability', fontsize=20)
    axs[0,3].set_title('Predicted labels',      fontsize=20)
    axs[1,0].set_title('True postive',          fontsize=20)
    axs[1,1].set_title('True negative',         fontsize=20)
    axs[1,2].set_title('False positive',        fontsize=20)
    axs[1,3].set_title('False negative',        fontsize=20)
    for ax in axs.ravel():
        ax.set_axis_off()
    matplotlib.pyplot.tight_layout()
    if savefig:
        file = os.path.join(paths.figures, 'predstats_tile{:d}.png'.format(i))
        matplotlib.pyplot.savefig(file, facecolor='white', dpi=100, format='png')
    matplotlib.pyplot.show()

#%% TRAINING DATA

"""
Notes:
- Vannes  (0250_6745) is incomplete
- Orleans (0600_6770) is incomplete
"""

# Loads images as blocks
images = search_files(paths.images_raw, 'tiff?$')
images = np.array([read_image(file) for file in images])
images = np.concatenate((
    images_to_blocks(images=images, imagesize=images.shape, blocksize=(256, 256), shift=True),
    images_to_blocks(images=images, imagesize=images.shape, blocksize=(256, 256), shift=False)
))

# Loads labels as blocks
labels = search_files(paths.labels_raw, 'tiff?$')
labels = np.array([read_image(file) for file in labels])
labels = np.concatenate((
    images_to_blocks(images=labels, imagesize=labels.shape, blocksize=(256, 256), shift=True),
    images_to_blocks(images=labels, imagesize=labels.shape, blocksize=(256, 256), shift=False)
))

# Computes samples
probas = {'training':0.8, 'validation':0.1, 'testing':0.1}
images_training, images_validation, images_testing = sample_split(images=images, probas=probas, seed=1)
labels_training, labels_validation, labels_testing = sample_split(images=labels, probas=probas, seed=1)
del(images, labels, probas)

# Writes samples
for folder in [paths.images_training, paths.labels_training, paths.images_validation, paths.labels_validation, paths.images_testing, paths.labels_testing]:
    reset_folder(folder=folder, remove=True)

for i in range(images_training.shape[0]):
    write_image(image=images_training[i,...], file=os.path.join(paths.images_training, 'image_train_{:05d}.tif'.format(i)))
    write_image(image=labels_training[i,...], file=os.path.join(paths.labels_training, 'label_train_{:05d}.tif'.format(i)))

for i in range(images_validation.shape[0]):
    write_image(image=images_validation[i,...], file=os.path.join(paths.images_validation, 'image_valid_{:05d}.tif'.format(i)))
    write_image(image=labels_validation[i,...], file=os.path.join(paths.labels_validation, 'label_valid_{:05d}.tif'.format(i)))

for i in range(images_testing.shape[0]):
    write_image(image=images_testing[i,...], file=os.path.join(paths.images_testing, 'image_test_{:05d}.tif'.format(i)))
    write_image(image=labels_testing[i,...], file=os.path.join(paths.labels_testing, 'label_test_{:05d}.tif'.format(i)))

del(images_training, labels_training, images_validation, labels_validation, images_testing, labels_testing)




#%% MODEL

def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = keras.layers.Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x

def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    y = keras.layers.Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = keras.layers.concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y

def Unet(n_classes:int, input_size:tuple, nfilters:int):
    # Input
    inputs  = keras.layers.Input(shape=input_size, name='image_input')
    # Contraction
    conv1   = conv_block(inputs, nfilters=nfilters)
    pool1   = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2   = conv_block(pool1, nfilters=nfilters*2)
    pool2   = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3   = conv_block(pool2, nfilters=nfilters*4)
    pool3   = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4   = conv_block(pool3, nfilters=nfilters*8)
    pool4   = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4   = keras.layers.Dropout(0.5)(pool4)
    conv5   = conv_block(pool4, nfilters=nfilters*16)
    conv5   = keras.layers.Dropout(0.5)(conv5)
    # Extension
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=nfilters*8)
    deconv6 = keras.layers.Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=nfilters*4)
    deconv7 = keras.layers.Dropout(0.5)(deconv7)
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=nfilters*2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=nfilters)
    # Output
    outputs = keras.layers.Conv2D(n_classes, kernel_size=(1, 1), activation='sigmoid')(deconv9)
    # Model
    model   = keras.models.Model(inputs=inputs, outputs=outputs, name='Unet')
    return model

model = Unet(n_classes=1, input_size=(256, 256, 3), nfilters=16)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#%% PRE-PROCESSING

# Pre-processing
images_generator = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=45,
    width_shift_range=0.25,
    height_shift_range=0.25,
    brightness_range=(0.75, 1.25),
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1)

labels_generator = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.25,
    height_shift_range=0.25,
    brightness_range=(0.75, 1.25),
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1)

images_generator.fit(images_train[:10], augment=True, seed=1)
labels_generator.fit(labels_train[:10], augment=True, seed=1)

images_generator = images_generator.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=1)

labels_generator = labels_generator.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=1)

# combine generators into one which yields image and masks
train_generator = zip(images_generator, labels_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)

# unet.fit_generator(images_train, augment=True, seed=1)

#%% ESTIMATES PARAMETERS

callbacks = [
    keras.callbacks.EarlyStopping(patience=2),
    keras.callbacks.ModelCheckpoint(filepath=os.path.join(paths.models, 'unet.{epoch:02d}-{val_acc:.4f}.h5')),
]

# (!) time
training = model.fit(
    images_train, labels_train, 
    validation_split=0.1,
    batch_size=32, 
    epochs=2,
    verbose=1,
    callbacks=callbacks)

training.history

#%% PREDICTS IMAGES

# Predict images
# performance = unet.evaluate(images_test, labels_test) # 81%
# print('Test loss: {:.4f}\nTest accuracy: {:.4f}'.format(*performance))

probas_predict = unet.predict(images_train[:100,...], verbose=1)
labels_predict = probas_predict >= 0.5

# For plot only
images_test = search_files(paths.train_images, 'tiff?$')
images_test = [read_as_blocks(file, standardise=False) for file in images_test]
images_test = np.array(list(itertools.chain(*images_test)))
images_test = list(itertools.compress(images_test, index))
images_test = np.array(images_test)
labels_test = labels_train


compute_statistics(1)
compute_statistics(8)
compute_statistics(12)
compute_statistics(35)
compute_statistics(38)
compute_statistics(40)
compute_statistics(43)
compute_statistics(45)
compute_statistics(47)
compute_statistics(48)
compute_statistics(49)
compute_statistics(50)
compute_statistics(51)
compute_statistics(52)
compute_statistics(53)



os.path.exists(paths.figures)
matplotlib.pyplot.imshow(images_train[1,...])
matplotlib.pyplot.imshow(labels_train[1,...])
matplotlib.pyplot.imshow(images_test[1,...])
matplotlib.pyplot.imshow(labels_test[1,...])

#%% DEPRECIATED

# Standardise images
# keras.preprocessing.image.ImageDataGenerator
# datagen = ImageDataGenerator(rescale=1.0/255.0)
# train_iterator = datagen.flow(images_train, trainY, batch_size=64)
# valid_iterator = datagen.flow(images_valid, testY, batch_size=64)