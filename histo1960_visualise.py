#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Predictions for the Arthisto 1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
'''

#%% HEADER

# Modules
import itertools
import numpy as np
import tensorflow

from histo1960_utilities import *
from keras import layers, models
from os import path
from skimage import color, exposure

# Sets tensorflow verbosity
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

# Samples
cities = '(0400_6420|0625_6870|0650_6870|0875_6245|0875_6270|0825_6520|0825_6545|0550_6295|0575_6295).tif$'

#%% FUNCTIONS

def mean_loss(input_image, filter_index):
    '''Mean loss function'''
    activation = extractor(input_image)
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tensorflow.reduce_mean(filter_activation)

@tensorflow.function
def gradient_ascent_step(image, filter_index, learning_rate):
    with tensorflow.GradientTape() as tape:
        tape.watch(image)
        loss  = mean_loss(image, filter_index)
    # Compute gradients.
    gradients  = tape.gradient(loss, image)
    # Normalize gradients.
    gradients  = tensorflow.math.l2_normalize(gradients)
    image     += learning_rate * gradients
    return loss, image

def visualise_filter(index:int, iterations:int=30, learning_rate:float=10.0):
    # Initialises image
    image = np.random.uniform(-0.125, 0.125, size=(1, 256, 256, 3))
    # Gradient optimisation
    for iteration in range(iterations):
        loss, image = gradient_ascent_step(image, index, learning_rate)
    # Decode the resulting input image
    image = image[0].numpy()
    image = deprocess_image(image)
    return loss, image

def deprocess_image(image:np.ndarray):
    # Normalize array: center on 0., ensure variance is 0.15
    image -= image.mean()
    image /= image.std() + 1e-5
    image *= 0.15
    # Center crop
    image = image[28:-28, 28:-28, :]
    # Clip to [0, 1]
    image += 0.5
    image = np.clip(image, 0, 1)
    # Convert to RGB array
    image *= 255
    image = np.clip(image, 0, 255).astype("uint8")
    return image

def display_grid(images:list, gridsize:tuple, figsize:tuple=(20, 20.4), cmap:str='gray', path:str=None, dpi:int=300) -> None:
    '''Displays multiple images'''
    fig, axs = pyplot.subplots(nrows=gridsize[0], ncols=gridsize[1], figsize=figsize)
    for ax, image in zip(axs.ravel(), images):
        ax.imshow(image, cmap=cmap)
        ax.set_axis_off()
        pyplot.tight_layout(pad=0.5)
    if path is not None:
        pyplot.savefig(path, dpi=dpi)
    else:
        pyplot.show()

#%% VISUALISE ACTIVATIONS

# Loads model
model = models.load_model(path.join(paths['models'], 'unet64_220609.h5'))
model.summary()

# Loads data
images = np.load(path.join(paths['statistics'], 'images_test.npy'))
keep   = list(map(not_empty, images, itertools.repeat('any')))
images = images[keep]

# Pick an image
random.shuffle(images)
display_grid(images[:16], (4, 4))
image  = images[[5]]
display(np.squeeze(image), path=path.join(paths['figures'], f'fig_activations_image.jpg'))

# Extract activations
for layer in ['encoder1', 'encoder2', 'bottleneck']:
    print(f'Processing {layer}')
    extractor   = model.get_layer(name=f'{layer}_activation2')
    extractor   = models.Model(inputs=model.inputs, outputs=extractor.output)
    activations = extractor.predict(image)
    activations = np.swapaxes(activations, 0, 3)
    activations = exposure.rescale_intensity(activations, out_range=(0, 1))
    random.shuffle(activations)
    display_grid(activations[:16], (4, 4), path=path.join(paths['figures'], f'fig_activations_{layer}.jpg'))

#%% VISUALISE FILTERS

# Loads model
model = tensorflow.keras.applications.ResNet50V2(weights='imagenet', include_top=False)
model.summary()

# Selects layer
layer     = model.get_layer(name='conv3_block4_out')
nfeat     = layer.output_shape[-1]
extractor = models.Model(inputs=model.inputs, outputs=layer.output)

# Single filter
loss, image = visualise_filter(0, iterations=30, learning_rate=10)
display(image)

# All filters
images = list()
losses = list()
for filter_index in range(64):
    print(f'Processing image {filter_index:02d}', end =' - ')
    loss, image = visualise_filter(filter_index)
    print(f'Loss: {float(loss):.4f}')
    images.append(image)
    losses.append(loss)

images = np.array(images)
random.shuffle(images)

display_grid(images[:16], (4, 4), path=path.join(paths['figures'], f'fig_filters.jpg'), dpi=150)

# %%

