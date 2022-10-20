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
from matplotlib import offsetbox, patheffects
from numpy import linalg
from os import path
from skimage import color, exposure, transform
from sklearn import cluster, decomposition, manifold

# Sets tensorflow verbosity
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

# Samples
cities = '(0400_6420|0625_6870|0650_6870|0875_6245|0875_6270|0825_6520|0825_6545|0550_6295|0575_6295).tif$'

#%% FUNCTIONS

def mean_loss(extractor, input_image, filter_index):
    '''Mean loss function'''
    activation = extractor(input_image)
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tensorflow.reduce_mean(filter_activation)

@tensorflow.function
def gradient_ascent_step(extractor, image, filter_index, learning_rate):
    with tensorflow.GradientTape() as tape:
        tape.watch(image)
        loss  = mean_loss(extractor, image, filter_index)
    # Compute gradients.
    gradients  = tape.gradient(loss, image)
    # Normalize gradients.
    gradients  = tensorflow.math.l2_normalize(gradients)
    image     += learning_rate * gradients
    return loss, image

def visualise_filter(layer, index:int, iterations:int=30, learning_rate:float=10.0):
    # Initialises image
    image = np.random.uniform(-0.125, 0.125, size=(1, 256, 256, 3))
    # Initialises model
    extractor = models.Model(inputs=model.inputs, outputs=layer.output)    
    # Gradient optimisation
    for iteration in range(iterations):
        loss, image = gradient_ascent_step(extractor, image, index, learning_rate)
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
    image = np.clip(image, 0, 255).astype('uint8')
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

#%% VISUALISE LAYERS OUTPUT

# Loads model
model = models.load_model(path.join(paths['models'], 'unet64_220609.h5'))
for i in range(len(model.layers)):
  print(model.get_layer(index=i).name)

# Loads data
images = np.load(path.join(paths['statistics'], 'images_test.npy'))
keep   = list(map(not_empty, images, itertools.repeat('any')))
images = images[keep]

# Pick an image
random.seed(3)
random.shuffle(images)
display_grid(images, (5, 5), figsize=(20, 20.5))
image  = images[[8]]

display(np.squeeze(image))

# Extract activations
activations = list()
for layer_name in ['encoder1', 'encoder2', 'encoder3', 'encoder4', 'bottleneck']:
    print(f'Processing {layer_name}')
    extractor  = model.get_layer(name=f'{layer_name}_activation2')
    extractor  = models.Model(inputs=model.inputs, outputs=extractor.output)
    activation = extractor.predict(image)
    activation = np.swapaxes(activation, 0, 3)
    sample     = random.choice(np.arange(len(activation)), 8)
    activation = activation[sample]
    activation = exposure.rescale_intensity(activation, out_range=(0, 1))
    activation = [transform.resize(img, (256, 256)) for img in activation]
    activations.append(activation)

activations = np.array(sum(activations, []))

display_grid(activations, (5, 5), figsize=(20, 20.5), cmap='inferno')
display_grid(activations, (5, 5), figsize=(20, 20.5), cmap='inferno', path=path.join(paths['figures'], f'fig_activations.jpg'), dpi=150)

#%% VISUALISE FILTERS

# Loads model
model = tensorflow.keras.applications.ResNet50V2(weights='imagenet', include_top=False)
# model = models.load_model(path.join(paths['models'], 'unet64_220609.h5'))

for i in range(len(model.layers)):
  print(model.get_layer(index=i).name)

# Single filter
layer    = model.get_layer(name='bottleneck_activation1')
_, image = visualise_filter(layer, 1)
display(image)

# All filters
images = list()
layer_names = ['conv1_conv', 'conv2_block3_out', 'conv3_block3_out', 'conv4_block3_out', 'conv5_block3_out']
# layer_names = ['encoder1_activation2', 'encoder2_activation2', 'encoder3_activation2', 'encoder4_activation2', 'bottleneck_activation2']
for layer_name in layer_names:
    print(f'Processing layer {layer_name}')
    layer = model.get_layer(name=layer_name)
    nfeat = layer.output_shape[-1]
    for filter_index in random.choice(np.arange(nfeat), 5):
        print(f'Processing filter {filter_index:04d}')
        loss, image = visualise_filter(layer, filter_index)
        images.append(image[50:-50, 50:-50, ...])

images = np.array(images)
display_grid(images, (5, 5), figsize=(20, 20.5))
display_grid(images, (5, 5), figsize=(20, 20.5), path=path.join(paths['figures'], f'fig_filters.jpg'), dpi=150)

#%% REPRESENTATIONS

# Loads data
images = np.load(path.join(paths['statistics'], 'images_test.npy'))
keep   = list(map(not_empty, images, itertools.repeat('any')))
images = images[keep] / 255

# Loads model
model = models.load_model(path.join(paths['models'], 'unet64_220609.h5'))

# Extracts representations
extractor = model.get_layer(name=f'bottleneck_activation1')
extractor = models.Model(inputs=model.inputs, outputs=extractor.output)
represent = extractor.predict(images)

# Averages representations
represent = np.mean(represent, axis=(1,2))

# Maintains some spatial dimension
# represent = extractor.predict(images)
# represent = layers.AveragePooling2D((8, 8))(represent)
# represent = represent.numpy().reshape(916, 2*2*1024)

# TSNE (clusters)
tsne = manifold.TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, learning_rate='auto', random_state=1)
x, y = tsne.fit_transform(represent).T
x = np.interp(x, (x.min(), x.max()), (-1, 1))
y = np.interp(y, (y.min(), y.max()), (-1, 1))

# PCA (distances)
# pca  = decomposition.PCA(2)
# x, y = pca.fit_transform(represent).T
# x = np.interp(x, (x.min(), x.max()), (-1, 1))
# y = np.interp(y, (y.min(), y.max()), (-1, 1))

# Clustering (! not recommended)
xy = np.column_stack((x, y))
bandwidth = cluster.estimate_bandwidth(xy)
meanshift = cluster.MeanShift(bandwidth=0.2, bin_seeding=True)
meanshift.fit(xy)
groups    = meanshift.predict(xy)

# Scatter plot
fig = pyplot.figure(figsize=(10, 10))
ax  = pyplot.axes()
ax.scatter(x, y, c=groups)
for l, xy in zip(np.unique(meanshift.labels_), meanshift.cluster_centers_):
    ax.annotate(str(l), (xy[0], xy[1]), fontsize=20, path_effects=[patheffects.withStroke(linewidth=5, foreground='white')])
pyplot.tight_layout()
# pyplot.savefig(path.join(paths['figures'], f'fig_tsne.jpg'), dpi=300)
pyplot.show()

# Scatter plot with images
# np.unique(meanshift.labels_)
for i, group in enumerate([0, 4, 7]):
    print(f'Processing group {group}')
    subset = groups == group
    fig, ax = pyplot.subplots(figsize=(10, 10)) 
    for xx, yy, image in zip(x[subset], y[subset], images[subset]):
        img = transform.rescale(image, 0.25,  preserve_range=True, channel_axis=2)
        img = offsetbox.OffsetImage(img, zoom=1)
        box = offsetbox.AnnotationBbox(img, (xx, yy), frameon=True, pad=0.1)
        ax.add_artist(box)
        ax.update_datalim([(xx, yy)])
        ax.autoscale()
        ax.set_axis_off()
    pyplot.grid()
    pyplot.tight_layout()
    # pyplot.savefig(path.join(paths['figures'], f'fig_cluster{i}.jpg'), dpi=300)
    pyplot.show()

#%% ASSOCIATIONS

def cosine_similarity(x1, x2):
    '''Computes cosine similarity between two vectors'''
    x1, x2 = (np.squeeze(x1), np.squeeze(x2))
    cosim  = np.dot(x1, x2) / (linalg.norm(x1) * linalg.norm(x2))
    return cosim

# Image of a legend - legend vector + another legend vector ?

display_grid(images[groups == 0][:25], (5, 5))

image            = images[groups == 0][13]
image_represent  = represent[groups == 5][13]
group1_represent = np.mean(represent[groups == 5], axis=0)
group2_represent = np.mean(represent[groups == 4], axis=0)

operation = image_represent - group1_represent + group2_represent
distance  = np.apply_along_axis(lambda x: cosine_similarity(operation, x), 1, represent)
indexes   = distance.argsort()[-5:][::-1]
matches   = images[indexes]
match = transform.rotate(matches[0], 90)

compare(matches)
compare([image, matches[0]])

display(image, path=path.join(paths['figures'], f'fig_association0.jpg'))
display(match, path=path.join(paths['figures'], f'fig_association1.jpg'))