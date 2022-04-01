#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Performs predictions for the Arthisto 1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.03.28
'''
# Source: https://github.com/keras-team/keras-io/blob/master/examples/generative/dcgan_overriding_train_step.py
#%% HEADER

# Modules
import numpy as np
import tensorflow

from arthisto1960_utilities import *
from numpy import random
from os import path
from tensorflow import keras

# TensorFlow
print('TensorFlow version:', tensorflow.__version__)
print('GPU Available:', len(tensorflow.config.experimental.list_physical_devices('GPU')))

# Paths
os.chdir('/Users/clementgorin/Dropbox/research/arthisto/arthisto1960')

paths = dict(
    data='../data_1960',
    images='../data_1960/images',
    labels='../data_1960/labels',
    models='../data_1960/models',
    predictions='../data_1960/predictions',
    statistics='../data_1960/statistics'
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

# Generator
data_generator   = keras.preprocessing.image.ImageDataGenerator(rescale=1)
images_generator = data_generator.flow(images, batch_size=32, shuffle=True, seed=1)

#%% DISCRIMINATOR

discriminator = keras.Sequential(
    [
        keras.Input(shape=(256, 256, 3)),
        keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        # keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
        # keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid'),
    ],
    name='discriminator',
)
discriminator.summary()

#%% GENERATOR MODEL

latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        keras.layers.Dense(32 * 32 * 64),
        keras.layers.Reshape((32, 32, 64)),
        keras.layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2DTranspose(16, kernel_size=4, strides=2, padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2DTranspose(8, kernel_size=4, strides=2, padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(3, kernel_size=5, padding='same', activation='sigmoid'),
    ],
    name='generator',
)
generator.summary()

#%% OVERRIDES TRAINING

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tensorflow.shape(real_images)[0]
        random_latent_vectors = tensorflow.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tensorflow.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tensorflow.concat(
            [tensorflow.ones((batch_size, 1)), tensorflow.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tensorflow.random.uniform(tensorflow.shape(labels))

        # Train the discriminator
        with tensorflow.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tensorflow.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tensorflow.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tensorflow.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }

#%% CALLBACKS

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tensorflow.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("generated_img_%03d_%d.png" % (epoch, i))


#%% TRAINING

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

gan.fit(
    images_generator, 
    epochs=15, 
    steps_per_epoch=len(images) // 32
    #callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)]
)

#%% PREDICTIONS

newimages = gan.generator(tensorflow.random.normal(shape=(2, 128))) # Batch size and random vector
newimages = (np.array(newimages))
display(newimages[1])
# %%
