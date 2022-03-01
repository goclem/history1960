# Notes on semantic segmentation

Semantic segmentation refers to the process of linking each pixel in an image to a class label. We can think of semantic segmentation as image classification at a pixel level. For example, in an image that has many cars, segmentation will label all the objects as car objects. However, a separate class of models known as instance segmentation is able to label the separate instances where an object appears in an image. This kind of segmentation can be very useful in applications that are used to count the number of objects, such as counting the amount of foot traffic in a mall.

Semantic segmentation performs two tasks, which are learned simultaneously during training. The firt is what by mapping a pixel input to a discrete label, the second is where by assigning this class to all pixels composing the object. Traditional neural netorks are concerned with what.

## Data

**Format**: Images need to be stored in a lossless format such as *.tiff* ou *.png*. Each training observation is composed of a raw image (X) and a mask of the same dimension (y).

**Size**: We probably need to split the image into several smaller images with an overlap. At the overlap, probabilities can be aggregated using a weighted average whose weights depend on the distance to the centre of the image. The image with the largest par of the object is more likely to form a correct prediction. However, since forming correct prediction depends on the spatial context of the image, too small images may come at the expense of accuracy.

## Pre-processing

**Colours**: Since the network needs to distinguish between essentially black patterns, the image may be converted into gray scale. This would reduce the computational requirement and allow to train a deeper network. The performance of the network is directly linked to the number of layers it has.

**Contrast**: The contrast may be enhanced using a locally adaptive contrast enhancement method such as CLAHE in OpenCV. This dramatically increased the accuracy on the Etat-Major project.

**Normalisation**: Any regularised model is sensitive to scale so the input needs to be normalised between 0 and 1.


## Class imbalances

In remote sensing, class im- balance represents often a problem for tasks like land cover mapping, as small objects get less prioritised in an effort to achieve the best overall accuracy.

## Modelling

**Keras** is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano libraries.

**Class of modes**: Is is best to use semantic or instance segmentation.

**Small objects**: Need for very precise boundaries. In traditional segmentations tasks, the classified objects use most of the image (see paper). Maybe Unet is decent because medical imaging are mainly black and white, and detected objects can be small.

## Training

**Pre-trained networks** for historical maps (see papers). There are some pre-trained architectures with parameters that ensure faster optimisation. However, our task seem quite unique and even models trained on remote sensing images may not o the trick.

**Augmentation**: We can perform image augmentation to increase the size of the training set and make it less sensitive to differences in colours, position, orientation, scale among others. This is done by apply random rotation, scale, and flipping transformations. A good image augmentation library is https://github.com/aleju/imgaug.

## Prediction

The algorithm outputs for every observation, a vector of probability associated with every class.


## Computations

**GPU computing**: There happens to be a really good GPU on my work computer (Radeon Pro 580X 8GB). However, deep learning support for AMD is not as developed as for NVDIA through the CUDA library. A solution was found using OpenCL computing framework with the PaidML compatibility framework to work with Keras Python library using a TensorFlow backend."OpenCL-enabled GPUs, such as those from AMD, via the PlaidML Keras backend" A convolutional neural network for digit recognition was successfully trained in 156 seconds on 10 000 images with 99.6% accuracy.

**Architecture**: Deep learning is a very prolific field of research, and there are hundreds of existing segmentation architectures, let alone the custom architectures. It is probably best to rely on existing, general-purpose architectures such as UNet. These models are used to process images much more complex than those we have (traffic, remote sensing, etc.). This restricts the search space for the best estimator, and might give access to pre-trained models for faster convergence. 

**Environment**: GDAL, Numpy, Keras, TensorFlow, OpenCV

## Parameters

- Architecture

- Input size
- Input channels

- Batch size

- Loss function 
- Regularisation: parameter and type

- Training sample size
- Test sample size
- Quality metric

- Convolution layers:
  - Kernel size
  - Padding
  - Stride
- Activation layers: 
  - Function: ReLU
- Pooling layers: 
  - Function: Max
  - Extent: 2x2