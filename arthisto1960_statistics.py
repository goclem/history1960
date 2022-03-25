#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Computes statistics statistics for the Arthisto 1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.03.23
'''

#%% HEADER

# Modules
import numpy as np

from arthisto1960_utilities import *
from numpy import random
from pandas import DataFrame
from skimage import segmentation
from os import path

# Paths
paths = dict(
    data='../data_1960',
    images='../data_1960/images',
    labels='../data_1960/labels',
    models='../data_1960/models',
    predictions='../data_1960/predictions',
    statistics='../data_1960/statistics'
)

# Samples
training    = '(0250_6745|0350_6695|0400_6445|0550_6295|0575_6295|0700_6520|0700_6545|0700_7070|0875_6245|0875_6270|0900_6245|0900_6270|0900_6470|1025_6320).tif$'
legend_1900 = '(0600_6895|0625_6895|0600_6870|0625_6870|0625_6845|0600_6845|0650_6895|0650_6870|0650_6845|0675_6895|0675_6870|0675_6845|0850_6545|0825_6545|0850_6520|0825_6520|0825_6495).tif$'
legend_N    = '(0400_6570|0425_6570|0400_6595|0425_6595|0425_6545|0400_6545|0425_6520|0400_6520|0425_6395|0425_6420|0400_6395|0400_6420|0425_6720|0450_6720|0425_6745|0450_6745|0450_6695|0425_6695|0425_6670|0450_6670|0450_6570|0450_6595|0450_6545|0450_6520|0450_6945|0450_6920|0475_6920|0475_6795|0500_6795|0475_6770|0500_6770|0500_6720|0475_6720|0475_6695|0500_6695|0475_6670|0450_6645|0475_6645|0500_6645|0525_6670|0500_6670|0525_6645|0500_6620|0525_6620|0475_6620|0550_6820|0525_6820|0550_6895|0575_6895|0550_6870|0575_6870|0575_6845|0550_6845|0550_6670|0575_6670|0550_6695|0575_6695|0575_6645|0550_6645|0475_6495|0450_6495|0475_6470|0450_6470|0450_6420|0450_6395|0475_6420|0475_6395|0475_6320|0500_6320|0525_6495|0500_6495|0500_6520|0525_6520|0525_6320|0525_6345|0500_6345|0600_6670|0600_6695|0600_6645|0625_6495|0650_6495|0650_6520|0625_6520|0725_6320|0700_6320|0725_6345|0700_6345|0775_6420|0750_6420|0725_6420|0775_6445|0750_6445|0725_6445|0775_6395|0725_6395|0750_6395|0775_6370|0800_6370|0775_6345|0800_6345|1150_6170|1150_6145|1150_6120|1175_6195|1150_6195|1175_6170|1175_6145|1175_6120|1175_6095|1150_6095|1200_6095|1175_6070|1200_6070|1200_6220|1200_6195|1175_6220|1200_6170|1200_6145|1225_6170|1225_6145|1200_6120|1225_6120|1225_6095|1250_6120|1250_6145).tif$'

#%% FUNCTIONS

# Computes prediction sets
def compute_sets(label_test:np.ndarray, label_pred:np.ndarray) -> np.ndarray:
    # Formats labels
    label_test = label_test.astype(bool)
    label_pred = label_pred.astype(bool)
    # Computes masks
    pixset_tp = np.logical_and(label_test, label_pred)
    pixset_tn = np.logical_and(np.invert(label_test), np.invert(label_pred))
    pixset_fp = np.logical_and(np.invert(label_test), label_pred)
    pixset_fn = np.logical_and(label_test, np.invert(label_pred))
    pixsets   = np.array([pixset_tp, pixset_tn, pixset_fp, pixset_fn])
    return pixsets

# Computes subset without borders
def mask_borders(sets:np.ndarray, label_test:np.ndarray) -> np.ndarray:
    subset = np.invert(segmentation.find_boundaries(label_test))
    subset = np.tile(subset, (sets.shape[0], 1, 1, 1))
    masked = np.where(subset, sets, False)
    return masked 

# Computes prediction statistics
def compute_statistics(sets:np.ndarray):
    tp, tn, fp, fn = np.sum(sets, axis=(1, 2, 3))
    with np.errstate(divide='ignore', invalid='ignore'): # Returns Inf when dividing by 0
        recall    = np.divide(tp, (tp + fn)) # Among the building pixels, {recall}% are classified as building
        precision = np.divide(tp, (tp + fp)) # Among the pixels classified as buildings, {precision}% are in fact buildings
        accuracy  = np.divide((tp + tn), (tp + tn + fp + fn))
    statistics = dict(recall=recall, precision=precision, accuracy=accuracy)
    return statistics

# Displays prediction masks
def display_statistics(image:np.ndarray, sets:np.ndarray, colour=(255, 255, 0)) -> None:
        # Formats titles
        counts = np.sum(sets, axis=(1, 2, 3))
        titles = ['True positive ({:d})', 'True negative ({:d})', 'False positive ({:d})', 'False negative ({:d})']
        titles = list(map(lambda title, count: title.format(count), titles, counts))
        # Formats images
        image  = (image * 255).astype(int)
        images = [np.where(np.tile(mask, (1, 1, 3)), colour, image) for mask in sets]
        # Displays images
        fig, axs = pyplot.subplots(2, 2, figsize=(10, 10))
        for image, title, ax in zip(images, titles, axs.ravel()):
            ax.imshow(image)
            ax.set_title(title, fontsize=20)
            ax.axis('off')
        pyplot.tight_layout(pad=2.0)
        pyplot.show()

#%% COMPUTES LABELS

files = search_files(paths['predictions'], pattern='proba.*tif$')
for i, file in enumerate(files):
    print('{file} ({index:04d}/1023)'.format(file=path.basename(file), index=i + 1))
    os.system('gdal_calc.py --overwrite -A {proba} --outfile={label} --calc="A>=0.5" --NoDataValue=0 --type=Byte --quiet'.format(proba=file, label=file.replace('proba', 'label')))
del files, i, file

#%% COMPUTES PREDICTION STATISTICS

# Loads data
labels_test = np.load(path.join(paths['statistics'], 'labels_test.npy'))
labels_pred = np.load(path.join(paths['statistics'], 'labels_pred.npy'))
images_test = np.load(path.join(paths['statistics'], 'images_test.npy'))
probas_pred = np.load(path.join(paths['statistics'], 'probas_pred.npy'))

# Subsets tiles where there are labels
subset = np.logical_and(
    np.sum(labels_test, axis=(1, 2, 3)) > 0, 
    np.sum(labels_pred, axis=(1, 2, 3)) > 0)

labels_test = labels_test[subset]
labels_pred = labels_pred[subset]
images_test = images_test[subset]
probas_pred = probas_pred[subset]
del subset

# Compute sets and removes border
pixsets = np.array(list(map(compute_sets, labels_test, labels_pred)))
pixsets = np.array(list(map(mask_borders, pixsets, labels_test)))

# Aggregated statistics
stats = np.sum(pixsets, axis=0) # Or np.add.reduce(pixsets)
stats = compute_statistics(stats)

# Statistics per tile
stats = list(map(compute_statistics, pixsets))
stats = DataFrame.from_dict(stats)

# Display statistics
subset = stats.sort_values(by='precision', ascending=True).index[:5]
for image, pixset in zip(images_test[subset], pixsets[subset]):
    display_statistics(image, pixset)
del subset

#%% COMPUTES VECTORS    

files = search_files(paths['predictions'], pattern='label.*tif$')
for i, file in enumerate(files):
    print('{file} ({index:04d}/1023)'.format(file=path.basename(file), index=i + 1))
    os.system('gdal_polygonize.py {raster} {vector} -q'.format(raster=file, vector=file.replace('tif', 'gpkg')))
del files, i, file

#%% AGGREGATES VECTORS

args = dict(
    pattern=path.join(paths['predictions'], '*.gpkg'),
    outfile=path.join(paths['data'], 'buildings1960.gpkg')
)
os.system('ogrmerge.py -single -overwrite_ds -f GPKG -o {outfile} {pattern}'.format(**args))
os.system('find {directory} -name "*.gpkg" -type f -delete'.format(directory=paths['predictions']))
del args

#%% AGGREGATES RASTERS

# Removes no data values for aggregation
files = search_files(paths['predictions'], pattern='label.*tif$')
for i, file in enumerate(files):
    print('{file} ({index:04d}/1023)'.format(file=path.basename(file), index=i + 1))
    os.system('gdal_edit.py -unsetnodata {raster}'.format(raster=file))
    # os.system('gdal_edit.py -a_nodata 0 {raster}'.format(raster=file)) # Sets nodata to 0
del files, i, file

# Extracts extent
# reference = rasterio.open('../data_project/ca.tif')
# reference.bounds
# reference.nodatavals
# del reference

args = dict(
    pattern = path.join(paths['predictions'], 'label*.tif'),
    vrtfile = path.join(paths['data'], 'buildings1960.vrt'),
    outfile = path.join(paths['data'], 'buildings1960.tif'),
    reffile = '../data_project/ca.tif'
)

os.system('gdalbuildvrt -overwrite {vrtfile} {pattern}'.format(**args))
os.system('gdalwarp -overwrite {vrtfile} {outfile} -t_srs EPSG:3035 -te 3210400 2166600 4191800 3134800 -tr 200 200 -r average -ot Float32'.format(**args))
os.remove(args['vrtfile'])
# os.system('find {directory} -name "label*.tif" -type f -delete'.format(directory=paths['predictions']))
del args

for file in search_files(paths['predictions'], pattern=f'proba_{training}'):
    os.system('open {}'.format(file))

os.system('gdal_calc.py --overwrite -A {outfile} -B {reffile} --outfile={outfile} --calc="(A*(B!=0))-(B==0)" --NoDataValue=-1 --type=Float32 --quiet'.format(**args))