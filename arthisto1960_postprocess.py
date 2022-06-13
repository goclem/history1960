#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Post-processing for the Arthisto1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.05.30
'''

#%% HEADER

# Modules
import numpy as np

from arthisto1960_utilities import *
from skimage import morphology, segmentation
from os import path

# Samples
training = identifiers(search_data(paths['labels']), regex = True)
cities   = dict(paris='0625_6870|0650_6870', marseille='0875_6245|0875_6270', lyon='0825_6520|0825_6545', toulouse='0550_6295|0575_6295')

#%% COMPUTES LABELS

files = search_data(paths['predictions'], pattern='proba.*tif$')
for i, file in enumerate(files):
    print('{file} ({index:04d}/{total:04d})'.format(file=path.basename(file), index=i + 1, total=len(files)))
    os.system('gdal_calc.py --overwrite -A {proba} --outfile={label} --calc="A>=0.5" --type=Byte --quiet'.format(proba=file, label=file.replace('proba', 'label')))
del files, i, file

#%% COMPUTES VECTORS    

# pattern = '({ids}).tif'.format(ids='|'.join(cities.values()))
files = search_data(paths['predictions'], pattern=f'label_{training}')

# Computes vectors
for i, file in enumerate(files):
    print('{file} ({index:02d}/{total:02d})'.format(file=path.basename(file), index=i + 1, total=len(files)))
    os.system('gdal_edit.py -a_nodata 0 {raster}'.format(raster=file))
    os.system('gdal_polygonize.py {raster} {vector} -q'.format(raster=file, vector=file.replace('tif', 'gpkg')))
    os.system('gdal_edit.py -unsetnodata {raster}'.format(raster=file))
del files, i, file

# Aggregates vectors
args = dict(
    pattern=path.join(paths['predictions'], '*.gpkg'),
    outfile=path.join(paths['data'], 'cities1960.gpkg')
)
os.system('ogrmerge.py -single -overwrite_ds -f GPKG -o {outfile} {pattern}'.format(**args))
os.system('find {directory} -name "*.gpkg" -type f -delete'.format(directory=paths['predictions']))
del args

#%% AGGREGATES RASTERS

# Removes no data values for aggregation
files = search_data(paths['predictions'], pattern='label.*tif$')
for i, file in enumerate(files):
    print('{file} ({index:04d}/1023)'.format(file=path.basename(file), index=i + 1))
    os.system('gdal_edit.py -unsetnodata {raster}'.format(raster=file))
    # os.system('gdal_edit.py -a_nodata 0 {raster}'.format(raster=file)) # Sets nodata to 0
del files, i, file

args = dict(
    pattern = path.join(paths['predictions'], 'label*.tif'),
    vrtfile = path.join(paths['data'], 'buildings1960.vrt'),
    outfile = path.join(paths['data'], 'buildings1960.tif'),
    reffile = '../data_project/ca.tif'
)

# Extracts extent
# reference = rasterio.open(args['reffile])
# reference.bounds
# reference.nodatavals
# del reference

os.system('gdalbuildvrt -overwrite {vrtfile} {pattern}'.format(**args))
os.system('gdalwarp -overwrite {vrtfile} {outfile} -t_srs EPSG:3035 -te 3210400 2166600 4191800 3134800 -tr 200 200 -r average -ot Float32'.format(**args))
os.remove(args['vrtfile'])
# os.system('find {directory} -name "label*.tif" -type f -delete'.format(directory=paths['predictions']))

# Masks non-buildable
os.system('gdal_calc.py --overwrite -A {outfile} -B {reffile} --outfile={outfile} --calc="(A*(B!=0))-(B==0)" --NoDataValue=-1 --type=Float32 --quiet'.format(**args))
del args

#%% Display results
pattern = '({ids}).tif$'.format(ids='|'.join(cities.values()))
[os.system('open {}'.format(file)) for file in search_data(paths['images'], pattern=f'image_{training}')]
[os.system('open {}'.format(file)) for file in search_data(paths['predictions'], pattern=f'label_{training}')]

