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
from os import path

#%% COMPUTES LABELS

files = search_files(paths['predictions'], pattern='proba.*tif$')
for i, file in enumerate(files):
    print('{file} ({index:04d}/1023)'.format(file=path.basename(file), index=i + 1))
    os.system('gdal_calc.py --overwrite -A {proba} --outfile={label} --calc="A>=0.5" --NoDataValue=0 --type=Byte --quiet'.format(proba=file, label=file.replace('proba', 'label')))
del files, i, file

#%% COMPUTES VECTORS    

pattern = '({ids}).tif'.format(ids='|'.join(cities.values()))
files   = search_files(paths['predictions'], pattern=f'label_{pattern}')

# Computes vectors
for i, file in enumerate(files):
    print('{file} ({index:01d}/{total:01d})'.format(file=path.basename(file), index=i + 1, total=len(files)))
    os.system('gdal_edit.py -a_nodata 0 {raster}'.format(raster=file))
    os.system('gdal_polygonize.py {raster} {vector} -q'.format(raster=file, vector=file.replace('tif', 'gpkg')))
    os.system('gdal_edit.py -unsetnodata {raster}'.format(raster=file))
del files, file, i

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

# Masks non-buildable
os.system('gdal_calc.py --overwrite -A {outfile} -B {reffile} --outfile={outfile} --calc="(A*(B!=0))-(B==0)" --NoDataValue=-1 --type=Float32 --quiet'.format(**args))
del args

#%% Display results
pattern = '({ids}).tif'.format(ids='|'.join(cities.values()))

for file in search_files(paths['images'], pattern=f'image_{pattern}'):
    os.system('open {}'.format(file))

for file in search_files(paths['predictions'], pattern=f'label_{pattern}'):
    os.system('open {}'.format(file))