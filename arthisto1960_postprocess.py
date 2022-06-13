#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Post-processing for the Arthisto1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
'''

#%% HEADER

# Modules
import numpy as np

from arthisto1960_utilities import *
from skimage import morphology, segmentation
from os import path

# Samples
training = identifiers(search_data(paths['labels']), regex = True)
cities   = '(0625_6870|0650_6870|0875_6245|0875_6270|0825_6520|0825_6545|0550_6295|0575_6295).tif$'

#%% COMPUTES LABELS

files = search_data(paths['predictions'], pattern='proba.*tif$')
files = filter_identifiers(files, search_data(paths['predictions'], pattern='label.*tif$'))

for i, file in enumerate(files):
    print('{file} {index:3d}/{total:3d}'.format(file=path.basename(file), index=i + 1, total=len(files)))
    os.system('gdal_calc.py --overwrite -A {proba} --outfile={label} --calc="A>=0.5" --type=Byte --quiet'.format(proba=file, label=file.replace('proba', 'label')))
del files, i, file

#%% AGGREGATES RASTERS

args = dict(
    pattern = path.join(paths['predictions'], 'label*.tif'),
    vrtfile = path.join(paths['data'], 'buildings1960.vrt'),
    outfile = path.join(paths['data'], 'buildings1960.tif'),
    reffile = '../data_project/ca.tif'
)

# Extracts extent
# reference = rasterio.open(args['reffile'])
# reference.bounds
# reference.nodatavals
# del reference

os.system('gdalbuildvrt -overwrite {vrtfile} {pattern}'.format(**args))
os.system('gdalwarp -overwrite {vrtfile} {outfile} -t_srs EPSG:3035 -te 3210400 2166600 4191800 3134800 -tr 200 200 -r average -ot Float32'.format(**args))
os.remove(args['vrtfile'])

# Masks non-buildable
os.system('gdal_calc.py --overwrite -A {outfile} -B {reffile} --outfile={outfile} --calc="(A*(B!=0))-(B==0)" --NoDataValue=-1 --type=Float32 --quiet'.format(**args))
del args

#%% COMPUTES VECTORS    

# Vectorises individual tiles
files = search_data(paths['predictions'], pattern=f'label_{training}') + search_data(paths['predictions'], pattern=f'label_{cities}')

for i, file in enumerate(files):
    print('{file} {index:2d}/{total:2d}'.format(file=path.basename(file), index=i + 1, total=len(files)))
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

#%% Display results
pattern = '({ids}).tif$'.format(ids='|'.join(cities.values()))
[os.system('open {}'.format(file)) for file in search_data(paths['images'], pattern=f'image_{training}')]
[os.system('open {}'.format(file)) for file in search_data(paths['predictions'], pattern=f'label_{training}')]

