#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Prepares data for the Arthisto1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.03.18
'''

#%% MODULES
from arthisto1960_utilities import *
from os import path

paths = dict(
    images_raw='/Users/clementgorin/Dropbox/data/ign_scan50',
    labels_raw='../shared_ras/training1960',
    images='../data_1960/images', 
    labels='../data_1960/labels'
)

#%% FORMATS IMAGES

srcfiles = search_files(paths['images_raw'], 'tif$')

for srcfile in srcfiles:
    print(path.basename(srcfile))
    outfile = path.basename(srcfile).replace('sc50', 'image')
    outfile = path.join(paths['images'], outfile)
    if not path.exists(outfile):
        os.system('gdal_translate -ot byte {srcfile} {outfile}'.format(srcfile=srcfile, outfile=outfile))

#%% FORMATS LABELS

# Builds file paths
srclabels = search_files(directory=paths['labels_raw'], pattern='label_\\d{4}_\\d{4}\\.gpkg$')
srclabels = list(filter(re.compile('^(?!.*(incomplete|pending)).*').search, srclabels)) # Drops incomplete tiles
srcimages = search_files(directory=paths['labels_raw'], pattern=identifiers(srclabels, regex=True) + '.tif')
outlabels = [path.join(paths['labels'], path.basename(file).replace('.gpkg', '.tif')) for file in srclabels]

# Rasterises label vectors
for srclabel, srcimage, outlabel in zip(srclabels, srcimages, outlabels):
    rasterise(srcVecPath=srclabel, srcRstPath=srcimage, outRstPath=outlabel, burnValue=1)
del srclabels, srcimages, outlabels

training = search_files(directory=paths['labels_raw'], pattern='label_\\d{4}_\\d{4}\\.gpkg$')
identifiers(srclabels, regex=True)