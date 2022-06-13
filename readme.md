# Semantic segmentation of historical maps

Clément Gorin, gorinclem@gmail.com

**Description:** Extracting buildings depicted in historical maps provides novel insights for urban and spatial economics. Convolutional networks provide state-of-the art performance in various image processing tasks including semantic segmentation. I implement a fully convolutional network to extract features automatically from a collection of 20th century maps covering mainland France. This approach is efficient, scalable and readily applicable to other historical maps with minimal manual labelling.


## Structure

```
./scripts
	/arthisto1960_environment.yml	/arthisto1960_utilities.py
	/arthisto1960_preprocess.py	/arthisto1960_model.py	/arthisto1960_optimise.py
	/arthisto1960_predict.py	/arthisto1960_postprocess.py	/arthisto1960_statistics.py

./data
	/images
	/labels
	/models
	/predictions
	/statistics```

## Data

The SCAN50 collection contains 1023 geocoded raster tiles representing mainland France in at the end of the 1950’s. Each tile represents an area of 25 km2 with a 5x5 m resolution. This collection is a patchwork of five different map types, with a varying number of representations and colours. The images can be downloaded form the website of the French National Geographical Institute (IGN). 

Key | Value
--- | ---
Number rasters | 1023
Extent | 5000 x 5000
Resolution | 5 x 5 metres
CRS | Lambert 93 (EPSG:2154)


The label data was created by manually vectorising 17 tiles. We thank Olena Bogdan, Célian Jounin, Siméon Mangematin, Matéo Moglia, Yoann Mollier-Loison, Rémi Pierotti and Nathan Vieira for their research assisantship. The training sample contains both urban and non-urban areas and is roughly balanced across the different legends.

<!--
<img src='figures/fig_legend.jpg' width='500' height='500'>
<img src='figures/fig_style.jpg' width='500' height='500'>
-->


## Model


<img src='figures/fig_unet.pdf'>

## Results



