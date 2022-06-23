# Semantic segmentation of historical maps

Clément Gorin, gorinclem@gmail.com

Extracting buildings depicted in historical maps provides novel insights for urban and spatial economics. Convolutional networks provide state-of-the art performance in various image processing tasks including semantic segmentation. A U-Net architecture is implemented to extract features automatically from a collection of 20th century maps covering mainland France. The model achieves more than 95% precision and recall on a test sample. This approach is efficient, scalable and readily applicable to various collections of historical maps with minimal manual labelling.

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

The *SCAN50 historique* collection of maps covers mainland France and Corse at the end of the 1950’s. The database consists of 1023 raster tiles covering an area of 25 km2 with a 5 x 5 m resolution. This collection is a patchwork of five different map types, with a varying number of representations and colours. The rasters can be downloaded form the [website](https://geoservices.ign.fr/scanhisto) of the French National Geographical Institute (IGN). 

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

Model 


**Structure**: The U-Net architecture for semantic segmentation is fully convolutional. The parameters are updated by gradient descent. Contraction path, expansion path

Every convolutional block contains two convolution, activation and batch-normalisation layers. Spatial dropout is implemented for further regularisation.

<img src='figures/fig_unet.pdf'>

**Optimisation**: The model parameters are updated iteratively using a variant of the gradient descent algorithm (i.e. Adam), which minimises a logistic loss function (i.e. inverse log-likelihood of the binomial distribution) that accounts for the class imbalance in the data. To prevent overfitting, a validation sample is used to assess the generalisation performance after each optimisation iteration. Each iteration uses a batch of 32 images to avoid local minima and saddle-points.



## Results



