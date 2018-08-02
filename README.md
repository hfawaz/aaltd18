# Data augmentation using synthetic data for time series classification with deep residual networks
This is the companion repository for our paper titled "Data augmentation using synthetic data for time series classification with deep residual networks".
This paper has been accepted for an oral presentation at the [Workshop on Advanced Analytics and Learning on Temporal Data (AALTD) 2018](https://project.inria.fr/aaldt18/) in the [European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML/PKDD) 2018](http://www.ecmlpkdd2018.org/).

## Data
The data used in this project comes from the [UCR archive](http://www.cs.ucr.edu/~eamonn/time_series_data/), which contains the 85 univariate time series datasets we used in our experiements. 

## Code
The code is divided as follows: the distance folder contains the DTW distance in Cython instead of pure python in order to reduce the running time. 
The dba.py file contains the DBA algorithm. 
The utils/ folder contains the necessary functions to read the datasets and visualize the plots. 
The knn.py file contains the K nearest neighbor algorithm which is mainly used when computing the weights for the data augmentation technique. 
The resnet.py file contians the keras and tesnorflow code to define the architecture and train the deep learning model. 
The augment.py file contains the method that generates the random weights (Average Selected) with a function that does the actual augmentation for a given training set of time series.  

## Prerequisites
All python packages needed are listed in utils/pip-requirements.txt file and can be installed simply using the pip command. 

## Results
The main contribution of a data augmentation technique is to improve the performance (accuracy) of a deep learning model especially for time series datasets with small training sets such as the DiatomSizeReduction (the smallest in the UCR archive) where we managed to increase the model's accuracy from 30% (without data augmentation) to 96% with data augmentation for a residual network architecture. 

