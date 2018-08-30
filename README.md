# Data augmentation using synthetic data for time series classification with deep residual networks
This is the companion repository for [our paper](https://arxiv.org/abs/1808.02455) titled "Data augmentation using synthetic data for time series classification with deep residual networks".
This paper has been accepted for an oral presentation at the [Workshop on Advanced Analytics and Learning on Temporal Data (AALTD) 2018](https://project.inria.fr/aaldt18/) in the [European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML/PKDD) 2018](http://www.ecmlpkdd2018.org/).

![architecture resnet](https://github.com/hfawaz/aaltd18/blob/master/png/resnet-archi.png)

## Data
The data used in this project comes from the [UCR archive](http://www.cs.ucr.edu/~eamonn/time_series_data/), which contains the 85 univariate time series datasets we used in our experiements. 

## Code
The code is divided as follows: 
* The [distance](https://github.com/hfawaz/aaltd18/tree/master/distances/dtw) folder contains the DTW distance in Cython instead of pure python in order to reduce the running time.  
* The [dba.py](https://github.com/hfawaz/aaltd18/blob/master/dba.py) file contains the DBA algorithm.  
* The [utils](https://github.com/hfawaz/aaltd18/tree/master/utils) folder contains the necessary functions to read the datasets and visualize the plots.  
* The [knn.py](https://github.com/hfawaz/aaltd18/tree/master/knn.py) file contains the K nearest neighbor algorithm which is mainly used when computing the weights for the data augmentation technique.  
* The [resnet.py](https://github.com/hfawaz/aaltd18/tree/master/resnet.py) file contians the keras and tesnorflow code to define the architecture and train the deep learning model.  
* The [augment.py](https://github.com/hfawaz/aaltd18/tree/master/augment.py) file contains the method that generates the random weights (Average Selected) with a function that does the actual augmentation for a given training set of time series.  

## Prerequisites
All python packages needed are listed in utils/pip-requirements.txt file and can be installed simply using the pip command. 

[Cython](http://cython.org/)  
[numpy](http://www.numpy.org/)  
[pandas](https://pandas.pydata.org/)  
[sklearn](http://scikit-learn.org/stable/)  
[scipy](https://www.scipy.org/)  
[matplotlib](https://matplotlib.org/)  
[tensorflow-gpu](https://www.tensorflow.org/)  
[keras](https://keras.io/)  

## Results
The main contribution of a data augmentation technique is to improve the performance (accuracy) of a deep learning model especially for time series datasets with small training sets such as the DiatomSizeReduction (the smallest in the UCR archive) where we managed to increase the model's accuracy from 30% (without data augmentation) to 96% with data augmentation for a residual network architecture. 

Meat             |  DiatomSizeReduction
:-------------------------:|:-------------------------:
![plot-meat-dataset](https://github.com/hfawaz/aaltd18/blob/master/png/plot-meat.png)  |  ![plot-diatomsizereduction-dataset](https://github.com/hfawaz/aaltd18/blob/master/png/plot-generalization.png)

## Cite

If you re-use this work, please cite:

```
@InProceedings{IsmailFawaz2018,
  Title                    = {Data augmentation using synthetic data for time series classification with deep residual networks},
  Author                   = {Ismail Fawaz, Hassan and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
  Booktitle                = {International Workshop on Advanced Analytics and Learning on Temporal Data, {ECML} {PKDD}},
  Year                     = {2018}
}
```
