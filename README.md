# SVD-faces-recognition
This repo provides faces detection software based on SVD decomposition.


REQUIREMENTS:
python-3 
dask
skimage

Tested on:
Numpy 1.22.4
Python 3.8.8
Dask 2021.04.0
Skimage 0.18.1
Matplotlib 3.4.3

USAGE:
# Importing the predictor
from recognition import SVDrecognition
# Creating the predictor
rec = SVDrecognition()
# Fitting the predictor 
# Train_path - path to folder with train pictures
# By default - './train'
rec.fit(train_path)
# Predicting the result for new image:
img = matplotlib.pyplot.imread(test_image_path)
rec.predict(img)
