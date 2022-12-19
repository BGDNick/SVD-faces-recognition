#!/usr/bin/env python
# coding: utf-8

# In[423]:


from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import dask.array as da
from math import sqrt


# In[428]:


class SVDrecognition():
    def __init__(self):
        '''
        Init of the class do not pass anything here 
        '''
        None
        
    def fit(self, train_path='./train', num_vecs=10):
        '''
        Calculates SVD decomposition of face images
        
        Parameters
        -----------
        train_path - path to folder containing images for train
        num_vecs - amount of first n vectors from U, that are used for the projection
        '''
        # Getting all images names from train folder
        imgs_names = [join(train_path, f) for f in listdir(train_path) if isfile(join(train_path, f))]
        imgs_names_full = []
        imgs = []
        # Calculating mean image from all images
        mean = np.zeros(plt.imread(imgs_names[0]).shape[:2])
        for name in imgs_names:
            img = rgb2gray(plt.imread(name))
            # Skip images that have different shape from mean image
            if img.shape == mean.shape:
                imgs.append(img)
                imgs_names_full.append(name)
                mean += img / len(imgs_names)
        # Substraction of mean image from train images
        for ind, img in enumerate(imgs):
            (img - mean).flatten()
            imgs[ind] = (img - mean).flatten()
        imgs = np.array(imgs).T
        imgs = da.from_array(imgs)
        # Calculation of SVD decomposition
        U, S, Vt = da.linalg.svd(imgs)
        
        imgs = np.array(imgs)
        self.imgs = imgs
        Vt = np.array(Vt)
        self.U = np.array(U)[:,:num_vecs]
        # Calculation of projection of all images on first vectors of U
        self.X = self.U.T @ imgs
        self.mean = mean
        self.names = imgs_names_full
        self.N = len(imgs_names_full)
        
    def predict(self, img, return_all=False, return_test=False):
        '''
        Finds closest projection of new face to faces obtained from training using first num_vectors from U
        
        Parameters
        -----------
        img - numpy array of image either in format (M,N) or (M,N,3)
        return_all - if True returns full epsilon vector
        '''
        # Convertion image to grayscale if it is colorized
        if len(img.shape) == 3:
            img = rgb2gray(img)
        # Substraction of mean train image from prediction image
        img = (img - self.mean).flatten()
        # Calculation of projection of the image on first vectors of U
        img_vec = self.U.T @ img
        # Calculation of restored from projection
        img_vec_restored = self.U @ img_vec
        # Calculation of test epsilon, which indicates how restored image derives from initial one
        eps_test = sqrt((img - img_vec_restored) @ (img - img_vec_restored))
        epses = np.zeros(self.N)
        # Calculation of norm between test image and train images projections
        for i in range(self.N):
            epses[i] = sqrt((self.X.T[i] - img_vec)@(self.X.T[i] - img_vec))
        if return_all:
            if return_test:
                return (epses, eps_test, self.names[np.argmin(epses)])
            else:
                return (epses, self.names[np.argmin(epses)])
        if return_test:
            return (epses[np.argmin(epses)], eps_test, self.names[np.argmin(epses)])
        else:
            return (epses[np.argmin(epses)], self.names[np.argmin(epses)])
    
    def get_U(self):
        return self.U.copy()
    
    def get_X(self):
        return self.X.copy()
    
    def get_names(self):
        return self.names.copy()
