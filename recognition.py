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
        None
        
    def fit(self, train_path='./train'):
        imgs_names = [join(train_path, f) for f in listdir(train_path) if isfile(join(train_path, f))]
        imgs_names_full = []
        imgs = []
        mean = np.zeros(plt.imread(imgs_names[0]).shape[:2])
        for name in imgs_names:
            img = rgb2gray(plt.imread(name))
            if img.shape == mean.shape:
                imgs.append(img)
                imgs_names_full.append(name)
                mean += img / len(imgs_names)
        for ind, img in enumerate(imgs):
            (img - mean).flatten()
            imgs[ind] = (img - mean).flatten()
        imgs = np.array(imgs).T
        imgs = da.from_array(imgs)
        U, S, Vt = da.linalg.svd(imgs)
        
        imgs = np.array(imgs)
        self.imgs = imgs
        Vt = np.array(Vt)
        self.U = np.array(U)
        self.X = self.U.T @ imgs
        self.mean = mean
        self.names = imgs_names_full
        self.N = len(imgs_names_full)
        
    def predict(self, img):
        if len(img.shape) == 3:
            img = rgb2gray(img)
            
        img = (img - self.mean).flatten()
        img_vec = self.U.T @ img
        img_vec_restored = self.U @ img_vec
        eps_test = sqrt((img - img_vec_restored) @ (img - img_vec_restored))
        epses = np.zeros(self.N)
        for i in range(self.N):
            epses[i] = sqrt((self.X.T[i] - img_vec)@(self.X.T[i] - img_vec))
        return (epses[np.argmin(epses)], self.names[np.argmin(epses)])

