#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Commonly used in anomaly detection algorithms.
"""

__author__ = "Brandon Doyle"
__email__ = "bjd2385@aperiodicity.com"

from math import sqrt
import numpy as np

class dM(object):
  
    """ compute the Mahalanobis distance to every pixel in an array """

    def __init__(self, array):
        self.array = array
        self.Xpix = array.shape[0]
        self.Ypix = array.shape[1]
 
    def Mahalanobis(self):
        
        """ Compute the Mahalanobis distance to every point in RGB 
        space within the image """
        
        mean_vector = np.mean(self.array, axis=(0, 1))
        variance_covariance = self.__variance_covariance_()
 
        # iterate over the original image and store dM in this new array
        distances = np.zeros([self.Xpix, self.Ypix])
        for i in range(self.Xpix):
            for j in range(self.Ypix):
                distances[i][j] = sqrt(np.dot(np.dot(np.transpose(\
                self.array[i][j] - mean_vector), variance_covariance), \
                self.array[i][j] - mean_vector))
        
        return distances
 
    def __variance_covariance_(self, number=10000):
    
        """ create a variance-covariance matrix by randomly sampling the
        RGB space """
    
        reshaped_array = self.array.reshape((self.Xpix * self.Ypix, 3))
        
        # sample the above array to reduce computing time
        choices = np.random.randint(0, len(reshaped_array), number)
        reshaped_array = np.array([reshaped_array[i] for i in choices])
        average = np.mean(reshaped_array)
 
        # compute the variance-covariance matrix for these RGB data
        matrix = np.array(sum([np.outer(np.array([reshaped_array[i] - \
           average]), np.array(reshaped_array[i] - average)) for i in \
            range(len(reshaped_array))]) / len(reshaped_array))
 
        return np.linalg.inv(matrix)
