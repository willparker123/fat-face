# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 12:14:14 2019

@author: rp13102
"""
import numpy as np

# been kim's proxy
class DensityEstimator(object):
    def __init__(self, 
                 distance_function = None,
                 transformation_function = None):
# =============================================================================
#         if k is None: 
#             self.k = 10 
#         else: 
#             self.k = k
#             
# =============================================================================
        if distance_function is None: 
            self.distance_function = lambda x, y: np.linalg.norm(x.reshape(-1, 1) - y.reshape(-1, 1))
        else:
            self.distance_function = distance_function
            
        if transformation_function is None: 
            self.transformation_function = lambda x: -np.log(x)
        else:
            self.transformation_function = transformation_function
            
    def fit(self, X):
        self.X = X
        self.n_samples, _ = X.shape
       
    def score_samples_single(self, xtest):
        distances = np.zeros(self.n_samples)
        for idx in range(self.n_samples):
            distances[idx] = self.distance_function(xtest, self.X[idx, :])
        return self.transformation_function(np.sort(distances)[self.k])
    
    def score_samples(self, xtest, k):
        n_samples_test = xtest.shape[0]
        self.k = k
        if n_samples_test == 1:
            return self.score_samples_single(xtest)
        else:
            scores = np.zeros((n_samples_test, 1))
            for idx in range(n_samples_test):
                scores[idx] = self.score_samples_single(xtest[idx, :])
            return scores