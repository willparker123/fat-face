#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:42:30 2019

@author: A897WD
"""

from network.network import Network
import network.mnist_loader as mnist_loader
import pickle
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from cvxopt import matrix, solvers
from all_object import CFGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_classification
from sklearn.neural_network import MLPClassifier as NN
import matplotlib.pyplot as plt

import matplotlib.patheffects as path_effects
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF
#from kde import plot_decision_boundary

def get_data():  
    t = 2
    t1 = 10 * np.random.random_sample(100 * t) - 0.50
    t0 = np.random.normal(0, 0.40, 100 * t)
    x1 = np.vstack((t0, t1)).T
    y1 = np.ones(100 * t)
    
    t1 = 6 * np.random.random_sample(100 * t) - 0.50
    t0 = np.random.normal(0, 0.50, 100 * t)
    x2 = np.vstack((t1, t0)).T
    y2 = np.zeros(100 * t)
    
    mean = [3.50, 8.00]
    sigma = 0.50
    cov = sigma * np.identity(2)
    n3 = 50 * t
    x3 = np.random.multivariate_normal(mean, cov, n3)
    y3 = np.zeros(n3)
    
    X = np.concatenate((x1, x2, x3), axis=0)
    y = np.concatenate((y1, y2, y3))
    
    return X, y

np.random.seed(123)

X, y= get_data()
training_data = []
for i in range(X.shape[0]):
    training_data.append((X[i,], y[i]))
net = Network([2, 3, 1])
net.SGD(training_data, 50, 50, 0.01, test_data=training_data)