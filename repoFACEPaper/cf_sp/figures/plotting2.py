# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 19:27:30 2019

@author: rp13102
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as NN
import matplotlib.patheffects as path_effects
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF
from kde import plot_decision_boundary

def get_data():  
    t1 = 10 * np.random.random_sample(100) - 0.50
    t0 = np.random.normal(0, 0.40, 100)
    x1 = np.vstack((t0, t1)).T
    y1 = np.ones(100)
    
    t1 = 6 * np.random.random_sample(100) - 0.50
    t0 = np.random.normal(0, 0.50, 100)
    x2 = np.vstack((t1, t0)).T
    y2 = np.zeros(100)
    
    mean = [3.50, 8.00]
    sigma = 0.50
    cov = sigma * np.identity(2)
    n3 = 50
    x3 = np.random.multivariate_normal(mean, cov, n3)
    y3 = np.zeros(n3)
    
    X = np.concatenate((x1, x2, x3), axis=0)
    y = np.concatenate((y1, y2, y3))
    
    return X, y

np.random.seed(123)

X, y= get_data()
X = np.delete(X, 230, axis=0)
y = np.delete(y, 230)
#mdl = NN(max_iter=1000)
mdl = GPC(1.0 * RBF(1.0))
mdl.fit(X, y)

fig, ax = plt.subplots()
plot_decision_boundary(X, y, ax, mdl, title='Neural Network Classification')

subject = np.array([0.20, 7.25])
clss = 0

ax.scatter(subject[0], subject[1], 
           c='lightgreen',
           edgecolor='k',
           s=200,
           marker='x',
           alpha=0.75,
           zorder=2)

# =============================================================================
# target0 = np.array([1.50, 6.57])
# cap = 'A'
# 
# ax.arrow(subject[0], subject[1], 
#          (target0[0] - subject[0]), (target0[1] - subject[1]), 
#          head_width=0.2, 
#          head_length=0.1, 
#          width=0.02,
#          fc='k', 
#          ec='k')
# text = plt.text(target0[0] + 0.2, 
#          target0[1] - 0.1, 
#          'A',
#          color='lightgreen',
#          weight='bold',
#          fontsize=15)
# text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
#                        path_effects.Normal()]) 
#  
# cap = 'B'
# target0b = subject + 2*(target0 - subject)
# ax.arrow(subject[0], subject[1], 
#          (target0b[0] - subject[0]), (target0b[1] - subject[1]), 
#          head_width=0.2, 
#          head_length=0.1, 
#          width=0.02,
#          fc='k', 
#          ec='k')
# 
# text = plt.text(target0b[0] + 0.2, 
#          target0b[1] - 0.1, 
#          'B',
#          color='lightgreen',
#          weight='bold',
#          fontsize=15)
# text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
#                        path_effects.Normal()]) 
#  
# 
# target1 = np.array([1.43, -0.12])
# cap = 'D'
# 
# ax.arrow(subject[0], subject[1], 
#          (target1[0] - subject[0]), (target1[1] - subject[1]), 
#          head_width=0.2, 
#          head_length=0.1, 
#          width=0.02,
#          fc='k', 
#          ec='k')
# 
# text = plt.text(target1[0] + 0.2, 
#          target1[1] - 0.1, 
#          'D',
#          color='lightgreen',
#          weight='bold',
#          fontsize=15)
# text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
#                        path_effects.Normal()])
# 
# target2 = np.array([3, 7.80])
# cap = 'C'
# 
# ax.arrow(subject[0], subject[1], 
#          (target2[0] - subject[0]), (target2[1] - subject[1]), 
#          head_width=0.2, 
#          head_length=0.1, 
#          width=0.02,
#          fc='k', 
#          ec='k')
# 
# text = plt.text(target2[0] + 0.2, 
#          target2[1] - 0.1, 
#          'C',
#          color='lightgreen',
#          weight='bold',
#          fontsize=15)
# text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
#                        path_effects.Normal()])
# =============================================================================
