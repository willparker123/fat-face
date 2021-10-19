# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 14:15:14 2019

@author: rp13102
"""

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut
import matplotlib.pyplot as plt
import matplotlib as mpl

mean = np.array([0, 0])
cov = np.array([[1, 0], [0, 1]])
N = 100

X = np.random.multivariate_normal(mean, cov, N)

def plot_density_kde(X, ax):
    h = 0.1
    xmin, ymin = np.min(X, axis=0)
    xmax, ymax = np.max(X, axis=0)

    xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                         np.arange(ymin, ymax, h))

    cm = plt.cm.Blues
    cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])    
    
    newx = np.c_[xx.ravel(), yy.ravel()]
    bandwidths = 10 ** np.linspace(-2, 1, 10)
    my_cv = LeaveOneOut()

    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=my_cv,
                        iid=True)

    grid.fit(X)
    mdl = grid.best_estimator_
    Z = mdl.score_samples(newx)

    Z = Z.reshape(xx.shape)
    contour_plot = ax.contourf(xx, yy, np.exp(Z), 
                                 levels=20,
                                 cmap=cm, 
                                 alpha=.8)
    plt.colorbar(contour_plot, ax=ax)
    
# =============================================================================
#     ax.scatter(X[:, 0], X[:, 1], c=y, 
#                   cmap=cm_bright,
#                   edgecolors='k',
#                   zorder=2)
# =============================================================================
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.grid(color='k', linestyle='-', linewidth=0.50, alpha=0.75)

    #plt.xticks(())
    #plt.yticks(())
    ax.set_title('Shortest Path - Density Based Distances (DBD)')

    return mdl

def plot_path(ax):
    start = np.array([2, 2])
    end = np.array([2, -2])
    
    t = np.linspace(0, 1, 1000)
    x0 = ((end[0]**3 - start[0]**3) * t + start[0]**3)**(1/3)
    x1 = ((end[1]**3 - start[1]**3) * t + start[1]**3)**(1/3)
    
    ax.plot(x0, x1, color='r')
    return x0, x1
    
fig, ax = plt.subplots()
#plot_density_kde(X, ax)
plot_path(ax)