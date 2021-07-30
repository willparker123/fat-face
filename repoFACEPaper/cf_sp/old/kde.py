# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 09:40:43 2019

@author: rp13102
"""
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
from functools import partial

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF

from dijsktra_algorithm import Graph, dijsktra_toall
from density_estimator import DensityEstimator


def log_inv(x):
    return -np.log(x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_volume_of_sphere(d):
    return math.pi**(d/2)/math.gamma(d/2 + 1)

def plot_decision_boundary(X, y, ax, clf, title='GPC'):
    h = 0.1
    xmin = np.min(X[:, 0])
    xmax = np.max(X[:, 0])
    
    ymin = np.min(X[:, 1])
    ymax = np.max(X[:, 1])
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                         np.arange(ymin, ymax, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])
    
    #plt.figure()
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    #ax.set_xticks(())
    #ax.yticks(())
    
    newx = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict_proba(newx)[:, 1]

    Z = Z.reshape(xx.shape)
    contour_plot = ax.contourf(xx, yy, Z, 
                               levels=20, 
                               cmap=cm, 
                               alpha=.8)

    plt.colorbar(contour_plot, ax=ax)
    ax.scatter(X[:, 0], X[:, 1], c=y, 
               cmap=cm_bright,
               edgecolors='k',
               alpha=0.50,
               zorder=1)
    ax.grid(color='k', linestyle='-', linewidth=0.50, alpha=0.75)
    ax.set_title(title)
    
def plot_path(X, path, ax):
    n_nodes = len(path)
    args = {'color': 'lightgreen',
            'marker': 'x',
            's': 100}
    
# =============================================================================
#     plt.text(X[int(path[0]), 0] + 0.1, 
#              X[int(path[0]), 1] + 0.1, 
#              'START',
#              bbox=dict(facecolor='green', alpha=0.5))
#     
# =============================================================================
# =============================================================================
#     ax.text(X[int(path[-1]), 0] + 0.1, 
#              X[int(path[-1]), 1] + 0.1, 
#              'END',
#              bbox=dict(facecolor='green', alpha=0.5))
# =============================================================================
    
    for idx in range(n_nodes-1):
        i = int(path[idx])
        j = int(path[idx + 1])
        #ax.scatter(X[i, 0], X[i, 1], **args)
        ax.plot(X[[i, j], 0], X[[i, j], 1], 'k', alpha=0.50)
        
    ax.scatter(X[j, 0], X[j, 1],
               color='k',
               marker='o',
               facecolors='lightgreen',
               edgecolors='lightgreen',
               alpha = 0.50,
               zorder=1,
               s=150)

def get_edges(kernel):
    edges = []

    n_samples = kernel.shape[0]
    for i in range(n_samples):
        for j in range(i):
            weight = kernel[i, j]
            if weight != 0 :
                edges.append([i, j, weight])
                #edges.append([str(j), str(i), weight])
    return edges


def get_weights(X, 
                density_scorer, 
                mode,
                weight_func = log_inv):
    epsilon = 0.75
    n_samples, _ = X.shape
    k = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        v0 = X[i, :].reshape(-1, 1)
        for j in range(i):
            v1 = X[j, :].reshape(-1, 1)
            dist = np.linalg.norm(v0 - v1)
            if dist <= epsilon:
                midpoint = (v0 + v1)/2
                density = density_scorer(midpoint.reshape(1, -1))
                if mode == 1:
                    k[i, j] = weight_func(np.exp(density)) * dist
                else:
                    k[i, j] = weight_func(sigmoid(density)) * dist
            else:
                k[i, j] = 0
            k[j, i] = k[i, j]
    return k

def get_weights_kNN(X, 
                 n_neighbours = 5,
                 weight_func = log_inv):
    n_samples, n_ftrs = X.shape
    volume_sphere = get_volume_of_sphere(n_ftrs)
    const = (n_neighbours / (n_samples * volume_sphere))**(1/n_ftrs)
    
    k = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        v0 = X[i, :].reshape(-1, 1)
        for j in range(n_samples):
            v1 = X[j, :].reshape(-1, 1)
            dist = np.linalg.norm(v0 - v1)
            k[i, j] = dist
        t = np.argsort(k[i, :])[(n_neighbours+1):]
        mask = np.ix_(t)
        k[i, mask] = 0
        
    for i in range(n_samples):
        v0 = X[i, :].reshape(-1, 1)
        for j in range(n_samples):
            v1 = X[j, :].reshape(-1, 1)
            if k[i, j] != 0:
                current_value = k[i, j]
                k[i, j] = current_value * weight_func(const / (current_value**n_ftrs))
    return k

def get_weights_e(X, 
                 epsilon = 0.75,
                 weight_func = log_inv):
    n_samples, n_ftrs = X.shape
    
    k = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        v0 = X[i, :].reshape(-1, 1)
        for j in range(i):
            v1 = X[j, :].reshape(-1, 1)
            dist = np.linalg.norm(v0 - v1)
            if dist <= epsilon:
                k[i, j] = dist * weight_func((epsilon / dist)**n_ftrs)
                k[j, i] = k[i, j]
    return k

def plot_density_kde(X, y, ax):
    h = 0.1
    xmin, ymin = np.min(X, axis=0)
    xmax, ymax = np.max(X, axis=0)

    xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                         np.arange(ymin, ymax, h))

    cm = plt.cm.Blues
    cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])    
    
    newx = np.c_[xx.ravel(), yy.ravel()]
    bandwidths = 10 ** np.linspace(-2, 1, 100)
    my_cv = LeaveOneOut()

    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=5,
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
    
    ax.scatter(X[:, 0], X[:, 1], c=y, 
                  cmap=cm_bright,
                  edgecolors='k',
                  zorder=2)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.grid(color='k', linestyle='-', linewidth=0.50, alpha=0.75)

    #plt.xticks(())
    #plt.yticks(())
    ax.set_title('Shortest Path - Density Based Distances (DBD)')

    return mdl

def plot_density_sphere(X, y, K, ax):
    h = 0.1
    xmin, ymin = np.min(X, axis=0)
    xmax, ymax = np.max(X, axis=0)

    xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                         np.arange(ymin, ymax, h))

    cm = plt.cm.Blues
    cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])    
    
    newx = np.c_[xx.ravel(), yy.ravel()]
    mdl = DensityEstimator()
    mdl.fit(X)
    Z = mdl.score_samples(newx, K)

    Z = Z.reshape(xx.shape)
    contour_plot = ax.contourf(xx, yy, Z, 
                                 levels=20,
                                 cmap=cm, 
                                 alpha=.8)
    plt.colorbar(contour_plot, ax=ax)
    
    ax.scatter(X[:, 0], X[:, 1], c=y, 
                  cmap=cm_bright,
                  edgecolors='k',
                  zorder=2)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.grid(color='k', linestyle='-', linewidth=0.50, alpha=0.75)

    #plt.xticks(())
    #plt.yticks(())
    ax.set_title('Shortest Path - Growing Sphere (GS)')

    return mdl

# =============================================================================
# K = 10
# n_samples = 200
# test_index = 200
# scaler = StandardScaler()
# np.random.seed(123)
# X, y = make_moons(n_samples=n_samples, noise=0.080)
# X = scaler.fit_transform(X)
# 
# X = np.concatenate((X, np.array([[1.70, 0.45], 
#                                  [-0.55, 0.55],
#                                  [0.50, 0.40]])))
# y = np.concatenate((y, np.array([1, 1, 1])))
# 
# fig, ax = plt.subplots(nrows=2, ncols=3)
# 
# kd_estimator = plot_density_kde(X, y, ax[0, 0])
# density_scorer = kd_estimator.score_samples
# kernel = get_weights(X, density_scorer, 1, weight_func=lambda x: -np.log(x))
# 
# graph = Graph()
# edges = get_edges(kernel)
# for edge in edges:
#     graph.add_edge(*edge)
# 
# mdl = GPC(1.0 * RBF(1.0))
# mdl.fit(X, y)
# plot_decision_boundary(X, y, ax[1, 0], mdl, title='GPC - DBD')
# plot_decision_boundary(X, y, ax[1, 1], mdl, title='GPC - GS')
# plot_decision_boundary(X, y, ax[1, 2], mdl, title='GPC - kNN-graph')
# plot_decision_boundary(X, y, ax[0, 2], mdl, title='GPC - e-graph')
# 
# predictions = mdl.predict_proba(X)
# # =============================================================================
# # path = dijsktra(graph, '100', '102')
# # print(path)
# # plot_path(X, path, ax)   
# # plt.show()
# # =============================================================================
# lim = 0.15
# dist, paths = dijsktra_toall(graph, test_index)
# 
# for item, val in paths.items():
#     if (y[item] != y[test_index] 
#         and predictions[item, y[item]] >= 0.90
#         and np.exp(kd_estimator.score_samples(X[int(item), :].reshape(1, -1))) >= lim):
#         plot_path(X, val, ax[0, 0])
#         plot_path(X, val, ax[1, 0])
#  
# ###################################################
# 
# d_estimator = plot_density_sphere(X, y, K, ax[0, 1])
# density_scorer = partial(d_estimator.score_samples, k=K)
# kernel = get_weights(X, density_scorer, 0, weight_func=lambda x: -np.log(x))
# 
# graph = Graph()
# edges = get_edges(kernel)
# for edge in edges:
#     graph.add_edge(*edge) 
#     
# radius_limit = 1.10
# dist, paths = dijsktra_toall(graph, test_index)
# 
# for item, val in paths.items():
#     if (y[item] != y[test_index] 
#         and predictions[item, y[item]] >= 0.90
#         and d_estimator.score_samples(X[int(item), :].reshape(1, -1), K+1) >= radius_limit):
#         plot_path(X, val, ax[0, 1])
#         plot_path(X, val, ax[1, 1])
# 
# ##############################
# 
# kernel = get_weights_kNN(X, n_neighbours=20)
# 
# graph = Graph()
# edges = get_edges(kernel)
# for edge in edges:
#     graph.add_edge(*edge) 
#     
# dist, paths = dijsktra_toall(graph, test_index)
# 
# for item, val in paths.items():
#     if (y[item] != y[test_index] 
#         and predictions[item, y[item]] >= 0.90):
#         plot_path(X, val, ax[1, 2])
#         
# ##############################
# 
# kernel = get_weights_e(X, epsilon=0.55)
# 
# graph = Graph()
# edges = get_edges(kernel)
# for edge in edges:
#     graph.add_edge(*edge) 
#     
# dist, paths = dijsktra_toall(graph, test_index)
# 
# for item, val in paths.items():
#     if (y[item] != y[test_index] 
#         and predictions[item, y[item]] >= 0.90):
#         plot_path(X, val, ax[0, 2])
# =============================================================================
