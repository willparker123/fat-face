# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:33:03 2019

@author: rp13102
"""

from copy import deepcopy

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm

from functools import partial
from cvxopt import matrix, solvers
from sklearn.neural_network import MLPClassifier as NN

from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

import math

from dijsktra_algorithm import Graph, dijsktra_toall, dijsktra_tosome
from density_estimator import DensityEstimator

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_volume_of_sphere(d):
    return math.pi**(d/2)/math.gamma(d/2 + 1)

def get_edges(kernel):
    edges = []

    n_samples = kernel.shape[0]
    for i in range(n_samples):
        for j in range(n_samples):
            if kernel[i, j] != 0 :
                edges.append([i, j, kernel[i, j]])
    return edges

class CFGenerator(object):
    def __init__(self,
                 method=None,
                 weight_function=None,
                 predictor=None,
                 prediction_threshold=None,
                 density_threshold=None,
                 K=None,
                 radius_limit=None,
                 n_neighbours=None,
                 epsilon=None,
                 distance_threshold=None,
                 edge_conditions=None,
                 howmanypaths=None,
                 undirected=False):
        
        self.edge_conditions = edge_conditions
        self.undirected = undirected
        
        if method is None:
            self.method = 'knn'
        elif method in ['knn', 'kde', 'gs', 'egraph']:
            self.method = method
        else:
            raise ValueError('Unknown method')
          
        if howmanypaths is None:
            self.howmanypaths = 5
        else:
            self.howmanypaths = howmanypaths
            
        if weight_function is None:
            self.weight_function = lambda x: -np.log(x)
        else:
            self.weight_function = weight_function
            
        if predictor is None:
            self.predictor = NN(max_iter=int(1e6))#GPC(1.0 * RBF(1.0)) #
        else:
            if not hasattr(predictor, 'predict_proba'):
                raise ValueError('Predictor needs to have attribute: \'predict proba\'')
            else:
                self.predictor = predictor
    
        if prediction_threshold is None:
            self.prediction_threshold = 0.60    
        else:
            self.prediction_threshold = prediction_threshold
         
        if density_threshold is None:
            self.density_threshold = 1e-5
        else:
            self.density_threshold = density_threshold
            
        if K is None:
            self.K = 10
        else:
            self.K = K
        
        if epsilon is None:
            self.epsilon = 0.75
        else:
            self.epsilon = epsilon
            self.distance_threshold = distance_threshold
            
        if radius_limit is None:
            self.radius_limit = 1.10
        else:
            self.radius_limit = radius_limit
            
        if n_neighbours is None:
            self.n_neighbours = 20
        else:
            self.n_neighbours = n_neighbours
        
        if distance_threshold is None:
            self.distance_threshold = 'a'
        else:
            self.distance_threshold = distance_threshold
       
    def plot_pred_and_density(self):
        if self.method in ['kde', 'gs']:
            #fig, ax = plt.subplots(nrows=2, ncols=1)
            _, ax = plt.subplots()
            #_, ax1 = plt.subplots()
            
            if self.method == 'kde':
                self.plot_decision_boundary(ax)
            else:
                self.plot_gs_scores(ax)
        elif self.method in ['knn', 'egraph']:
            fig, ax = plt.subplots()
            self.plot_decision_boundary(ax)
        return ax
    
    def get_weights_e(self):
        k = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            for j in range(i):
                v1 = self.X[j, :].reshape(-1, 1)
                if not self.check_conditions(v0, v1):
                    continue
                dist = np.linalg.norm(v0 - v1)
                if dist <= self.epsilon:
                    dist = self.epsilon
                    k[i, j] = dist * self.weight_function((self.epsilon / dist)**self.n_features)
                    k[j, i] = k[i, j]
        return k

    def get_weights_kNN(self):       
        volume_sphere = get_volume_of_sphere(self.n_features)
        const = (self.n_neighbours / (self.n_samples * volume_sphere))**(1/self.n_features)
        k = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            counter = 0
            for j in range(self.n_samples):
                v1 = self.X[j, :].reshape(-1, 1)
# =============================================================================
#                 if not self.check_conditions(v0, v1):
#                     continue
#                 k[i, j] = np.linalg.norm(v0 - v1)
# =============================================================================
                dist = np.linalg.norm(v0 - v1)
                if (self.check_conditions(v0, v1) 
                    and dist <= self.distance_threshold):
                    k[i, j] = self.distance_threshold
                    
                else:
                    counter += 1
            t = np.argsort(k[i, :])[(1+counter+self.n_neighbours):]
            mask = np.ix_(t)
            k[i, mask] = 0
            
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            for j in range(self.n_samples):
                v1 = self.X[j, :].reshape(-1, 1)
                if k[i, j] != 0:
                    current_value = k[i, j]
                    k[i, j] = current_value * self.weight_function(const / (current_value**self.n_features))
        return k

    def get_weights_kde(self, 
                    density_scorer,
                    mode):
        k = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            for j in range(self.n_samples):#range(i):
                v1 = self.X[j, :].reshape(-1, 1)
                if not self.check_conditions(v0, v1):
                    continue
                dist = np.linalg.norm(v0 - v1)
                if dist <= self.distance_threshold:
                    midpoint = (v0 + v1)/2
                    density = density_scorer(midpoint.reshape(1, -1))
                    if mode == 1:
                        k[i, j] = self.weight_function(np.exp(density)) * dist
                    else:
                        k[i, j] = self.weight_function(sigmoid(density)) * dist
                else:
                    k[i, j] = 0
                #k[j, i] = k[i, j]
        return k
            
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = self.X.shape
        if self.n_samples != self.y.shape[0]:
            raise ValueError('Inconsistent dimensions')
        self.predictor.fit(self.X, self.y)
        self.predictions = self.predictor.predict_proba(X)
        self.kernel = self.get_kernel()
        self.fit_graph()
        
    def prepare_grid(self):
        h = 0.1
        V = self.X.clone().detach().numpy()
        xmin, ymin = np.min(V, axis=0)
        xmax, ymax = np.max(V, axis=0)
    
        self.xx, self.yy = np.meshgrid(np.arange(xmin, xmax, h),
                             np.arange(ymin, ymax, h))
    
        self.cm = plt.cm.Blues
        self.cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])    
        
        newx = np.c_[self.xx.ravel(), self.yy.ravel()]
        return newx
    
    def plot_decision_boundary(self, ax, title='GPC'):
        newx = self.prepare_grid()
       
        self.cm = plt.cm.RdYlBu
        
        ax.set_xlim(self.xx.min(), self.xx.max())
        ax.set_ylim(self.yy.min(), self.yy.max())
        

        newx = np.c_[self.xx.ravel(), self.yy.ravel()]
        newx = Variable(torch.tensor(newx, dtype=torch.float32))
        Z = self.predictor.predict(newx)[:, 1].detach().numpy()
    
        self.plot_density_scores(Z, ax, self.method)
# =============================================================================
#         Z = Z.reshape(self.xx.shape)
#         contour_plot = ax.contourf(xx, yy, Z, 
#                                    levels=20, 
#                                    cmap=cm, 
#                                    alpha=.8)
#     
#         plt.colorbar(contour_plot, ax=ax)
#         ax.scatter(X[:, 0], X[:, 1], c=y, 
#                    cmap=cm_bright,
#                    edgecolors='k',
#                    zorder=2)
#         ax.grid(color='k', 
#                 linestyle='-', 
#                 linewidth=0.50, 
#                 alpha=0.75)
#         ax.set_title(title)
# =============================================================================
    
    def plot_density_scores(self, Z, ax,
                            title=None):
        Z = Z.reshape(self.xx.shape)
        contour_plot = ax.contourf(self.xx, self.yy, Z, 
                                   levels=20,
                                   cmap=self.cm, 
                                   alpha=.8)
        #plt.colorbar(contour_plot, ax=ax)
        
        ax.set_xlim(self.xx.min(), self.xx.max())
        ax.set_ylim(self.yy.min(), self.yy.max())
        
        ax.scatter(self.X[:, 0], self.X[:, 1], 
                   c=self.y, 
                   cmap=self.cm_bright,
                   edgecolors='k',
                   zorder=1)
        ax.grid(color='k', 
                linestyle='-', 
                linewidth=0.50, 
                alpha=0.75)

        if title is not None:
            pass #ax.set_title(title)
    
    def get_kernel(self):
        if self.method == 'kde':
            self.get_kde()
            density_scorer = self.density_estimator.score_samples
            kernel = self.get_weights_kde(density_scorer, 1)
            
        elif self.method == 'gs':
            self.get_gs_estimator()
            density_scorer = partial(self.density_estimator.score_samples, k=self.K)
            kernel = self.get_weights_kde(density_scorer, 0)
            
        elif self.method == 'knn':
            kernel = self.get_weights_kNN()
            
        elif self.method == 'egraph':
            kernel = self.get_weights_e()
         
        self.kernel = kernel
        return kernel
    
    def fit_graph(self):        
        self.graph = Graph(undirected=self.undirected)
        edges = get_edges(self.kernel)
        for edge in edges:
            self.graph.add_edge(*edge) 

    def plot_path(self, path, ax, color='lightgreen', extra_point=None):
        V = self.X.clone().detach().numpy()
        if self.n_features != 2:
            return 0
        n_nodes = len(path)
        if isinstance(extra_point, np.ndarray):
            ax.plot([self.X[-1, 0], extra_point[0]],
                    [self.X[-1, 1], extra_point[1]],
                    'k', alpha=0.50)
            ax.scatter(extra_point[0], extra_point[1],
                       color='k',
                       marker='o',
                       facecolors='lightyellow',
                       edgecolors='lightyellow',
                       alpha = 0.80,
                       zorder=1,
                       s=250)
            
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
            ax.plot(V[[i, j], 0], V[[i, j], 1], 'k', alpha=0.50)
        
        ax.scatter(V[path[-1], 0], V[path[-1], 1],
                   color='k',
                   marker='o',
                   facecolors=color,
                   edgecolors=color,
                   alpha = 0.50,
                   zorder=2,
                   s=150)
    
    def condition(self, item):
        # this function is only needed for GS -- been kim's method
        
        pred = self.predictions[item, self.y[item]]
        if (self.y[item] == self.target_class
                and pred >= self.prediction_threshold):
            if self.method == 'gs':           
                gs_score = self.density_estimator.score_samples(self.X[int(item), :].reshape(1, -1), self.K+1)
                if gs_score >= self.radius_limit:
                    return (pred, gs_score), True
            elif self.method == 'kde':
                kde = np.exp(self.density_estimator.score_samples(self.X[int(item), :].reshape(1, -1))) 
                if kde >= self.density_threshold:
                    return (pred, kde), True
            elif self.method in ['knn', 'egraph']:
                return (pred), True
        return 0, False

    def check_conditions(self, v0, v1):
        if self.edge_conditions is None:
            return True
        else:
            return self.edge_conditions(v0, v1)
        
    def check_individual_conditions(self, v0, v1):
        return self.individual_edge_conditions(v0, v1)
    
    def modify_kernel(self):
        personal_kernel = self.kernel.copy()
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            for j in range(i):
                v1 = self.X[j, :].reshape(-1, 1)

                if not self.check_individual_conditions(v0, v1):                   
                    personal_kernel[i, j] = 0
                    personal_kernel[j, i] = 0
        return personal_kernel
                    
    def compute_sp(self, 
                   starting_point,
                   target_class,
                   starting_point_index,
                   plot = True,
                   individual_edge_conditions=None):
        self.individual_edge_conditions = individual_edge_conditions
        if self.n_features != 2:
            plot = False
        self.target_class = target_class
        #V = self.X.clone().detach().numpy()
        V = deepcopy(self.X)
        #starting_point_index = np.where((V == starting_point).all(axis=1))[0][0]
        #starting_point_index = np.where(self.X == starting_point)[0][0]
        t0 = np.where(self.predictions >= self.prediction_threshold)[0]
        t1 = np.where(self.y == self.target_class)[0]
        if self.method == 'kde':
            kde = np.exp(self.density_estimator.score_samples(deepcopy(self.X))) 

            t2 = np.where(kde >= self.density_threshold)[0]
            self.candidate_targets = list(set(t0).intersection(set(t1)).intersection(set(t2)))
        else:
            self.candidate_targets = list(set(t0).intersection(set(t1)))
            
        if self.individual_edge_conditions is None:
            dist, paths = dijsktra_toall(
                                deepcopy(self.graph), 
                                starting_point_index
                                )
# =============================================================================
#             dist, paths = dijsktra_tosome(self.graph, 
#                                           starting_point_index, 
#                                           self.candidate_targets)
# =============================================================================
        else:
            self.personal_kernel = self.modify_kernel(self.kernel)
            self.personal_graph = Graph()
            edges = get_edges(self.personal_kernel)
            for edge in edges:
                self.personal_graph.add_edge(*edge) 
            dist, paths = dijsktra_tosome(self.personal_graph, 
                                          starting_point_index, 
                                          self.candidate_targets)
        
        if plot:
            ax = self.plot_pred_and_density()
# =============================================================================
#         all_paths = []
#         for item, path in paths.items():
#             all_paths.append((item, self.X[item, :], dist[item], path))
# =============================================================================
        all_paths = []
        for item, path in paths.items():
            value, satisfied = self.condition(item)

            if satisfied:
                all_paths.append((item, self.X[item, :], dist[item], value, path))
        all_paths = sorted(all_paths, key=lambda x: x[2])
        
        if plot:
            self.plot_paths(ax, all_paths)             
        return all_paths 
    
    def plot_paths(self, ax, all_paths):
        counter = 0
        colors=cm.Greens(np.linspace(0,1,self.howmanypaths))

        for idx, item in enumerate(all_paths):
            if counter > self.howmanypaths - 1:
                    break
            path = item[-1]
            if self.method in ['kde', 'gs']:
                self.plot_path(path, ax, colors[counter])
            else:
                self.plot_path(path, ax, colors[counter])
            counter += 1   
            
    def get_gs_estimator(self):
        self.density_estimator = DensityEstimator()
        self.density_estimator.fit(self.X)
        
    def plot_gs_scores(self, ax):
        newx = self.prepare_grid()
        if (self.n_features == 2):
            Z = self.density_estimator.score_samples(newx, self.K)
            self.plot_density_scores(Z, ax) 
            
    def get_kde(self):
        bandwidths = 10 ** np.linspace(0, 1, 10)  
        #bandwidths = [0.65]
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=4,
                            iid=True)    
        #grid.fit(self.X.numpy())
        grid.fit(deepcopy(self.X))
        self.density_estimator = grid.best_estimator_
        
    def plot_density(self, ax):
        newx = self.prepare_grid()
        if (self.n_features == 2):
            Z = np.exp(self.density_estimator.score_samples(newx))
            self.plot_density_scores(Z, ax)
    
            