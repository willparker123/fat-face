# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:33:03 2019

@author: rp13102
"""

from copy import deepcopy
import numpy as npo
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
    return 1/(1 + npo.exp(-x))

def get_volume_of_sphere(d):
    return math.pi**(d/2)/math.gamma(d/2 + 1)

def get_edges(kernel):
    edges = []

    n_samples = kernel.shape[0]
    for i in range(n_samples):
        for j in range(i):
            weight = kernel[i, j]
            if weight != 0 :
                edges.append([i, j, weight])
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
                 undirected=True):
        
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
            self.weight_function = lambda x: -npo.log(x)
        else:
            self.weight_function = weight_function
            
        if predictor is None:
            self.predictor = NN(max_iter=10000)#GPC(1.0 * RBF(1.0))
        else:
            self.predictor = predictor
# =============================================================================
#             if not hasattr(predictor, 'predict_proba'):
#                 pass #raise ValueError('Predictor needs to have attribute: \'predict proba\'')
#             else:
#                 self.predictor = predictor(max_iter=10000)
# =============================================================================
    
        if prediction_threshold is None:
            self.prediction_threshold = 0.60    
        else:
            self.prediction_threshold = prediction_threshold
         
        if density_threshold is None:
            self.density_threshold = 0.001
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
        
    def get_weights_e(self):
        k = npo.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            for j in range(i):
                v1 = self.X[j, :].reshape(-1, 1)
                dist = npo.linalg.norm(v0 - v1)
                if dist <= self.epsilon:
                    k[i, j] = self.epsilon
                    #k[i, j] = dist * self.weight_function((self.epsilon / dist)**self.n_features)
                    k[j, i] = k[i, j]
        return k

    def get_weights_kNN(self):       
        volume_sphere = get_volume_of_sphere(self.n_features)
        const = (self.n_neighbours / (self.n_samples * volume_sphere))**(1/self.n_features)
        k = npo.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            counter = 0
            for j in range(self.n_samples):
                v1 = self.X[j, :].reshape(-1, 1)
                k[i, j] = npo.linalg.norm(v0 - v1)
# =============================================================================
#                 if self.edge_coditions(v0, v1):
#                     dist = npo.linalg.norm(v0 - v1)
#                     k[i, j] = dist
#                 else:
#                     counter += 1
# =============================================================================
            t = npo.argsort(k[i, :])[(1+counter+self.n_neighbours):]
            mask = npo.ix_(t)
            k[i, mask] = 0
            
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            for j in range(self.n_samples):
                v1 = self.X[j, :].reshape(-1, 1)
                if k[i, j] != 0:
                    current_value = k[i, j]
                    k[i, j] = current_value * self.weight_function(const / (current_value**self.n_features))
        return k

    def get_weights(self, 
                    density_scorer,
                    mode):
        k = npo.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            for j in range(i):
                v1 = self.X[j, :].reshape(-1, 1)
                dist = npo.linalg.norm(v0 - v1)
                if dist <= self.distance_threshold:
                    midpoint = (v0 + v1)/2
                    density = density_scorer(midpoint.reshape(1, -1))
                    if mode == 1:
                        k[i, j] = self.weight_function(npo.exp(density)) * dist
                    else:
                        k[i, j] = self.weight_function(sigmoid(density)) * dist
                else:
                    k[i, j] = 0
                k[j, i] = k[i, j]
        return k
            
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = self.X.shape
        if self.n_samples != self.y.shape[0]:
            raise ValueError('Inconsistent dimensions')
        #self.predictor.fit(self.X, self.y)
        self.predictions = self.predictor.predict(deepcopy(X)) #predict_proba(X)
        kernel = self.get_kernel()
        self.kernel = kernel
        self.fit_graph(kernel)
        
    def prepare_grid(self):
        h = 0.1
        xmin, ymin = npo.min(self.X, axis=0)
        xmax, ymax = npo.max(self.X, axis=0)
    
        self.xx, self.yy = npo.meshgrid(npo.arange(xmin, xmax, h),
                             npo.arange(ymin, ymax, h))
    
        self.cm = plt.cm.Blues
        self.cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])    
        
        newx = npo.c_[self.xx.ravel(), self.yy.ravel()]
        return newx
    
    def plot_decision_boundary(self, ax, title='GPC'):
        newx = self.prepare_grid()
       
        self.cm = plt.cm.RdYlBu
        
        ax.set_xlim(self.xx.min(), self.xx.max())
        ax.set_ylim(self.yy.min(), self.yy.max())
        

        newx = npo.c_[self.xx.ravel(), self.yy.ravel()]
        Z = self.predictor.predict(deepcopy(newx))[:, 1] # predict_proba
        Z = Z.data.numpy()
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
        plt.colorbar(contour_plot, ax=ax)
        
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
            kernel = self.get_weights(density_scorer, 1)
            
        elif self.method == 'gs':
            self.get_gs_estimator()
            density_scorer = partial(self.density_estimator.score_samples, k=self.K)
            kernel = self.get_weights(density_scorer, 0)
            
        elif self.method == 'knn':
            kernel = self.get_weights_kNN()
            
        elif self.method == 'egraph':
            kernel = self.get_weights_e()
         
        self.kernel = kernel
        return kernel
    
    def fit_graph(self, kernel):        
        self.graph = Graph(undirected=self.undirected)
        edges = get_edges(kernel)
        for edge in edges:
            self.graph.add_edge(*edge) 

    def plot_path(self, path, ax, color='lightgreen', extra_point=None):
        if self.n_features != 2:
            return 0
        n_nodes = len(path)
        if isinstance(extra_point, npo.ndarray):
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
            ax.plot(self.X[[i, j], 0], self.X[[i, j], 1], 'k', alpha=0.50)
        
        ax.scatter(self.X[path[-1], 0], self.X[path[-1], 1],
                   color='k',
                   marker='o',
                   facecolors=color,
                   edgecolors=color,
                   alpha = 0.50,
                   zorder=2,
                   s=150)
    
    def condition(self, item):
        pred = self.predictions[item, self.y[item]]
        if (self.y[item] == self.target_class
                and pred >= self.prediction_threshold):
            if self.method == 'gs':           
                gs_score = self.density_estimator.score_samples(self.X[int(item), :].reshape(1, -1), self.K+1)
                if gs_score >= self.radius_limit:
                    return (pred, gs_score), True
            elif self.method == 'kde':
                kde = npo.exp(self.density_estimator.score_samples(self.X[int(item), :].reshape(1, -1))) 
                if kde >= self.density_threshold:
                    return (pred, kde), True
            elif self.method in ['knn', 'egraph']:
                return (pred), True
        return 0, False
     
    def solve_qp(self, starting_point, fixed):
        solvers.options['show_progress'] = False

        Q = matrix(npo.ones((self.n_features, self.n_features), dtype=float))
        p = 2*matrix(starting_point.reshape(-1, 1))
        
        G = matrix(npo.zeros((self.n_features, self.n_features), dtype=float))
        h = matrix(npo.zeros((self.n_features, 1), dtype=float))
        
        A = matrix(npo.identity(self.n_features, dtype=float))
        t = npo.zeros((self.n_features, 1), dtype=float)
        for item in fixed:
            t[item] = starting_point[item]
            A[item, item] = 1
        b = matrix(t.reshape(-1, 1))  

        sol=solvers.qp(Q, p, G, h, A, b)
        return sol
    
    def check_conditions(self, v0, v1):
        return self.edge_conditions(v0, v1)
    
    def modify_kernel(self, base_kernel, fixed):
        personal_kernel = base_kernel.copy()
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            for j in range(i):
                v1 = self.X[j, :].reshape(-1, 1)
                                #if not npo.all(v0[fixed] == v1[fixed]):

                if not self.check_conditions(v0, v1):                   
                    personal_kernel[i, j] = 0
                    personal_kernel[j, i] = 0
        return personal_kernel
                    
    def compute_sp(self, 
                   starting_point,
                   target_class,
                   fixed = None,
                   plot = True):
        counter = 0
        if self.n_features != 2:
            plot = False
        self.target_class = target_class
        starting_point_index = npo.where((self.X == starting_point).all(axis=1))[0][0]
        predictions = self.predictor.predict_proba(deepcopy(self.X))[:, self.target_class]
        
        t0 = npo.where(predictions >= self.prediction_threshold)[0]
        t1 = npo.where(self.y == self.target_class)[0]
        self.candidate_targets = list(set(t0).intersection(set(t1)))

# =============================================================================
#         if not candidate_targets:
#             return -1
# =============================================================================
        if self.edge_conditions is None:
            #dist, paths = dijsktra_toall(self.graph, starting_point_index)
            dist, paths = dijsktra_tosome(self.graph, 
                                          starting_point_index, 
                                          self.candidate_targets)
        else:
            self.personal_kernel = self.modify_kernel(self.kernel, fixed)
            self.personal_graph = Graph()
            edges = get_edges(self.personal_kernel)
            for edge in edges:
                self.personal_graph.add_edge(*edge) 
            #dist, paths = dijsktra_toall(self.personal_graph, starting_point_index)
            dist, paths = dijsktra_tosome(self.personal_graph, 
                                          starting_point_index, 
                                          self.candidate_targets)
        
        if plot:
            if self.method in ['kde', 'gs']:
                #fig, ax = plt.subplots(nrows=2, ncols=1)
                _, ax0 = plt.subplots()
                #_, ax1 = plt.subplots()
                
                if self.method == 'kde':
                    self.plot_density(ax0)
                else:
                    self.plot_gs_scores(ax0)
                _, ax1=plt.subplots()
                self.plot_decision_boundary(ax1)
            elif self.method in ['knn', 'egraph']:
                fig, ax = plt.subplots()
                self.plot_decision_boundary(ax)
        all_paths = []
        for item, path in paths.items():
            value, satisfied = self.condition(item)

            if satisfied:
                all_paths.append((item, self.X[item, :], dist[item], value, path))
                #all_paths.append((item, self.X[item, :], dist[item][0], value, path))
        all_paths = sorted(all_paths, key=lambda x: x[2])
        colors=cm.Greens(npo.linspace(0,1,self.howmanypaths))

        for idx, item in enumerate(all_paths):
            if counter > self.howmanypaths - 1:
                    break
            path = item[-1]
            if plot:
                if self.method in ['kde', 'gs']:
                    
                    self.plot_path(path, ax0, colors[counter])
                    #self.plot_path(path, ax1, colors[counter])
                else:
                    self.plot_path(path, ax, colors[counter])
            counter += 1    
        #ax0.set_title([self.K, self.radius_limit])     
        if plot:
            if self.method in ['kde', 'gs']:
                self.ax = ax1
            else:
                self.ax = ax
        return all_paths
      
    def get_gs_estimator(self):
        self.density_estimator = DensityEstimator()
        self.density_estimator.fit(self.X)
        
    def plot_gs_scores(self, ax):
        newx = self.prepare_grid()
        if (self.n_features == 2):
            Z = self.density_estimator.score_samples(newx, self.K)
            self.plot_density_scores(Z, ax) 
            
    def get_kde(self):
        bandwidths = 10 ** npo.linspace(-2, 1, 100)  
        #bandwidths = [0.60]
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=20,
                            iid=True)    
        grid.fit(self.X)
        self.density_estimator = grid.best_estimator_
        
    def plot_density(self, ax):
        newx = self.prepare_grid()
        if (self.n_features == 2):
            Z = npo.exp(self.density_estimator.score_samples(newx))
            self.plot_density_scores(Z, ax)
    
            