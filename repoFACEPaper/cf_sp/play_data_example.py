# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:21:26 2019

@author: rp13102
"""
import numpy as npo
from cvxopt import matrix, solvers
from new import CFGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_classification
from sklearn.neural_network import MLPClassifier as NN
import matplotlib.pyplot as plt

import matplotlib.patheffects as path_effects
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF
#from kde import plot_decision_boundary
from adv2 import net_supp, plot_decision_boundary

import torch.nn as nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from advertorch.attacks import L1PGDAttack, LinfPGDAttack, L2PGDAttack, LBFGSAttack, CarliniWagnerL2Attack

def get_adv_(model, x, target):
    adversary = LBFGSAttack(
                model.predict, 
                loss_fn=nn.MSELoss(reduction="sum"),
                num_classes=2,
                #eps=3.,
                #nb_iter=250,
                clip_min=0, 
                clip_max=10.0,
                targeted=True,
                #penalty=penalty,
                #mad=mad
                )
    cln_data = Variable(torch.from_numpy(x.reshape(1, -1)).type(torch.FloatTensor).to(device))
    target = Variable(torch.from_numpy(npo.array([target])).float().to(device))
    return adversary.perturb(cln_data, target)
    
def get_data():  
    t = 2
    t1 = 10 * npo.random.random_sample(100 * t) - 0.50
    t0 = npo.random.normal(0, 0.40, 100 * t)
    x1 = npo.vstack((t0, t1)).T
    y1 = npo.ones(100 * t)
    
    t1 = 6 * npo.random.random_sample(100 * t) - 0.50
    t0 = npo.random.normal(0, 0.50, 100 * t)
    x2 = npo.vstack((t1, t0)).T
    y2 = npo.zeros(100 * t)
    
    mean = [3.50, 8.00]
    sigma = 0.50
    cov = sigma * npo.identity(2)
    n3 = 50 * t
    x3 = npo.random.multivariate_normal(mean, cov, n3)
    y3 = npo.zeros(n3)
    
    X = npo.concatenate((x1, x2, x3), axis=0)
    y = npo.concatenate((y1, y2, y3))
    
    return X, y

npo.random.seed(123)

X, y= get_data()
#X = np.delete(X, 230, axis=0)
#y = np.delete(y, 230)
#mdl = NN(max_iter=1000)
# =============================================================================
# mdl_cls = GPC(1.0 * RBF(1.0))
# mdl_cls.fit(X, y)
# 
# =============================================================================

subject = npo.array([0.20, 7.25])
clss = 1
X = npo.concatenate((X, subject.reshape(1, -1)))
y = npo.concatenate((y, [1])).astype(int)

# =============================================================================
# ax.scatter(subject[0], subject[1], 
#            c='lightgreen',
#            edgecolor='k',
#            s=200,
#            marker='x',
#            alpha=0.75,
#            zorder=2)
# =============================================================================
from copy import deepcopy

target_class = 0
netmdl = net_supp()
net=netmdl.fit(X, y)
import torch
from torch.autograd import Variable


# =============================================================================
# 
# from gp_play import gp_model
# mdl = gp_model().fit(X, y)
# 
# =============================================================================

# =============================================================================
# med = npo.median(X, axis=0)
# mad = npo.median(npo.abs(X - med), axis=0)
# penalties = [1, 3, 5]
# thresholds = [0.70, 0.90]
# ax = plot_decision_boundary(X, y, net)
# 
# import itertools
# vs = []
# params = list(itertools.product(penalties, thresholds))
# legend_params = []
# bla = []
# colormap = plt.get_cmap('Purples')
# colors = [colormap(k) for k in npo.linspace(0, 1, len(params))]
# 
# for idx, item in enumerate(params):
#     penalty, threshold = item
#     name = 'p = ' + str(penalty) + ', ' + 't = ' + str(threshold)
#     legend_params.append(name)
#     v = netmdl.get_adv(deepcopy(subject), 
#                        deepcopy(target_class), 
#                        penalty=penalty,
#                        mad=mad,
#                        threshold=threshold
#                        );
#     t = ax.scatter(v[0][0], v[0][1], 
#                    zorder = 2, 
#                    color = colors[idx], 
#                    marker='*', 
#                    s=200, 
#                    alpha = 0.50, 
#                    edgecolor='b',
#                    label=idx)
#     bla.append(t)
# #vs = npo.array(vs)
# #ax.scatter(vs[:, 0], vs[:, 1], marker='x', s=300, labels=idx)
# ax.legend(bla,
#            legend_params,
#            scatterpoints=1,
#            loc=1,
#            ncol=1,
#            fontsize=14,
#            frameon=True,
#            edgecolor='b',
#            framealpha=0.25
#            )
# 
# ax.scatter(subject[0], subject[1], 
#                zorder = 2, 
#                color = 'lightgreen', 
#                marker='x', 
#                s=200, 
#                alpha = 0.50, 
#                )
# =============================================================================
# =============================================================================
# X = npo.asarray(X)
# y = npo.asarray(y)
# # kde
# X = Variable(torch.tensor(X, dtype=torch.float32))
# #y = Variable(torch.tensor(y, dtype=torch.float32))
# for dist in [0.25, 0.50, 5]:
#     for dens in [0.001]:
#         mdl = CFGenerator(
#                         method='kde', 
#                       distance_threshold=dist,
#                       density_threshold=dens,
#                       predictor=net,
#                       prediction_threshold=0.75,
#                       howmanypaths=5)
#         mdl.fit(X, y)
#         starting_point = subject
#         target_class = 0
#         p=mdl.compute_sp(starting_point, target_class)
# =============================================================================
#mdl.ax.scatter(subject[0], subject[1], marker='x', s=300)
#mdl.ax.scatter(v[0][0], v[0][1], marker='x', s=300)

#

# egraph

# =============================================================================
# X = Variable(torch.tensor(X, dtype=torch.float32))
# 
# for dist in [0.25, 0.50, 1, 3]:
#     mdl = CFGenerator(
#                     method='egraph', 
#                   distance_threshold=dist,
#                   predictor = net,
#                   epsilon=dist,
#                   prediction_threshold=0.75,
#                   howmanypaths=5)
#     mdl.fit(X, y)
#     starting_point = subject
#     target_class = 0
#     p=mdl.compute_sp(starting_point, target_class)
# =============================================================================
        
# =============================================================================
# # knn
# X = Variable(torch.tensor(X, dtype=torch.float32))
# for n_neighbours in [10]:
#     for dist in [0.80]:
#         mdl = CFGenerator(method='knn', 
#                       n_neighbours=n_neighbours,
#                       distance_threshold=dist,
#                       predictor=net,
#                       prediction_threshold=0.75,
#                       howmanypaths=5)
#         mdl.fit(X, y)
#         starting_point = subject
#         target_class = 0
#         p=mdl.compute_sp(starting_point, target_class)
# =============================================================================
        
# =============================================================================
# # knn
# for n_neighbours in [4]:
#     for dens in [0.05]:
#         mdl = CFGenerator(method='knn', 
#                       n_neighbours=n_neighbours,
#                       density_threshold='a',
#                       prediction_threshold=0.75,
#                       howmanypaths=10)
#         mdl.fit(X, y)
#         starting_point = subject
#         target_class = 0
#         p, ax=mdl.compute_sp(starting_point, target_class)
#         
# =============================================================================
# =============================================================================
# # gs
# for gamma in [0.25, 0.50, 1, 2]:
#     for radius_limit in [1]:
#         K = 5
#         radius_limit = K
#         mdl = CFGenerator(method='gs', 
#                           K=K,
#                           radius_limit=radius_limit,
#                           distance_threshold=0.50,
#                           prediction_threshold=0.75,
#                           howmanypaths=10)
#         mdl.fit(X, y)
#         starting_point = subject
#         target_class = 0
#         p=mdl.compute_sp(starting_point, target_class)
# =============================================================================
        
# =============================================================================
# 
# v = []
# for idx, item in enumerate(results):
#     if len(item) > 0:
#         v.append(idx)
#         
# for t in range(10):
#     if len(results[v[t]]) > 1:
#         rows = results[v[t]][0][-1]
#     else:
#         rows = results[v[t]][-1]
#     print(df.columns)
#     for item in rows:
#         print(scaler.inverse_transform(X[item, :]))
#     print('\n')
# =============================================================================
