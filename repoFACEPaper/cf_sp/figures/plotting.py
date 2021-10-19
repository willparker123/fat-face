# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:13:53 2019

@author: rp13102
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as NN
import matplotlib.patheffects as path_effects

from kde import plot_decision_boundary

np.random.seed(123)
x1 = np.random.multivariate_normal([1.25, -1.25], [[0.20, 0], [0, 0.20]], 200)
x2 = np.random.multivariate_normal([-6, 5.50], [[0.80, 0], [0, 0.80]], 30)
x3 = np.random.multivariate_normal([-5, 0.50], [[0.35, 0], [0, 0.35]], 50)

X = np.concatenate((x1, x2, x3), axis=0)
y = np.ravel(np.concatenate((np.ones(230), np.zeros(50))))
# =============================================================================
# 
# # =============================================================================
# # X = np.concatenate((X, np.array([-4, 2]).reshape(1, -1)), axis=0)
# # y = np.concatenate((y, np.zeros(1)))
# # 
# # =============================================================================
# fig, ax = plt.subplots(nrows=1)
# 
# # =============================================================================
# # kd_estimator = plot_density_kde(X, y, ax[0])
# # density_scorer = kd_estimator.score_samples
# # kernel = get_weights(X, density_scorer, 1, weight_func=lambda x: -np.log(x))
# # 
# # graph = Graph()
# # edges = get_edges(kernel)
# # for edge in edges:
# #     graph.add_edge(*edge)
# # =============================================================================
# 
# mdl = NN(max_iter=1000)
# #mdl = GPC(1.0 * RBF(1.0))
# mdl.fit(X, y)
# plot_decision_boundary(X, y, ax, mdl, title='Neural Network Classification')
# 
# subject = np.array([-5, 1])
# clss = 0
# 
# ax.scatter(subject[0], subject[1], 
#            c='lightgreen',
#            edgecolor='k',
#            s=200,
#            marker='x',
#            alpha=0.75,
#            zorder=2)
# 
# target0 = np.array([-4.50, 2.32])
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
# target1 = np.array([0.60, -1.20])
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
# target2 = np.array([-6, 4.50])
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
