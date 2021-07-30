# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:21:26 2019

@author: rp13102
"""
import numpy as np
from cvxopt import matrix, solvers
#from all_object import CFGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_classification
from sklearn.neural_network import MLPClassifier as NN
import matplotlib.pyplot as plt

#import sys
#sys.path.append(r'C:\Users\rp13102\Documents\GitHub\cf_sp')
from all_object import CFGenerator

def edge_conditions(v0, v1):
    for idx in [0, 4, 5]:
        if v0[idx] != v1[idx]:
            return 0

np.random.seed(42)
df = load_adult()
N = 1000
p = np.random.choice(30000, N, replace=False)

y = df['salary'].values.astype(int)
df.drop(labels=['salary'], axis=1, inplace=True)
X = df.values.astype(float)

y = y[p]
X = X[p, :]

scaler = StandardScaler()
X = scaler.fit_transform(X)
fixed = [0, 1, 4, 5]
results = []
for d in [5]:

    mdl = CFGenerator(method='egraph',
                      epsilon=d,
                      distance_threshold=d,
                      edge_conditions=edge_conditions)
    mdl.fit(X, y)

    for test_index in range(N):
        if y[test_index] == 0:
            continue
        print(test_index)
        starting_point = X[test_index, :]
        target_class = int(not y[test_index])
        p=mdl.compute_sp(starting_point, target_class, fixed=fixed)
        results.append(p)

v = []
for idx, item in enumerate(results):
    if len(item) > 0:
        v.append(idx)

for t in range(len(v)):
    rows = results[v[t]][0][-1]
# =============================================================================
#     if len(results[v[t]]) > 1:
#         rows = results[v[t]][0][-1]
#     else:
#         rows = results[v[t]][-1]
# =============================================================================
    if len(rows) <= 2:
        continue
    #print(df.columns)
    for item in rows:
        print(scaler.inverse_transform(X[item, :]))
    print('\n')
