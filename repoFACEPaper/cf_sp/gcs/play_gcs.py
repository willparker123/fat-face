# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:36:28 2019

@author: rp13102
"""

from sklearn.neural_network import MLPClassifier as NN
from sklearn.preprocessing import StandardScaler
import numpy as np
from functools import partial

from gcs import get_gcs
from all_object import CFGenerator

def inv_transform(item, columns, codes):
    output = []
    for idx, value in enumerate(item):
        col_name = columns[idx]
        if col_name in codes.keys():
            func = codes[col_name]
            output.append(func.inverse_transform(np.array(int(value)).reshape(-1, 1))[0])
        else:
            try:
                output.append(value[0])
            except:
                output.append(value)
    return output

def inv_scale(item, columns, scalers):
    output = []
    for idx, value in enumerate(item):
        col_name = columns[idx]
        if col_name in scalers.keys():
            func = scalers[col_name]
            try:
                output.append(func.inverse_transform(np.array(int(value)).reshape(-1, 1))[0][0])
            except:
                output.append(func.inverse_transform(np.array(value).reshape(-1, 1))[0][0])
        else:
            try:
                output.append(value[0])
            except:
                output.append(value)
    return output

def edge_conditions(v0, v1, columns, codes):
    conditions = {
                'Age': lambda x, y: x == y,
                'Sex': lambda x, y: x == y,
                'Purpose': lambda x, y: x == y
                }
    for condition, func in conditions.items():
        idx = columns.get_loc(condition)
        if not func(v0[idx], v1[idx]):
            return False
    return True

float_formatter = lambda x: "%.2f" % x
float_formatter_arr = lambda x: np.array([float_formatter(item) for item in x])
#X, y, cols, codes, scalers = get_gcs()
#X = scaler.fit_transform(X)
F = partial(inv_transform, columns=cols, codes=codes)
G = partial(inv_scale, columns=cols, scalers = scalers)
edge_conditions_partial = partial(edge_conditions, 
                                  columns=cols,
                                  codes=codes)
mdl = CFGenerator(method='knn', 
                  predictor=NN,
                  n_neighbours=150,
                  prediction_threshold=0.51,
                  edge_conditions=edge_conditions_partial)
mdl.fit(X, y)
I = np.where(y == 0)[0]
successes = []
for idx in range(300): 
    test_index = I[idx]
    starting_point = X[test_index, :]
    target_class = int(not y[test_index])
    p=mdl.compute_sp(starting_point, target_class)
    if not p:
        print(test_index, 'failed')
        continue
    else:        
        sorted_distances = np.argsort([item[3] for item in p])
        t = p[sorted_distances[0]]; v=t[1]    
        successes.append((test_index, t))
        s = np.ravel(F(v.reshape(1, -1)))
        sp = G(starting_point)

for entry in successes[:1]:
    r = entry[1]
    starting_point = X[entry[0], :]
    #sp = F(G(starting_point))
    #print(sp)
    print(list(cols)[:-1])

    for idx in r[-1]:
        v = X[idx, :]
        s = F(G(v))
        print(s, mdl.predictions[idx])
    print('\n\n')
