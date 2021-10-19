# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:36:28 2019

@author: rp13102
"""

from sklearn.neural_network import MLPClassifier as NN
from sklearn.preprocessing import StandardScaler
import numpy as np
from functools import partial

#from gcs import get_gcs
#import sys
#sys.path.append(r'C:\Users\rp13102\Documents\GitHub\experiment\ghent\fico')

#from load_data import get_data

from play_fico_2 import get_data
from all_object import CFGenerator

def inv_scale(item, columns, scalers):
    output = []
    scalers_keys = set(scalers.keys())
    for idx, value in enumerate(item):
        col_name = columns[idx]
        if col_name in scalers_keys:
            func = scalers[col_name]
            output.append(func.inverse_transform(np.array(value).reshape(-1, 1))[0][0])
        else:
            output.append(value)
# =============================================================================
#         else:
#             try:
#                 output.append(value[0])
#             except:
#                 output.append(value)
# =============================================================================
    return output

def edge_conditions(v0, v1, columns):
    return True
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
# =============================================================================
# 
# float_formatter = lambda x: "%.2f" % x
# float_formatter_arr = lambda x: np.array([float_formatter(item) for item in x])
# #X, y, cols, scalers, df = get_data()
# #X = scaler.fit_transform(X)
# X, newdf, df, y, scalers = get_data()
# cols = newdf.columns
# 
# edge_conditions_partial = partial(edge_conditions, 
#                                   columns=cols)
# mdl = CFGenerator(method='knn', 
#                   predictor=NN,
#                   n_neighbours=100,
#                   prediction_threshold=0.51,
#                   edge_conditions=edge_conditions_partial)
# mdl.fit(X, y)
# =============================================================================
# =============================================================================
# I = np.where(y == 0)[0]
# successes = []
# 
# for idx in range(500): 
#     print(idx)
#     test_index = I[idx]
#     starting_point = X[test_index, :]
#     target_class = int(not y[test_index])
#     p=mdl.compute_sp(starting_point, target_class)
#     if not p:
#         print(test_index, 'failed')
#         continue
#     else:        
#         sorted_distances = np.argsort([item[3] for item in p])
#         t = p[sorted_distances[0]]; v=t[1]    
#         successes.append((test_index, t))
#         s = np.ravel(v.reshape(1, -1))
#         sp = G(starting_point)
# 
# =============================================================================
G = partial(inv_scale, columns=cols, scalers = scalers)
for entry in successes[:10]:
    r = entry[1]
    starting_point = X[entry[0], :]
    #sp = F(G(starting_point))
    #print(sp)
    l = len(list(df.columns))
    indices = [idx for idx in r[-1]]
    scaled_rows = [G(X[idx, :]) for idx in r[-1]]
    df_rows = [df.loc[idx, :] for idx in r[-1]]
    for i in range(l):
        print(df.columns[i], '\t\t', 
              [item[i] for item in df_rows], 
              #[item[i] for item in scaled_rows]
              )
# =============================================================================
#     print(list(cols)[1:])
# 
#     for idx in r[-1]:
#         v = X[idx, :]
#         s = G(v)
#         print(s, mdl.predictions[idx])
# =============================================================================
    print('\n\n')
