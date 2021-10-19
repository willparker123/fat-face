# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 17:22:01 2019

@author: rp13102
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

import sys
sys.path.append(r'C:\Users\rp13102\Documents\GitHub\experiment\ghent\data')

def get_data():
    np.random.seed(123)
    path = r'C:\Users\rp13102\Documents\GitHub\experiment\ghent\data\\'
    filename = 'heloc_dataset_v1.csv'
    filepath = path + filename
    df = pd.read_csv(filepath,
                     delimiter=',')
    y = df['RiskPerformance'].copy()
    y.replace('Good', 1, inplace=True)
    y.replace('Bad', 0, inplace=True)
    y = y.values
    
    #p = np.random.choice(y.shape[0], 1000, replace=False)
    X = df.copy()
    X.drop('RiskPerformance', axis=1, inplace=True)
    
    scalers = {}
    for col_name in df.columns:
        if col_name == 'RiskPerformance':
            continue
        
        scaler = StandardScaler()
        X[col_name] = scaler.fit_transform(df[col_name].values.reshape(-1, 1))
        scalers[col_name] = scaler
            

    X = X.values.astype(float)
    
    return X, y, df.columns, scalers, df
    #return X[p, :], y[p], df.columns, scalers