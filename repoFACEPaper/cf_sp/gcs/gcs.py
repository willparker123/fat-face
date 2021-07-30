# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:24:15 2019

@author: rp13102
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def get_gcs():
    df = pd.read_csv(r'C:\Users\rp13102\Documents\GitHub\experiment\ghent\data\german_credit_data.csv')
    df.drop(df.columns[0], axis=1, inplace=True)
    df.fillna('na', inplace=True)
    codes = {}
    scalers = {}
    for col_name in df.columns:
        if(df[col_name].dtype == 'object'):
            df[col_name]= df[col_name].astype('category')
            labelencoder = LabelEncoder()            
            df[col_name] = labelencoder.fit_transform(df[col_name])
            codes[col_name] = labelencoder
        if df[col_name].dtype == 'int64':
            if col_name == 'Job':
                continue
            scaler = StandardScaler()
            df[col_name] = scaler.fit_transform(df[col_name].values.reshape(-1, 1))
            scalers[col_name] = scaler
            
    y = df['Risk'].values.astype(int)
    X = df.copy()
    X.drop(['Risk'], axis=1, inplace=True)
    X = X.values.astype(float)
    
    return X, y, df.columns, codes, scalers