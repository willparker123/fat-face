import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def encode_pandas_scaled(df):
    initial_columns = df.columns
    for column in initial_columns:
        print(column)
        if min(df[column]) < 0:
            unique = np.unique(df[column])
            neg_unique = unique[unique<0]
            for item in neg_unique:
                col_name = column + '_' + str(item)
                df[col_name] = 0
            for idx, val in df[column].iteritems():
                if val in neg_unique:
                    df.loc[idx, column] = 0
                    col_name = column + '_' + str(val)
                    df.loc[idx, col_name] = 1
    return df

def encode_pandas(df):
    initial_columns = df.columns
    newdf = df.copy()
    scalers = {}

    for col_idx, column in enumerate(initial_columns):
        print(column)
        if min(df[column]) < 0:
            unique = np.unique(df[column])
            neg_unique = unique[unique<0]
            counter = 1
            for item in neg_unique:
                col_name = column + '_' + str(item)
                newdf.insert(loc=col_idx+counter,
                             column=col_name,
                             value=0)
                counter += 1
            for idx, val in df[column].iteritems():
                if val in neg_unique:
                    newdf.loc[idx, column] = 0
                    col_name = column + '_' + str(val)
                    newdf.loc[idx, col_name] = 1
        scaler = StandardScaler()
        newdf[column] = scaler.fit_transform(newdf[column].values.reshape(-1, 1))
        scalers[column] = scaler
    return newdf, scalers

def get_data():    
    data_dir='./data//' 
    df=pd.read_csv(data_dir+"heloc_dataset_v1.csv")
    y = df['RiskPerformance'].copy()
    y.replace('Good', 1, inplace=True)
    y.replace('Bad', 0, inplace=True)
    y = y.values
    
    inputs=df[df.columns[1:]]
    newdf, scalers = encode_pandas(inputs)
    X = newdf.values.astype(float)
    return X, newdf, df, y, scalers

#X, newdf, df, y, scalers = get_data()