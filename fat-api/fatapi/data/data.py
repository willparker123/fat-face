import numpy as np
from typing import List
from fatapi.helpers import not_in_range

class Data():
    """
    Abstract class for numpy.array dataset and essential column indexes
    
    Parameters
    ----------
    dataset : np.array()
        Numpy.Array() containing rows of datapoints, where columns are the features
    categoricals? : List[int]
        List of column indexes of categorical features (including target features)
        -- Default value is no columns
    numericals? : List[int]
        List of column indexes of numerical features (including target features)
        -- Default value is all columns of dataset
    isEncoded? : boolean
        Bool showing whether the dataset has already been encoded/normalised
    
    Methods
    -------
    """
    
    def __init__(self, 
                dataset: np.array,
                dtype: str="data",
                categoricals: List[int]=[], 
                numericals: List[int]=[], 
                isEncoded: bool=True):
        if len(dataset.shape) < 2:
            raise ValueError("Invalid argument in __init__: dataset must be at least 2-dimensional")
        self.categoricals = []
        self.numericals = range(dataset.shape[1])
        if dtype:
            if dtype=="data" or dtype=="target":
                if dtype=="data":
                    self.numericals = range(dataset.shape[1])
                else:
                    self.categoricals = range(dataset.shape[1])
            else:
                raise ValueError("Invalid arguments in __init__: type must be 'data' or 'target'")
        if categoricals:
            self.categoricals = categoricals
        if numericals:
            self.numericals = numericals
        self.dataset = dataset
        self.isEncoded = isEncoded
        if not_in_range(self.dataset.shape[1], categoricals):
            raise ValueError("Invalid arguments in __init__: Index in categoricals is out of range")
        if not_in_range(self.dataset.shape[1], numericals):
            raise ValueError("Invalid arguments in __init__: Index in numericals is out of range")
        if self.dataset.shape[1] <= 0:
            raise ValueError("Invalid arguments in __init__: dataset has no rows / datapoints")
        
    def N_data(self, subset=None):
        return self.dataset.shape[0]
    
    def N_features(self):
        return self.dataset.shape[1]
    
    def __str__(self):
        return f"Data: {self.dataset}, Target: {self.target}, Categoricals: {self.categoricals}, Numericals: {self.numericals}, IsEncoded: {self.isEncoded}"