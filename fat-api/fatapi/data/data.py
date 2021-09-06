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
    
    def __init__(
        self,
        dataset: np.array,
        categoricals : List[int]=None,
        numericals : List[int]=None,
        isEncoded : bool=True,
        type : str=None,
        #**kwargs
    ) -> None:
        #if kwargs.get('dataset'):
        #    self.dataset = kwargs.get('dataset')
        #    self.train_dataset = kwargs.get('dataset')
        #    self.test_dataset = kwargs.get('dataset')
        #if kwargs.get('train_dataset') and kwargs.get('test_dataset'):
        #    num_colsa = kwargs.get('train_dataset').shape[1]
        #    num_colsb = kwargs.get('test_dataset').shape[1]
        #    if (num_colsa!=num_colsb):
        #        raise ValueError(f"Missing arguments in __init__: {'' if kwargs.get('dataset') else 'dataset'} {'' if kwargs.get('train_dataset') else 'train_dataset'} {'' if kwargs.get('test_dataset') else 'test_dataset'}")
        #    else:
        #        if not kwargs.get('dataset'):
        #            self.dataset = np.concatenate(kwargs.get('train_dataset'), kwargs.get('test_dataset'))
        #       self.train_dataset = kwargs.get('train_dataset')
        #       self.test_dataset = kwargs.get('test_dataset')
        #if not kwargs.get('dataset') and not (kwargs.get('train_dataset') and kwargs.get('test_dataset')):
        #    raise ValueError(f"Missing arguments in __init__: {'' if kwargs.get('dataset') else 'dataset'} {'' if kwargs.get('train_dataset') else 'train_dataset'} {'' if kwargs.get('test_dataset') else 'test_dataset'}")
        #if not_in_range(self.dataset.shape[1], self.features):
        #    raise ValueError("Index in features is out of range")
        #if not_in_range(self.dataset.shape[1], self.targets):
        #    raise ValueError("Index in targets is out of range")
        if not_in_range(dataset.shape[1], self.categoricals):
            raise ValueError("Invalid arguments in __init__: Index in categoricals is out of range")
        if not_in_range(dataset.shape[1], self.numericals):
            raise ValueError("Invalid arguments in __init__: Index in numericals is out of range")
        if dataset.shape[1] <= 0:
            raise ValueError("Invalid arguments in __init__: dataset has no rows / datapoints")
        else:
            if categoricals:
                self.categoricals = categoricals
            else:
                self.categoricals = []
            if numericals:
                self.numericals = numericals
            else:
                self.numericals = range(dataset.shape[1])
            if type:
                if type=="data" or type=="target":
                    if type=="data":
                        self.numericals = range(dataset.shape[1])
                    else:
                        self.categoricals = range(dataset.shape[1])
                else:
                    raise ValueError("Invalid arguments in __init__: type must be 'data' or 'target'")
            
                
            self.dataset = dataset
            self.isEncoded = isEncoded
        
    def N_data(self, subset=None):
        return self.dataset.shape[0]
    
    def N_features(self):
        return self.dataset.shape[1]
    
    def __str__(self):
        return f"Data: {self.dataset}, Target: {self.target}, Categoricals: {self.categoricals}, Numericals: {self.numericals}, IsEncoded: {self.isEncoded}"