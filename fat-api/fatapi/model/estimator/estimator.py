import numpy as np

from fatapi.helpers import not_in_range, keep_cols
from typing import Callable, List
class Estimator():
    """
    Abstract class for scaling and encoding data to be passed as a parameter to explainability methods
    
    Parameters
    ----------
    fit(X: numpy.array, columns?: List[int]) -> numpy.array:
        Fits the estimator to X - creates category dict internally
    transform(X: numpy.array, columns?: List[int]) -> numpy.array:
        Generate transformed data from X using columns[] - requires fit
    inverse_transform(X: numpy.array, columns?: List[int]) -> numpy.array:
        Generate original data from transformed X using columns[] - requires fit
    
    Methods
    -------
    fit(X: numpy.array, columns?: List[int]) -> numpy.array:
        Fits the estimator to X - creates category dict internally
    transform(X: numpy.array, columns?: List[int]) -> numpy.array:
        Generate transformed data from X using columns[] - requires fit
    inverse_transform(X: numpy.array, columns?: List[int]) -> numpy.array:
        Generate original data from transformed X using columns[] - requires fit
    encode(X: np.array, columns: List[int]):
        Fits and transforms the data using encoder
        -- If no encoder, returns X
    decode(X: np.array, columns: List[int]):
        Inverse_transforms the data using encoder
        -- If no encoder, returns X
    """
    def __init__(self, fit, transform, inverse_transform) -> None:
        if (callable(fit)):
            self.fit = fit
        else:
            raise ValueError("Invalid argument in __init__: fit is not a function")
        if (callable(transform)):
            self.transform = transform
        else:
            raise ValueError("Invalid argument in __init__: transform is not a function")
        if (callable(inverse_transform)):
            self.inverse_transform = inverse_transform
        else:
            raise ValueError("Invalid argument in __init__: inverse_transform is not a function")

    @property
    def fit(self) -> Callable:
        """
        Sets and changes the fit method of the estimator
        -------
        Callable
        """
        
        return self.fit

    @fit.setter
    def fit(self, fit) -> None:
        if callable(fit):
            self.fit = fit
        else:
            raise ValueError("Invalid argument in fit.setter: fit is not a function")
        
    def encode(self, X: np.array, columns: List[int]=None):
        if not_in_range(X.shape[1], columns):
            raise ValueError("Invalid arguments in encode: Index in parameter columns is out of range")
        X_copy = X
        if columns:
            cols = columns.sort()
            X_rem = keep_cols(X, cols)
        else:
            cols = range(len(X))
        self.fit(X_rem)
        X_rem = self.transform(X_rem)
        j=0
        for i in range(len(X)):
            if i in cols:
                X_copy[:, i] = X_rem[:, j]
                j+=1
        return X_copy
    
    def decode(self, X: np.array=None, columns: List[int]=None):
        if columns and not_in_range(X.shape[1], columns):
            raise ValueError("Invalid arguments in decode: Index in parameter columns is out of range")
        X_copy = X
        if columns:
            cols = columns.sort()
            X_rem = keep_cols(X, cols)
        else:
            cols = range(len(X))
        X_rem = self.inverse_transform(X_rem)
        j=0
        for i in range(len(X)):
            if i in cols:
                X_copy[:, i] = X_rem[:, j]
                j+=1
        return X_copy
        