import numpy as np

from fatapi.helpers import not_in_range, keep_cols, check_type
from typing import Callable, List

class Transformer(object):
    """
    Abstract class for scaling and encoding data to be passed as a parameter to explainability methods
    
    Parameters
    ----------
    fit()? : (X: np.ndarray, columns: List[int])
        Fits the transformer to X - creates category dict internally
    transform() : (X: np.ndarray) -> np.ndarray
        Generate transformed data from X using columns[] - requires fit
    inverse_transform() : (X: np.ndarray) -> np.ndarray
        Generate original data from transformed X using columns[] - requires fit
    
    Methods
    -------
    fit() : (X: np.ndarray, columns: List[int])
        Fits the transformer to X - creates category dict internally
    transform() : (X: np.ndarray) -> np.ndarray
        Generate transformed data from X using columns[] - requires fit
    inverse_transform() : (X: np.ndarray) -> np.ndarray
        Generate original data from transformed X using columns[] - requires fit
    encode() : (X: np.ndarray, columns?: List[int]) -> np.ndarray
        Fits and transforms the data using encoder
        -- If no encoder, returns X
    decode() : (X: np.ndarray, columns?: List[int]) -> np.ndarray
        Inverse_transforms the data using encoder
        -- If no encoder, returns X
    """
    def __init__(self, transform, inverse_transform, fit=None) -> None:
        self.fit = self.base_fit()
        if fit:
            self.fit = check_type(fit, "__init__", Callable[[np.ndarray, List[int]], None])
        self.transform = check_type(transform, "__init__", Callable[[np.ndarray], np.ndarray])
        self.inverse_transform = check_type(inverse_transform, "__init__", Callable[[np.ndarray], np.ndarray])

    def base_fit(self, X: np.ndarray, columns: List[int]) -> None:
        self.X = X
        self.columns = columns
        return
        
    def encode(self, X: np.ndarray, columns: List[int]=None) -> np.ndarray:
        if not_in_range(X.shape[1], columns):
            raise ValueError("Invalid arguments in encode: Index in parameter columns is out of range")
        X_copy = X
        if columns:
            cols = columns.sort()
            X_rem = keep_cols(X, cols)
        elif self.columns:
            cols = self.columns
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
    
    def decode(self, X: np.ndarray=None, columns: List[int]=None) -> np.ndarray:
        if columns and not_in_range(X.shape[1], columns):
            raise ValueError("Invalid arguments in decode: Index in parameter columns is out of range")
        X_copy = X
        if columns:
            cols = columns.sort()
            X_rem = keep_cols(X, cols)
        elif self.columns:
            cols = self.columns
        else:
            cols = range(len(X))
        X_rem = self.inverse_transform(X_rem)
        j=0
        for i in range(len(X)):
            if i in cols:
                X_copy[:, i] = X_rem[:, j]
                j+=1
        return X_copy