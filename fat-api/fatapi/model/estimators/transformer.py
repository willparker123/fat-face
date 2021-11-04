import numpy as np

from fatapi.helpers import not_in_range, keep_cols, check_type
from typing import Callable, List, Optional

class Transformer(object):
    """
    Abstract class for scaling and encoding data to be passed as a parameter to explainability methods
    
    Parameters
    ----------
    transformer : attr(fit, transform, inverse_transform)
        Object which has the methods above used to encode / normalise data
        -- Only required if fit, transform not supplied
    fit()? : (X: np.ndarray, columns: List[int])
        Fits the transformer to X - creates category dict internally
    transform()? : (X: np.ndarray) -> np.ndarray
        Generate transformed data from X using columns[] - requires fit
        -- Only required if transformer
    inverse_transform()? : (X: np.ndarray) -> np.ndarray
        Generate original data from transformed X using columns[] - requires fit
        -- Only required if transformer not supplied
    
    Methods
    -------
    fit() : (X: np.ndarray, columns: List[int])
        Fits the transformer to X - creates category dict internally
    transform() : (X: np.ndarray) -> np.ndarray
        Generate transformed data from X using columns[] - requires fit
    inverse_transform() : (X: np.ndarray) -> np.ndarray
        Generate original data from transformed X using columns[] - requires fit to X
        -- Default returns fitted X
    encode() : (X: np.ndarray, columns?: List[int]) -> np.ndarray
        Fits and transforms the data using encoder
        -- If no encoder, returns X
    decode() : (X: np.ndarray, columns?: List[int]) -> np.ndarray
        Inverse_transforms the data using encoder
        -- If no encoder, returns X
    """
    def __init__(self, transform: Callable[[np.ndarray], np.ndarray]=None, inverse_transform: Callable[[np.ndarray], np.ndarray]=None, fit: Callable[[np.ndarray, List[int]], None]=None, **kwargs) -> None:
        self.fit = self.base_fit
        self.X = None
        if not ('transformer' in kwargs or (transform is not None and inverse_transform is not None)):
            raise ValueError(f"Missing arguments in __init__: must provide 'transformer' or 'transform' and 'inverse_transform'")
        if 'transformer' in kwargs:
            self.transformer = kwargs.get('transformer')
            if hasattr(kwargs.get('transformer'), "transform"):
                self._transform = check_type(self.transformer.transform, "__init__", Callable)
            else:
                raise ValueError("Invalid argument in __init__: transformer does not have function transform")
            if hasattr(kwargs.get('transformer'), "fit"):
                self._fit = check_type(self.transformer.fit, "__init__", Callable)
            else:
                raise ValueError("Invalid argument in __init__: transformer does not have function fit")
            if hasattr(kwargs.get('transformer'), "inverse_transform"):
                self._inverse_transform = check_type(self.transformer.inverse_transform, "__init__", Callable)
            else:
                self._inverse_transform = lambda **kwargs: self.X
                print("Invalid argument in __init__: transformer does not have function inverse_transform - default returns fitted X")
        if fit:
            self._fit = check_type(fit, "__init__", Callable[[np.ndarray, List[int]], None])
        if transform:
            self._transform = check_type(transform, "__init__", Callable[[np.ndarray], np.ndarray])
        if inverse_transform:
            self._inverse_transform = check_type(inverse_transform, "__init__", Callable[[np.ndarray], np.ndarray])

    @property
    def fit(self) -> Callable[[np.ndarray, Optional[np.ndarray]], None]:
        """
        Sets and changes the fit method of the transformer

        """
        return self.base_fit

    @fit.setter
    def fit(self, fit) -> None:
        self._fit = check_type(fit, "fit.setter", Callable[[np.ndarray, Optional[np.ndarray]], None])
        
    @property
    def transform(self) -> Callable[[np.ndarray, Optional[List[int]]], np.ndarray]:
        """
        Sets and changes the transform method of the transformer

        """
        return self._transform

    @transform.setter
    def transform(self, transform) -> None:
        self._transform = check_type(transform, "transform.setter", Callable[[np.ndarray, Optional[List[int]]], np.ndarray])
        
    @property
    def inverse_transform(self) -> Callable[[np.ndarray, Optional[List[int]]], np.ndarray]:
        """
        Sets and changes the inverse_transform method of the transformer

        """
        return self._inverse_transform

    @inverse_transform.setter
    def inverse_transform(self, inverse_transform) -> None:
        self._inverse_transform = check_type(inverse_transform, "inverse_transform.setter", Callable[[np.ndarray, Optional[List[int]]], np.ndarray])
        
    def base_fit(self, X: np.ndarray, columns: List[int]=[]) -> None:
        self.X = X
        self.columns = columns
        if self._fit is not None:
            if len(columns)>0:
                self._fit(X, columns)
            else:
                self._fit(X)
        return 
        
    def encode(self, X: np.ndarray, columns: List[int]=None) -> np.ndarray:
        if not_in_range(X.shape[1], columns):
            raise ValueError("Invalid arguments in encode: Index in parameter columns is out of range")
        self.X = X
        X_copy = X
        if columns:
            cols = columns.sort()
            X_rem = keep_cols(X, cols)
        elif self.columns:
            cols = self.columns
        else:
            cols = range(len(X))
        self.fit(X_rem, cols)
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
        if X_rem is None:
            raise ValueError("Error in decode: inverse_transform required fitted X - self.X == None")
        j=0
        for i in range(len(X)):
            if i in cols:
                X_copy[:, i] = X_rem[:, j]
                j+=1
        return X_copy