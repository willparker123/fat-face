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
    
    def encode(self, X: np.ndarray, column_groups: List[List[int]]=None):
        if column_groups is not None and not_in_range(X.shape[1], column_groups):
            raise ValueError("Invalid arguments in scale: Index in parameter columns is out of range")
        X_copy = X
        X_copys = np.zeros((1, X.shape[1]))
        X_rems = []
        if column_groups is not None:
            cols = column_groups
            cols.sort()
            for index, col in enumerate(cols):
                if sorted(col) == list(range(min(col), max(col)+1)) and col[0] >= 0:
                    if index==len(cols)-1 or max(cols[index]) < min(cols[index+1]):
                        X_rems.append(keep_cols(X, col))
                    else:
                        raise ValueError(f"Error in scale(): column {col} in column_groups has max value >= min value in next column {cols[index+1]}")
                else:
                    raise ValueError(f"Error in scale(): column {col} in column_groups must be consecutive, all i > 0")
        else:
            cols = [range(X.shape[1])]
        X_transs = []
        for X_rem in X_rems:
            if X_rem.ndim < 2:
                X_rem = X_rem.reshape(-1, 1)
            self.fit(X_rem)
            X_trans = self.transform(X_rem)
            try:
                X_trans = self.transform(X_rem)
            except:
                raise ValueError(f"Error in scale(): cannot transform X_rem")
            if type(X_trans) == np.ndarray:
                X_transs.append(X_trans)
            else:
                try:
                    X_transs.append(X_trans.toarray())
                except:
                    raise ValueError(f"Error in scale(): cannot convert X_trans of type {type(X_trans)} to np.ndarray")
        j = 0
        i = 0
        while i < X.shape[1]:
            if any(min(cols[n])==i for n in range(len(cols))):
                X_rem_temp = X_transs[j]
                if X_rem_temp.ndim < 2:
                    # may be a problem
                    X_rem_temp = X_rem_temp.reshape(-1, 1)
                if i == 0:
                    X_copys = X_rem_temp
                else:
                    X_copys = np.column_stack((X_copys,X_rem_temp))
                j+=1
                i += X_rem_temp.shape[1]
            else:
                if i == 0:
                    X_copys = X_copy[:, i]
                else:
                    X_copys = np.column_stack((X_copys,X_copy[:, i]))
                i += 1
            if not i == X_copys.shape[1]:
                print(f"Message in scale(): scaled column [{i}] now at index [{X_copys.shape[1]}]")
        return X_copys
    
    def decode(self, X: np.ndarray=None, column_groups: List[List[int]]=None):
        if column_groups is not None and not_in_range(X.shape[1], column_groups):
            raise ValueError("Invalid arguments in unscale: Index in parameter columns is out of range")
        X_copy = X
        X_copys = np.zeros((0, X.shape[1]))
        X_rems = []
        if column_groups is not None:
            cols = column_groups
            cols.sort()
            for index, col in enumerate(cols):
                if sorted(col) == list(range(min(col), max(col)+1)) and col[0] >= 0:
                    if index==len(cols)-1 or max(cols[index]) < min(cols[index+1]):
                        X_rems.append(keep_cols(X, col))
                    else:
                        raise ValueError(f"Error in unscale(): column {col} in column_groups has max value >= min value in next column {cols[index+1]}")
                else:
                    raise ValueError(f"Error in unscale(): column {col} in column_groups must be consecutive, all i > 0")
        else:
            cols = [range(X.shape[1])]
        X_transs = []
        for X_rem in X_rems:
            if X_rem.ndim < 2:
                X_rem = X_rem.reshape(-1, 1)
            try:
                X_trans = self.inverse_transform(X_rem)
            except:
                raise ValueError(f"Error in unscale(): cannot inverse_transform X_rem")
            if type(X_trans) == np.ndarray:
                X_transs.append(X_trans)
            else:
                try:
                    X_transs.append(X_trans.toarray())
                except:
                    raise ValueError(f"Error in unscale(): cannot convert X_trans of type {type(X_trans)} to np.ndarray")
        j = 0
        i = 0
        while i < X.shape[1]:
            if any(min(cols[n])==i for n in range(len(cols))):
                X_rem_temp = X_transs[j]
                if X_rem_temp.ndim < 2:
                    # may be a problem
                    X_rem_temp = X_rem_temp.reshape(-1, 1)
                if i == 0:
                    X_copys = X_rem_temp
                else:
                    X_copys = np.column_stack((X_copys,X_rem_temp))
                j+=1
                i += X_rem_temp.shape[1]
            else:
                if i == 0:
                    X_copys = X_copy[:, i]
                else:
                    X_copys = np.column_stack((X_copys,X_copy[:, i]))
                i += 1
            if not i == X_copys.shape[1]:
                print(f"Message in unscale(): scaled column [{i}] now at index [{X_copys.shape[1]}]")
        return X_copys