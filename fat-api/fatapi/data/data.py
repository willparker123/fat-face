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
    ----------
    get_rows_as_data() : (row_indicies: List[int])
        Get rows with indicies given as argument from dataset and return as a Data object
    """

    def __init__(self, 
                dataset: np.array,
                categoricals: List[int]=[], 
                numericals: List[int]=[], 
                isEncoded: bool=True,
                dtype: str="data"):
        n_features = 1
        if len(dataset.shape) > 1:
            n_features = dataset.shape[1]
        self.n_features = n_features
        self.n_data = dataset.shape[0]
        if n_features == 1:
            dataset = np.reshape(dataset,(self.n_data,1))
        self.categoricals: List[int] = []
        self.numericals: List[int] = list(range(n_features))
        if dtype:
            if dtype=="data" or dtype=="target":
                if dtype=="data":
                    self.numericals = list(range(n_features))
                else:
                    self.categoricals = list(range(n_features))
            else:
                raise ValueError("Invalid arguments in __init__: type must be 'data' or 'target'")
        if categoricals:
            self.categoricals = categoricals
        if numericals:
            self.numericals = numericals
        self.dataset = dataset
        self.isEncoded = isEncoded
        if len(self.categoricals) > 0 and not_in_range(self.n_features, self.categoricals):
            raise ValueError("Invalid arguments in __init__: Index in categoricals is out of range")
        if len(self.numericals) > 0 and not_in_range(self.n_features, self.numericals):
            raise ValueError("Invalid arguments in __init__: Index in numericals is out of range")
        if self.n_features <= 0:
            raise ValueError("Invalid arguments in __init__: dataset has no rows / datapoints")
    
    def get_rows_as_data(self, row_indicies: List[int]):
        return Data(self.dataset[row_indicies, :], self.categoricals, self.numericals, self.isEncoded)

    @property
    def isEncoded(self) -> bool:
        """
        Sets and changes isEncoded
        -------
        Callable
        """
        
        return self.isEncoded

    @isEncoded.setter
    def isEncoded(self, isEncoded) -> None:
        if not type(isEncoded)==bool:
            raise ValueError("Invalid argument in isEncoded.setter: isEncoded is not of type bool")   
        else:
            self.isEncoded = isEncoded

    @property
    def dtype(self) -> bool:
        """
        Sets and changes dtype
        -------
        Callable
        """
        
        return self.dtype

    @dtype.setter
    def dtype(self, dtype) -> None:
        if not type(dtype)==str:
            raise ValueError("Invalid argument in dtype.setter: dtype is not of type string")   
        if not (dtype=="data" or dtype=="target"):
            raise ValueError("Invalid argument in dtype.setter: dtype is not 'data' or 'target'")   
        else:
            self.dtype = dtype

    @property
    def categoricals(self) -> List[int]:
        """
        Sets and changes categoricals
        -------
        Callable
        """
        
        return self.categoricals

    @categoricals.setter
    def categoricals(self, categoricals) -> None:
        if not type(categoricals)==List[int]:
            raise ValueError("Invalid argument in categoricals.setter: categoricals is not of type List[int]")   
        if len(self.categoricals) > 0 and not_in_range(self.n_features, self.categoricals):
            raise ValueError("Invalid argument in categoricals.setter: Index in categoricals is out of range")
        else:
            self.categoricals = categoricals

    @property
    def numericals(self) -> List[int]:
        """
        Sets and changes numericals
        -------
        Callable
        """
        
        return self.numericals

    @numericals.setter
    def numericals(self, numericals) -> None:
        if not type(numericals)==List[int]:
            raise ValueError("Invalid argument in numericals.setter: numericals is not of type List[int]")   
        if len(self.numericals) > 0 and not_in_range(self.n_features, self.numericals):
            raise ValueError("Invalid argument in numericals.setter: Index in numericals is out of range")
        else:
            self.numericals = numericals

    def __str__(self):
        return f"Data: {self.dataset}, Target: {self.target}, Categoricals: {self.categoricals}, Numericals: {self.numericals}, IsEncoded: {self.isEncoded}"