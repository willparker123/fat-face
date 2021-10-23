import numpy as np
from typing import List
from fatapi.helpers import not_in_range, check_type

class Data():
    """
    Abstract class for np.ndarray dataset and essential column indexes
    
    Parameters
    ----------
    dataset : np.ndarray
        Numpy.Array() containing rows of datapoints, where columns are the features
    categoricals? : List[int]
        List of column indexes of categorical features (including target features)
        -- Default value is no columns
    numericals? : List[int]
        List of column indexes of numerical features (including target features)
        -- Default value is all columns of dataset
    encoded? : boolean
        Bool showing whether the dataset has already been encoded/normalised
    dtype? : str
        Variable setting the default categoricals/numericals columns for "data" or "target"
        -- Default value is "data"

    Methods
    ----------
    get_rows_as_data() : (row_indicies: List[int])
        Get rows with indicies given as argument from dataset and return as a Data object
    """

    def __init__(self, 
                dataset: np.ndarray,
                categoricals: List[int]=[], 
                numericals: List[int]=[], 
                encoded: bool=False,
                dtype: str="data"):
        n_features = 1
        if len(dataset.shape) > 1:
            n_features = dataset.shape[1]
        self.n_features = n_features
        self.n_data = dataset.shape[0]
        if n_features == 1:
            dataset = np.reshape(dataset,(self.n_data,1))
        self._categoricals: List[int] = []
        self._numericals: List[int] = list(range(n_features))
        if dtype:
            if dtype=="data" or dtype=="target":
                if dtype=="data":
                    self._numericals = list(range(n_features))
                else:
                    self._categoricals = list(range(n_features))
            else:
                raise ValueError("Invalid arguments in __init__: type must be 'data' or 'target'")
        if categoricals:
            self._categoricals = categoricals
        if numericals:
            self._numericals = numericals
        self.dataset = dataset
        self._encoded = encoded
        if len(self._categoricals) > 0 and not_in_range(self.n_features, self._categoricals):
            raise ValueError("Invalid arguments in __init__: Index in categoricals is out of range")
        if len(self._numericals) > 0 and not_in_range(self.n_features, self._numericals):
            raise ValueError("Invalid arguments in __init__: Index in numericals is out of range")
        if self.n_features <= 0:
            raise ValueError("Invalid arguments in __init__: dataset has no rows / datapoints")
    
    def get_rows_as_data(self, row_indicies: List[int]):
        return Data(self.dataset[row_indicies, :], self.categoricals, self.numericals, self.encoded)

    @property
    def encoded(self) -> bool:
        """
        Sets and changes encoded

        """
        
        return self._encoded

    @encoded.setter
    def encoded(self, encoded) -> None:
        self._encoded = check_type(encoded, "encoded.setter", bool)

    @property
    def categoricals(self) -> List[int]:
        """
        Sets and changes categoricals

        """
        
        return self._categoricals

    @categoricals.setter
    def categoricals(self, categoricals) -> None:
        if len(self._categoricals) > 0 and not_in_range(self.n_features, self._categoricals):
            raise ValueError("Invalid argument in categoricals.setter: Index in categoricals is out of range")        
        self._categoricals = check_type(categoricals, "categoricals.setter", List[int])
        

    @property
    def numericals(self) -> List[int]:
        """
        Sets and changes numericals

        """
        
        return self._numericals

    @numericals.setter
    def numericals(self, numericals) -> None:
        if len(self._numericals) > 0 and not_in_range(self.n_features, self._numericals):
            raise ValueError("Invalid argument in numericals.setter: Index in numericals is out of range")
        self._numericals = check_type(numericals, "numericals.setter", List[int])

    def __str__(self):
        return f"Data: {self.dataset}, Categoricals: {self.categoricals}, Numericals: {self.numericals}, IsEncoded: {self.encoded}"