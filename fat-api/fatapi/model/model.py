
from fatapi.model.estimators import Transformer
from fatapi.data import Data
from fatapi.model import BlackBox
from typing import Callable, List, Optional
from fatapi.helpers import not_in_range, keep_cols, check_type
import numpy as np

class Model(object):
    """
    Abstract class containing methods used involving the model / interacting with the blackbox--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- model
    
    Parameters
    ----------

    data : fatapi.data.Data
        Data object containing data and feature columns
    target? : fatapi.data.Data
        Data object containing target data and target feature columns
    blackbox? : fatapi.model.BlackBox
        BlackBox object (classifier) containing (fit, predict, predict_proba, score) methods
    X_tofit? : List[int]
        List of column indexes for features in data.dataset; if None, all columns in data.dataset
    Y_tofit? : List[int]
        List of column indexes for features in data.dataset; if None, all columns in data.dataset
    predict()? : (X: np.ndarray) -> np.ndarray
        Method for predicting the class label of X
        -- Only required if blackbox not supplied
    predict_proba()? : (X: np.ndarray) -> np.ndarray
        Method for getting the probability of the prediction of X
        -- Only required if blackbox not supplied
    fit()? : (X: np.ndarray, Y?: np.ndarray) -> None
        Method for fitting model to X, Y
        -- Only required if blackbox not supplied
    score()? : (X: np.ndarray, Y: np.ndarray) -> np.ndarray
        Method for calculating a score when predicting X and comparing with Y
        -- Only required if blackbox not supplied
    scaler? : fatapi.model.Transformer
        Scaler for normalisation / scaling numericals features
    encoder? : fatapi.model.Transformer
        Encoder for encoding categorical features
    
    Methods
    -------
    get_fitted_data(): () -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
        Returns the dataset the model has been fitted to; tuple of (X) or (X, Y)
    train() -> (X: np.ndarray, Y: np.ndarray, cols_encode?: List[int], cols_scale?: List[int]) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        Calls self.fit() after scaling/normalising
    encode(): (X: np.ndarray, columns: List[int]) -> np.ndarray
        Fits and transforms the data using encoder
        -- If no encoder, returns X
    decode(): (X: np.ndarray, columns: List[int]) -> np.ndarray
        Inverse_transforms the data using encoder
        -- If no encoder, returns X
    scale(): (X: np.ndarray, columns: List[int]) -> np.ndarray
        Fits and transforms the data using scaler
        -- If no scaler, returns X
    unscale(): (X: np.ndarray, columns: List[int]) -> np.ndarray
        Inverse_transforms the data using scaler
        -- If no scaler, returns X
    """
    def __init__(self, data: Data, target: Data=None, X_tofit: List[int]=[], Y_tofit: List[int]=[], **kwargs) -> None:
        if not ('blackbox' in kwargs) or ('fit' in kwargs and 'predict' in kwargs and 'predict_proba' in kwargs):
            raise ValueError(f"Missing arguments in __init__: [{'' if 'blackbox' in kwargs else 'blackbox'}, {'' if 'fit' in kwargs else 'fit'}, {'' if 'predict' in kwargs else 'predict'}, {'' if 'predict_proba' in kwargs else 'predict_proba'}, {'' if 'score' in kwargs else 'score'}]")
        if 'blackbox' in kwargs:
            self.blackbox = check_type(kwargs.get("blackbox"), "__init__", BlackBox)
            self._fit = self.blackbox.fit
            self._predict = self.blackbox.predict
            self._predict_proba = self.blackbox.predict_proba
            if self.blackbox.score:
                self._score = self.blackbox.score
        else:
            if 'fit' in kwargs:
                self._fit = check_type(kwargs.get("fit"), "__init__", Callable[[np.ndarray, Optional[np.ndarray]], None])
            if 'predict' in kwargs:
                self._predict = check_type(kwargs.get("predict"), "__init__", Callable[[np.ndarray], np.ndarray])
            if 'predict_proba' in kwargs:
                self._predict_proba = check_type(kwargs.get("predict_proba"), "__init__", Callable[[np.ndarray], np.ndarray])
            if 'score' in kwargs:
                self._score = check_type(kwargs.get("score"), "__init__", Callable)
        data = check_type(data, "__init__", Data)
        if data.encoded or (kwargs.get('encoder') and kwargs.get('scaler')):
            d : Data = data
            self.data = d
        else:
            raise ValueError("Invalid argument in __init__: data must be encoded or (scaler, encoder) must be supplied")
        if 'encoder' in kwargs:
            self._encoder = check_type(kwargs.get("encoder"), "__init__", Transformer)
        if 'scaler' in kwargs:
            self._scaler = check_type(kwargs.get("scaler"), "__init__", Transformer)
        if target:
            target = check_type(target, "__init__", Data)
            if (target.n_data == data.n_data):
                if target.encoded or ('encoder' in kwargs and 'scaler' in kwargs):
                    t : Data = target
                    self.target = t
                else:
                    raise ValueError("Invalid argument in __init__: target must be encoded or (scaler, encoder) must be supplied")
            else:
                raise ValueError("Invalid argument in __init__: target is not of same shape[0] as data")
            
        self.X_tofit = X_tofit
        self.Y_tofit = Y_tofit
        if Y_tofit and not target:
            raise ValueError("Invalid arguments in __init__: target not supplied but columns Y_tofit supplied")
        if not target and Y_tofit:
            raise ValueError("Warning in __init__: no target supplied but Y_tofit supplied")
        try:
            self.set_data_tofit()
            self.fitted_data = self.train()
        except:
            raise ValueError("Error in model.train: self.train failed - please provide numpy.arrays to self.fit")

    def get_fitted_data(self):
        return self.fitted_data

    def set_data_tofit(self, dd: Data=None, tt: Data=None):
        if dd is not None:
            self.data.dataset = dd
        if tt is not None:
            self.target.dataset = tt
        if len(self.X_tofit)>0:
            d = keep_cols(self.data.dataset, self.X_tofit)
        if len(self.target.dataset)>0 and len(self.Y_tofit)>0:
            t = keep_cols(self.target.dataset, self.Y_tofit)
        return

    @property
    def fit(self) -> Callable[[np.ndarray, Optional[np.ndarray]], None]:
        """
        Sets and changes the fit method of the model

        """
        return self._fit

    @fit.setter
    def fit(self, fit) -> None:
        self._fit = check_type(fit, "fit.setter", Callable[[np.ndarray, Optional[np.ndarray]], None])
        
    @property
    def predict(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Sets and changes the predict method of the model

        """
        return self._predict

    @predict.setter
    def predict(self, predict) -> None:
        self._predict = check_type(predict, "predict.setter", Callable[[np.ndarray], np.ndarray])
        
    @property
    def predict_proba(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Sets and changes the predict_proba method of the model

        """
        return self._predict_proba

    @predict_proba.setter
    def predict_proba(self, predict_proba) -> None:
        self._predict_proba = check_type(predict_proba, "predict_proba.setter", Callable[[np.ndarray], np.ndarray])
        
    @property
    def score(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """
        Sets and changes the score method of the model

        """
        return self._score

    @score.setter
    def score(self, score) -> None:
        self._score = check_type(score, "score.setter", Callable[[np.ndarray, np.ndarray], np.ndarray])
        
    @property
    def encoder(self) -> Transformer:
        """
        Sets and changes the encoder method of the model
        -------
        Transformer
        """
        
        return self._encoder

    @encoder.setter
    def encoder(self, encoder) -> None:
        self._encoder = check_type(encoder, "encoder.setter", Transformer)
        
    @property
    def scaler(self) -> Transformer:
        """
        Sets and changes the scaler method of the model

        """
        
        return self._scaler

    @scaler.setter
    def scaler(self, scaler) -> None:
        self._scaler = check_type(scaler, "scaler.setter", Transformer)
        
    def encode(self, X: np.ndarray, columns: List[int]=None):
        if not_in_range(X.shape[1], columns):
            raise ValueError("Invalid arguments in encode: Index in parameter columns is out of range")
        if self.encoder:
            X_copy = X
            if columns:
                cols = columns
                cols.sort()
                X_rem = keep_cols(X, cols)
            else:
                cols = range(X.shape[1])
                X_rem = X
            X_rem = np.squeeze(X_rem)
            if X_rem.ndim < 2:
                X_rem = X_rem.reshape(-1, 1)
            self.encoder.fit(X_rem)
            X_trans = self.encoder.transform(X_rem)
            if type(X_trans) == np.ndarray:
                X_rem = X_trans
            else:
                X_rem = X_trans.toarray()
            j=0
            for i in range(X.shape[1]):
                if i in cols:
                    X_copy[:, i] = X_rem[:, j]
                    j+=1
            return X_copy
        else:
            raise ValueError(f"Invalid call to encode: encoder is required")
    
    def decode(self, X: np.ndarray=None, columns: List[int]=None):
        if columns and not_in_range(X.shape[1], columns):
            raise ValueError("Invalid arguments in decode: Index in parameter columns is out of range")
        if self.encoder:
            X_copy = X
            if columns:
                cols = columns
                cols.sort()
                X_rem = keep_cols(X, cols)
            else:
                cols = range(X.shape[1])
            X_rem = np.squeeze(X_rem)
            if X_rem.ndim < 2:
                X_rem = X_rem.reshape(-1, 1)
            X_trans = self.encoder.inverse_transform(X_rem)
            if type(X_trans) == np.ndarray:
                X_rem = X_trans
            else:
                X_rem = X_trans.toarray()
            j=0
            for i in range(X.shape[1]):
                if i in cols:
                    X_copy[:, i] = X_rem[:, j]
                    j+=1
            return X_copy
        else:
            raise ValueError(f"Invalid call to decode: encoder is required")
    
    def scale(self, X: np.ndarray, columns: List[int]=None):
        #print(f"SCALE: X: {X}")
        if not_in_range(X.shape[1], columns):
            raise ValueError("Invalid arguments in scale: Index in parameter columns is out of range")
        if self.scaler:
            X_copy = X
            X_rem = X_copy
            if columns:
                cols = columns
                cols.sort()
                X_rem = keep_cols(X, cols)
            else:
                cols = range(X.shape[1])
            X_rem = np.squeeze(X_rem)
            #print(f"SCALE: Xrem: {X_rem}")
            if X_rem.ndim < 2:
                X_rem = X_rem.reshape(-1, 1)
            #print(f"SCALE: Xrem: {X_rem}")
            self.scaler.fit(X_rem)
            X_trans = self.scaler.transform(X_rem)
            print(f"XTRAN: {self.scaler}")
            if type(X_trans) == np.ndarray:
                X_rem = X_trans
            else:
                X_rem = X_trans.toarray()
            #print(f"SCALE: X_trans: {X_trans}")
            print(f"SCALE: X_transRem: {X_rem}")
            j=0
            for i in range(X.shape[1]):
                if i in cols:
                    X_copy[:, i] = X_rem[:, j]
                    j+=1
            return X_copy
        else:
            raise ValueError(f"Invalid call to scale: scaler is required")
    
    def unscale(self, X: np.ndarray=None, columns: List[int]=None):
        if columns and not_in_range(X.shape[1], columns):
            raise ValueError("Invalid arguments in unscale: Index in parameter columns is out of range")
        if self.scaler:
            X_copy = X
            if columns:
                cols = columns
                cols.sort()
                X_rem = keep_cols(X, cols)
            else:
                cols = range(X.shape[1])
            X_rem = np.squeeze(X_rem)
            if X_rem.ndim < 2:
                X_rem = X_rem.reshape(-1, 1)
            X_trans = self.scaler.inverse_transform(X_rem)
            if type(X_trans) == np.ndarray:
                X_rem = X_trans
            else:
                X_rem = X_trans.toarray()
            j=0
            for i in range(X.shape[1]):
                if i in cols:
                    X_copy[:, i] = X_rem[:, j]
                    j+=1
            return X_copy
        else:
            raise ValueError(f"Invalid call to unscale: scaler is required")
    
    def train(self, X: np.ndarray=[], Y: np.ndarray=[], cols_encode: List[int]=None, cols_scale: List[int]=None):
        if len(Y)>0 and len(X)<1:
            raise ValueError("Invalid argument to model.train: X not provided - please provide only X or X and Y or nothing")
        else:
            if len(X)>0 and len(Y)>0:
                self.fit(X,Y)
                return (X,Y)
            if len(X)>0:
                self.fit(X)
                return (X)
            else:
                X_ = self.data.dataset
                if self.target is not None:
                    Y_ = self.target.dataset
                print(f"TRAIN X: {X_} Y: {Y_}")
                if not self.data.encoded:
                    if cols_encode is not None:
                        X_ = self.encode(self.data.dataset, cols_encode)
                    else:
                        if len(self.data.categoricals)>0:
                            X_ = self.encode(self.data.dataset, self.data.categoricals)
                    if cols_scale is not None:
                        X_ = self.scale(self.data.dataset, cols_scale)
                    else:
                        if len(self.data.numericals)>0:
                            X_ = self.scale(self.data.dataset, self.data.numericals)
                print(f"TRAINED X: {X_}")
                if not self.target.encoded and self.target is not None:
                    if cols_encode is not None:
                        Y_ = self.encode(self.target.dataset, cols_encode)
                    else:
                        if len(self.target.categoricals)>0:
                            Y_ = self.encode(self.target.dataset, self.target.categoricals)
                    if cols_scale is not None:
                        Y_ = self.scale(self.target.dataset, cols_scale)
                    else:
                        if len(self.target.numericals)>0:
                            Y_ = self.scale(self.target.dataset, self.target.numericals)
                print(f"TRAINED Y: {Y_}")
                if self.target is not None:
                    self.fit(X_,Y_)
                    print(f"Classification accuracy on Training Data: {self.score(X_,Y_)}")
                    return (X_,Y_)
                else:
                    self.fit(X_)
                    print(f"Classification accuracy on Training Data: {self.score(X_)}")
                    return (X_)
                
    
    def __str__(self):
        return f"Data: {self.data}, Target: {self.target}, X_tofit: {self.X_tofit}, Y_tofit: {self.Y_tofit}"