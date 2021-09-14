
from fatapi.model.estimators import Transformer
from fatapi.data import Data
from fatapi.model import BlackBox
from typing import Callable, List
from fatapi.helpers import not_in_range, keep_cols, check_type
import numpy as np

class Model(object):
    """
    Abstract class containing methods used involving the model / interacting with the blackbox model
    
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
    predict()? : (X: np.ndarray) ->np.ndarray()
        Method for predicting the class label of X
        -- Only required if blackbox not supplied
    predict_proba()? : (X: np.ndarray) -> np.ndarray()
        Method for predicting the probability of the prediction of X
        -- Only required if blackbox not supplied
    fit()? : (X: np.ndarray, Y?: np.ndarray) -> np.ndarray()
        Method for fitting model to X, Y
        -- Only required if blackbox not supplied
    score()? : (X: np.ndarray, Y?: np.ndarray) -> np.ndarray()
        Method for calculating a score when predicting X and comparing with Y
        -- Only required if blackbox not supplied
    scaler? : fatapi.model.Transformer
        Scaler for normalisation / scaling numericals features
    encoder? : fatapi.model.Transformer
        Encoder for encoding categorical features
    
    Methods
    -------
    get_fitted_data(): () -> (numpy.array, numpy.array?)
        Returns the dataset the model has been fitted to; tuple of (X) or (X, Y)
    train(X: np.ndarray, Y: np.ndarray):
        Calls self.fit() after scaling/normalising
    encode(X: np.ndarray, columns: List[int]):
        Fits and transforms the data using encoder
        -- If no encoder, returns X
    decode(X: np.ndarray, columns: List[int]):
        Inverse_transforms the data using encoder
        -- If no encoder, returns X
    scale(X: np.ndarray, columns: List[int]):
        Fits and transforms the data using scaler
        -- If no scaler, returns X
    unscale(X: np.ndarray, columns: List[int]):
        Inverse_transforms the data using scaler
        -- If no scaler, returns X
    """
    def __init__(self, data: Data, target: Data=None, X_tofit: List[int]=[], Y_tofit: List[int]=[], **kwargs) -> None:
        if not kwargs.get("blackbox") and not (kwargs.get('fit') and kwargs.get('predict') and kwargs.get('predict_proba') and kwargs.get('score')):
            raise ValueError(f"Missing arguments in __init__: [{'' if kwargs.get('blackbox') else 'blackbox'}, {'' if kwargs.get('fit') else 'fit'}, {'' if kwargs.get('predict') else 'predict'}, {'' if kwargs.get('predict_proba') else 'predict_proba'}, {'' if kwargs.get('score') else 'score'}]")
        if kwargs.get("blackbox"):
            self.blackbox = check_type(kwargs.get("blackbox"), BlackBox, "__init__")
            self._fit = self.blackbox.fit
            self._predict = self.blackbox.predict
            self._predict_proba = self.blackbox.predict_proba
            self._score = self.blackbox.score
        else:
            if kwargs.get('fit'):
                self._fit = check_type(kwargs.get("fit"), Callable, "__init__")
            if kwargs.get('predict'):
                self._predict = check_type(kwargs.get("predict"), Callable, "__init__")
            if kwargs.get('predict_proba'):
                self._predict_proba = check_type(kwargs.get("predict_proba"), Callable, "__init__")
            if kwargs.get('score'):
                self._score = check_type(kwargs.get("score"), Callable, "__init__")
        data = check_type(data, Data, "__init__")
        if data.encoded or (kwargs.get('encoder') and kwargs.get('scaler')):
            d : Data = data
            self.data = d
        else:
            raise ValueError("Invalid argument in __init__: data must be encoded or (scaler, encoder) must be supplied")
        if kwargs.get('encoder'):
            self._encoder = check_type(kwargs.get("encoder"), Transformer, "__init__")
        if kwargs.get('scaler'):
            self._scaler = check_type(kwargs.get("scaler"), Transformer, "__init__")
        if target:
            target = check_type(target, Data, "__init__")
            if (target.n_data == data.n_data):
                if target.encoded or (kwargs.get('encoder') and kwargs.get('scaler')):
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
            d, t = self.get_data_tofit()
            if len(t)>0:
                self.fitted_data = self.train(d, t)
            else:
                self.fitted_data = self.train(d)
        except:
            raise ValueError("Error in model.train: self.train failed - please provide numpy.arrays to self.fit")

    def get_fitted_data(self):
        return self.fitted_data

    def get_data_tofit(self, dd=None, tt=None):
        d = self.data.dataset
        if self.target:
            t = self.target.dataset
        else:
            t = []
        if dd:
            d = dd
        if tt:
            t = tt
        if len(self.X_tofit)>0:
            d = keep_cols(d, self.X_tofit)
        if len(t)>0 and len(self.Y_tofit)>0:
            t = keep_cols(t, self.Y_tofit)
        return d, t

    @property
    def fit(self) -> Callable:
        """
        Sets and changes the fit method of the model
        -------
        Callable
        """
        return self._fit

    @fit.setter
    def fit(self, fit) -> None:
        self._fit = check_type(fit, Callable, "fit.setter")
        
    @property
    def predict(self) -> Callable:
        """
        Sets and changes the predict method of the model
        -------
        Callable
        """
        return self._predict

    @predict.setter
    def predict(self, predict) -> None:
        self._predict = check_type(predict, Callable, "predict.setter")
        
    @property
    def predict_proba(self) -> Callable:
        """
        Sets and changes the predict_proba method of the model
        -------
        Callable
        """
        return self._predict_proba

    @predict_proba.setter
    def predict_proba(self, predict_proba) -> None:
        self._predict_proba = check_type(predict_proba, Callable, "predict_proba.setter")
        
    @property
    def score(self) -> Callable:
        """
        Sets and changes the score method of the model
        -------
        Callable
        """
        return self._score

    @score.setter
    def score(self, score) -> None:
        self._score = check_type(score, Callable, "score.setter")
        
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
        self._encoder = check_type(encoder, Transformer, "encoder.setter")
        
    @property
    def scaler(self) -> Transformer:
        """
        Sets and changes the scaler method of the model
        -------
        Callable
        """
        
        return self._scaler

    @scaler.setter
    def scaler(self, scaler) -> None:
        self._scaler = check_type(scaler, Transformer, "scaler.setter")
        
    def encode(self, X: np.ndarray, columns: List[int]=None):
        if not_in_range(X.shape[1], columns):
            raise ValueError("Invalid arguments in encode: Index in parameter columns is out of range")
        if self.encoder:
            X_copy = X
            if columns:
                cols = columns.sort()
                X_rem = keep_cols(X, cols)
            else:
                cols = range(len(X))
            self.encoder.fit(X_rem)
            X_rem = self.encoder.transform(X_rem)
            j=0
            for i in range(len(X)):
                if i in cols:
                    X_copy[:, i] = X_rem[:, j]
                    j+=1
            return X_copy
        else:
            return X
    
    def decode(self, X: np.ndarray=None, columns: List[int]=None):
        if columns and not_in_range(X.shape[1], columns):
            raise ValueError("Invalid arguments in decode: Index in parameter columns is out of range")
        if self.encoder:
            X_copy = X
            if columns:
                cols = columns.sort()
                X_rem = keep_cols(X, cols)
            else:
                cols = range(len(X))
            X_rem = self.encoder.inverse_transform(X_rem)
            j=0
            for i in range(len(X)):
                if i in cols:
                    X_copy[:, i] = X_rem[:, j]
                    j+=1
            return X_copy
        else:
            return X
    
    def scale(self, X: np.ndarray, columns: List[int]=None):
        if not_in_range(X.shape[1], columns):
            raise ValueError("Invalid arguments in scale: Index in parameter columns is out of range")
        if self.scaler:
            X_copy = X
            if columns:
                cols = columns.sort()
                X_rem = keep_cols(X, cols)
            else:
                cols = range(len(X))
            self.scaler.fit(X_rem)
            X_rem = self.scaler.transform(X_rem)
            j=0
            for i in range(len(X)):
                if i in cols:
                    X_copy[:, i] = X_rem[:, j]
                    j+=1
            return X_copy
        else:
            return X
    
    def unscale(self, X: np.ndarray=None, columns: List[int]=None):
        if columns and not_in_range(X.shape[1], columns):
            raise ValueError("Invalid arguments in unscale: Index in parameter columns is out of range")
        if self.scaler:
            X_copy = X
            if columns:
                cols = columns.sort()
                X_rem = keep_cols(X, cols)
            else:
                cols = range(len(X))
            X_rem = self.scaler.inverse_transform(X_rem)
            j=0
            for i in range(len(X)):
                if i in cols:
                    X_copy[:, i] = X_rem[:, j]
                    j+=1
            return X_copy
        else:
            return X
    
    def train(self, X: np.ndarray=[], Y: np.ndarray=[], cols_encode: List[int]=None, cols_scale: List[int]=None):
        if len(Y)>0 and len(X)<1:
            raise ValueError("Invalid argument to model.train: X not provided - please provide only X or X and Y or nothing")
        else:
            if len(X)>0 and len(Y)>0:
                self.fit(X,Y.ravel())
                return (X,Y)
            if len(X)>0:
                self.fit(X)
                return (X)
            else:
                X_, Y_ = self.data.dataset, self.target.dataset
                if not self.data.encoded:
                    if cols_encode:
                        X_ = self.encode(self.data.dataset, cols_encode)
                    else:
                        X_ = self.encode(self.data.dataset, self.data.categoricals)
                    if cols_scale:
                        X_ = self.scale(self.data.dataset, cols_scale)
                    else:
                        X_ = self.scale(self.data.dataset, self.data.numericals)
                if not self.target.encoded:
                    if cols_encode:
                        Y_ = self.encode(self.target.dataset, cols_encode)
                    else:
                        Y_ = self.encode(self.target.dataset, self.target.categoricals)
                    if cols_scale:
                        Y_ = self.scale(self.target.dataset, cols_scale)
                    else:
                        Y_ = self.scale(self.target.dataset, self.target.categoricals)
                X_, Y_ = self.get_data_tofit(X_, Y_)
                self.fit(X_,Y_.ravel())
                print(f"Classification accuracy on Training Data: {self.score(X_,Y_)}")
                return (X_,Y_)
    
    def __str__(self):
        return f"Data: {self.data}, Target: {self.target}, X_tofit: {self.X_tofit}, Y_tofit: {self.Y_tofit}"