
from fatapi.model.estimator import Estimator
from fatapi.data import Data
from fatapi.model import BlackBox
from typing import Callable, List
from fatapi.helpers import not_in_range, keep_cols
import numpy as np

class Model():
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
    predict()? : (X: np.array) ->np.array()
        Method for predicting the class label of X
        -- Only required if blackbox not supplied
    predict_proba()? : (X: np.array) -> np.array()
        Method for predicting the probability of the prediction of X
        -- Only required if blackbox not supplied
    fit()? : (X: np.array, Y?: np.array) -> np.array()
        Method for fitting model to X, Y
        -- Only required if blackbox not supplied
    score()? : (X: np.array, Y?: np.array) -> np.array()
        Method for calculating a score when predicting X and comparing with Y
        -- Only required if blackbox not supplied
    scaler? : fatapi.model.Estimator
        Scaler for normalisation / scaling numericals features
    encoder? : fatapi.model.Estimator
        Encoder for encoding categorical features
    
    Methods
    -------
    get_fitted_data(): () -> (numpy.array, numpy.array?)
        Returns the dataset the model has been fitted to; tuple of (X) or (X, Y)
    train(X: np.array, Y: np.array):
        Calls self.fit() after scaling/normalising
    encode(X: np.array, columns: List[int]):
        Fits and transforms the data using encoder
        -- If no encoder, returns X
    decode(X: np.array, columns: List[int]):
        Inverse_transforms the data using encoder
        -- If no encoder, returns X
    scale(X: np.array, columns: List[int]):
        Fits and transforms the data using scaler
        -- If no scaler, returns X
    unscale(X: np.array, columns: List[int]):
        Inverse_transforms the data using scaler
        -- If no scaler, returns X
    """
    def __init__(self, data: Data, target: Data=None, blackbox: BlackBox=None, X_tofit: List[int]=[], Y_tofit: List[int]=[], **kwargs) -> None:
        if not blackbox and not (kwargs.get('fit') and kwargs.get('predict') and kwargs.get('predict_proba') and kwargs.get('score')):
            raise ValueError(f"Missing arguments in __init__: {'' if blackbox else 'blackbox'} {'' if kwargs.get('fit') else 'fit'} {'' if kwargs.get('predict') else 'predict'} {'' if kwargs.get('predict_proba') else 'predict_proba'} {'' if kwargs.get('score') else 'score'}")
        if blackbox and (kwargs.get('fit') and kwargs.get('predict') and kwargs.get('predict_proba') and kwargs.get('score')):
            raise ValueError("Invalid arguments in __init__: please provide blackbox or (predict, predict_proba, fit, score)")
        self.blackbox = blackbox
        if (kwargs.get('fit') and kwargs.get('predict') and kwargs.get('predict_proba') and kwargs.get('score')):
            if kwargs.get('fit'):
                if (callable(kwargs.get('fit'))):
                    self._encoder = kwargs.get('fit')
                else:
                    raise ValueError("Invalid argument in __init__: fit is not a function")
            if kwargs.get('predict'):
                if (callable(kwargs.get('predict'))):
                    self._encoder = kwargs.get('predict')
                else:
                    raise ValueError("Invalid argument in __init__: predict is not a function")
            if kwargs.get('predict_proba'):
                if (callable(kwargs.get('predict_proba'))):
                    self._encoder = kwargs.get('predict_proba')
                else:
                    raise ValueError("Invalid argument in __init__: predict_proba is not a function")
            if kwargs.get('score'):
                if (callable(kwargs.get('score'))):
                    self._encoder = kwargs.get('score')
                else:
                    raise ValueError("Invalid argument in __init__: score is not a function")
        if type(data)==Data:
            if data.isEncoded or (kwargs.get('encoder') and kwargs.get('scaler')):
                d : Data = data
                self.data = d
            else:
                raise ValueError("Invalid argument in __init__: data must be isEncoded or (scaler, encoder) must be supplied")
        else:
            raise ValueError("Invalid argument in __init__: data must be of type fatapi.data.Data")
        if kwargs.get('encoder'):
            if (type(kwargs.get('encoder'))==Estimator):
                self._encoder = kwargs.get('encoder')
            else:
                raise ValueError("Invalid argument in __init__: encoder is not an Estimator")
        if kwargs.get('scaler'):
            if (type(kwargs.get('scaler'))==Estimator):
                self._encoder = kwargs.get('scaler')
            else:
                raise ValueError("Invalid argument in __init__: scaler is not an Estimator")
        if target:
            if (target.n_data == data.n_data):
                if type(target)==Data:
                    if target.isEncoded or (kwargs.get('encoder') and kwargs.get('scaler')):
                        t : Data = target
                        self.target = t
                    else:
                        raise ValueError("Invalid argument in __init__: target must be isEncoded or (scaler, encoder) must be supplied")
                else:
                    raise ValueError("Invalid argument in __init__: target is not of type fatapi.data.Data")
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
            raise ValueError("Invalid argument in model.fit: has to take (X: numpy.array, Y: numpy.array)")

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
        if self.blackbox:
            return self.blackbox.fit
        else:
            return self._fit

    @fit.setter
    def fit(self, _fit) -> None:
        if callable(_fit):
            self._fit = _fit
        else:
            raise ValueError("Invalid argument in fit.setter: _fit is not a function")
        
    @property
    def predict(self) -> Callable:
        """
        Sets and changes the predict method of the model
        -------
        Callable
        """
        if self.blackbox:
            return self.blackbox.predict
        else:
            return self._predict

    @predict.setter
    def predict(self, predict) -> None:
        if callable(predict):
            self._predict = predict
        else:
            raise ValueError("Invalid argument in predict.setter: _predict is not a function")
        
    @property
    def predict_proba(self) -> Callable:
        """
        Sets and changes the predict_proba method of the model
        -------
        Callable
        """
        if self.blackbox:
            return self.blackbox.predict_proba
        else:
            return self._predict_proba

    @predict_proba.setter
    def predict_proba(self, predict_proba) -> None:
        if callable(predict_proba):
            self._predict_proba = predict_proba
        else:
            raise ValueError("Invalid argument in predict_proba.setter: predict_probaf is not a function")
        
    @property
    def score(self) -> Callable:
        """
        Sets and changes the score method of the model
        -------
        Callable
        """
        if self.blackbox:
            return self.blackbox.score
        else:
            return self._score

    @score.setter
    def score(self, score) -> None:
        if callable(score):
            self._score = score
        else:
            raise ValueError("Invalid argument in score.setter: _score is not a function")
        
    @property
    def encoder(self) -> Estimator:
        """
        Sets and changes the encoder method of the model
        -------
        Estimator
        """
        
        return self._encoder

    @encoder.setter
    def encoder(self, encoder) -> None:
        if callable(encoder):
            self._encoder = encoder
        else:
            raise ValueError("Invalid argument in encoder.setter: encoder is not an Estimator")
        
    @property
    def scaler(self) -> Estimator:
        """
        Sets and changes the scaler method of the model
        -------
        Callable
        """
        
        return self._scaler

    @scaler.setter
    def scaler(self, scaler) -> None:
        if callable(scaler):
            self._scaler = scaler
        else:
            raise ValueError("Invalid argument in scaler.setter: scalerf is not a function")
        
    def encode(self, X: np.array, columns: List[int]=None):
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
    
    def decode(self, X: np.array=None, columns: List[int]=None):
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
    
    def scale(self, X: np.array, columns: List[int]=None):
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
    
    def unscale(self, X: np.array=None, columns: List[int]=None):
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
    
    def train(self, X: np.array=[], Y: np.array=[], cols_encode: List[int]=None, cols_scale: List[int]=None):
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
                if not self.data.isEncoded:
                    if cols_encode:
                        X_ = self.encode(self.data.dataset, cols_encode)
                    else:
                        X_ = self.encode(self.data.dataset, self.data.categoricals)
                    if cols_scale:
                        X_ = self.scale(self.data.dataset, cols_scale)
                    else:
                        X_ = self.scale(self.data.dataset, self.data.numericals)
                if not self.target.isEncoded:
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