from fatapi.model import Model
from fatapi.model.estimators import Transformer
from fatapi.data import Data
from fatapi.helpers import check_type
from typing import Callable, Tuple, Union
import numpy as np

class ExplainabilityMethod(object):
    """
    Abstract class for AI explainability methods such as FACE, CEM, PDP, ALE...
    Contains methods for generating counterfactuals and the data / model to apply the method to
    
    Parameters
    ----------
    data? : fatapi.data.Data
        Data object containing data and feature columns
    target? : fatapi.data.Data
        Data object containing target data and target feature columns
    factuals? : fatapi.data.Data
        Data object containing features of datapoints to be used in ExplainabilityMethod methods
    factuals_target? : fatapi.data.Data
        Data object containing target features of datapoints to be used in ExplainabilityMethod methods
    predict()? : (X: np.array) -> np.array()
        Method for predicting the class label of X
        -- Only required if model not supplied
    model? : fatapi.model.Model
        Model object used to get prediction values and class predictions
        -- Only required if predict not supplied
    explain()? : (self, X: numpy.array, Y?: numpy.array, _predict()?: Callable) -> numpy.array
        Generates counterfactual datapoints from X and Y using predict function or model predict function or argument predict function
    
    Methods
    -------
    explain() : (X?: numpy.array, Y?: numpy.array, _predict()?: Callable) -> numpy.array
        Generates counterfactual datapoints from X and Y using predict function or model predict function or argument predict function
        -- Uses factuals and factuals_target from preprocess_factuals if no X and Y given
    preprocess_factuals() : (factuals?: fatapi.data.Data, factuals_target?: fatapi.data.Data, model?: fatapi.model.Model, 
                                            scaler?: fatapi.model.estimators.Transformer, encoder?: fatapi.model.estimators.Transformer) -> numpy.array
        Uses encoder and scaler from black-box-model or argument to preprocess data as needed.
    """
    def __init__(self, factuals=None, factuals_target=None, **kwargs) -> None:
        if not (kwargs.get('predict') or kwargs.get('model')) or (kwargs.get('predict') and kwargs.get('model')):
            raise ValueError(f"Invalid arguments in __init__: please provide model or predict function but not both")
        if kwargs.get('model'):
            m = check_type(kwargs.get("model"), Model, "__init__")
            self._model: Model = m
            self._predict = self._model.predict
        if kwargs.get('predict'):
            self._predict = check_type(kwargs.get("predict"), Callable, "__init__")
        self._explain = lambda **kwargs: kwargs
        if kwargs.get('explain'):
            self._explain = check_type(kwargs.get("explain"), Callable, "__init__")
        if not factuals and factuals_target:
            raise ValueError("Invalid argument in __init__: factual targets supplied with no factuals - provide factuals argument if targets are the features")
        self._data=None
        self._target=None
        if kwargs.get('data'):
            self._data = check_type(kwargs.get("data"), Data, "__init__")
        if kwargs.get('target'):
            self._target = check_type(kwargs.get("target"), Data, "__init__")
        else:
            if factuals:
                self._factuals = factuals
            else:
                print("Warning: No datapoints supplied as factuals - counterfactual methods require factuals argument")
            if factuals_target:
                self._factuals_target = factuals_target
            else:
                print("Warning: No targets supplied for factuals (datapoints) - counterfactual methods require factuals_target argument")

    @property
    def data(self) -> Callable:
        """
        Sets and changes the data attribute
        -------
        Callable
        """
        
        return self._data

    @data.setter
    def data(self, data) -> None:
        self._data = check_type(data, Data, "data.setter")

    @property
    def target(self) -> Callable:
        """
        Sets and changes the target attribute
        -------
        Callable
        """
        
        return self._target

    @target.setter
    def target(self, target) -> None:
        self._target = check_type(target, Data, "target.setter")
        
    @property
    def predict(self) -> Callable:
        """
        Sets and changes the predict method of the explainability method
        -------
        Callable
        """
        
        return self._predict

    @predict.setter
    def predict(self, predict) -> None:
        self._predict = check_type(predict, Callable, "predict.setter")
        
    @property
    def model(self) -> Model:
        """
        Sets and changes the model attribute of the explainability method
        -------
        Callable
        """
        
        return self._model

    @model.setter
    def model(self, model) -> None:
        self._model = check_type(model, Model, "model.setter")
          
    @property
    def factuals(self) -> Data:
        """
        Sets and changes the default factuals the explainability method applies to
        -------
        Callable
        """
        
        return self._factuals

    @factuals.setter
    def factuals(self, factuals) -> None:
        self._factuals = check_type(factuals, Data, "factuals.setter")
        
    @property
    def factuals_target(self) -> Data:
        """
        Sets and changes the default factuals_target the explainability method applies to
        -------
        Callable
        """
        
        return self._factuals_target

    @factuals_target.setter
    def factuals_target(self, factuals_target) -> None:
            self._factuals_target = check_type(factuals_target, Data, "factuals_target.setter")
            if (self.factuals.dataset.shape[0]==self.factuals_target.dataset.shape[0]):
                self._factuals_target = factuals_target
            else:
                raise ValueError("Invalid argument in factuals_target.setter: factuals_target has a different number of points than factuals")

    def explain(self, X: np.array=[], Y: np.array=[], facts: np.array=[], facts_target: np.array=[], predict: Callable=None, **kwargs) -> Union[np.array, Tuple[np.array, np.array]]:
        facts_, facts_target_ = self.preprocess_factuals()
        if len(facts)>0:
            facts_ = facts
        if len(facts_target)>0:
            facts_target_ = facts_target
        X_ = []
        Y_ = []
        if len(X)<=0 and not self._model and not self.data:
            raise ValueError("Invalid argument in explain: dataset missing for X; provide X or self.data or self._model")
        if self.model:
            X_ = self.model.data
            if self.model.target:
                Y_ = self.model.target
        if self.data:
            X_ = self.data
        if self.target:
            Y_ = self.target
        X_, Y_ = self.preprocess_factuals(factuals=X_, factuals_target=Y_)
        if len(X)>0:
            X_ = X
        if len(Y)>0:
            Y_ = Y
        if len(Y_)>0 and len(facts_target_)==0:
            raise ValueError("Invalid arguments to explain: target for data supplied but no facts_target_")
        if len(Y_)<=0 and len(facts_target_)>0:
            raise ValueError("Invalid arguments to explain: facts_target_ supplied but no target for data")
        if predict:
            _predict = predict
        else:
            _predict = self._predict
        if len(Y_)>0 and not (X_.shape[0]==Y_.shape[0]):
                raise ValueError("Invalid argument in explain: different number of points in data and target")
        if not (facts_.shape[0]==facts_target_.shape[0]):
            raise ValueError("Invalid argument in explain: different number of points in facts and facts_target")
        if len(facts_target_)>0:
            if len(X_)>0:
                if len(Y_)>0:
                    return self._explain(X=X_, Y=Y_, factuals=facts_, factuals_target=facts_target_, predict=_predict)
                else:
                    return self._explain(X=X_, factuals=facts_, factuals_target=facts_target_, predict=_predict)
            else:
                return self._explain(factuals=facts_, factuals_target=facts_target_, predict=_predict)
        else:
            if len(X_)>0:
                if len(Y_)>0:
                    return self._explain(X=X_, Y=Y_, factuals=facts_, predict=_predict)
                else:
                    return self._explain(X=X_, factuals=facts_, predict=_predict)
            else:
                return self._explain(factuals=facts_, predict=_predict)

    def preprocess_factuals(self, factuals: Data=None, factuals_target: Data=None, model: Model=None, scaler: Transformer=None, encoder: Transformer=None) -> Union[Tuple[np.array, np.array], np.array]:
        if not self.factuals and not factuals:
            raise ValueError(f"Missing arguments in preprocess_factuals: must provide {'' if self.factuals else 'self.factuals'} or {'' if factuals else 'factuals'}")
        facts = None
        if self.factuals:
            facts = self.factuals
        if factuals:
            facts = factuals
        if not self.factuals_target:
            if factuals_target:
                print(f"Warning: targets for factuals provided in preprocess_factuals but no default self.factuals_target")
            else:
                print(f"Warning: no targets for factuals provided to preprocess_factuals - features only (X)")
        facts_target = None
        if self.factuals_target:
            facts_target = self.factuals_target
        if factuals_target:
            facts_target = factuals_target
        Y_ = []
        if facts_target:
            Y_ = facts_target.dataset
        X_ = facts.dataset
        if facts.encoded and facts_target.encoded:
            return X_, Y_
        else:
            if not (self.model or model or scaler or encoder):
                raise ValueError(f"Missing arguments in preprocess_factuals: must provide {'' if self.model else 'self.model'} or {'' if model else 'model'} or {'' if scaler else 'scaler'} or {'' if encoder else 'encoder'}")
            else:
                if self.model:
                    _encode = self.model.encode()
                    _scale = self.model.scale()
                if model:
                    _encode = model.encode()
                if encoder:
                    _encode = encoder.encode()
                if scaler:
                    _scale = scaler.encode()
                if _encode:
                    X_ = _encode(facts.dataset, facts.categoricals)
                if _scale:
                    X_ = _scale(facts.dataset, facts.numericals)
                if facts_target: 
                    if _encode:
                        Y_ = _encode(facts_target.dataset, facts_target.categoricals)
                    if _scale:
                        Y_ = _scale(facts_target.dataset, facts_target.numericals)
            return X_, Y_

    def __str__(self):
        return f"Factuals: {self.factuals}, Factual Targets: {self.factuals_target}"