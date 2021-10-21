
from typing import Callable
from fatapi.helpers import check_type
import numpy as np

class Optimiser(object):
    """
    Abstract class used for a gradient ascent / descent optimiser (e.g. in CEMMethod).
    
    Note: Initialised variables will be overwritten by Method defaults and supplied Method args - 
    call setters before explain() in Method to override the arguments for the optimiser in the Method constructor
    
    
    Parameters
    ----------
    objective() : (X: np.ndarray, Y: np.npdarray, predict()?: Callable[[np.ndarray], np.ndarray], predict:_proba()?: Callable[[np.ndarray],
                    np.ndarray], **kwargs) -> np.ndarray
        The objective function the optimiser is trying to solve. Accepts any other arguments that may be required as **kwargs
        (e.g. delta, beta, gamma, c, autoencoder for CEM)
    optimise()? : (objective?: Callable[..., np.ndarray], max_iterations?: int, initial_learning_rate?: float, 
                    decay_function?: Callable[[float, int, int], float], **kwargs) -> np.ndarray
        Optimises the supplied objective function using a supplied learning rate and optional decay function, 
        or using those set in the Optimiser object
    predict()? : (X: np.ndarray) -> np.ndarray
        Method for predicting the class label of X
        -- Only required if needed in optimise() or objective()
    predict_proba()? : (X: np.ndarray) -> np.ndarray
        Method for getting the probability of X being the predicted class label
        -- Only required if needed in optimise() or objective()
    initial_learning_rate?: float
        Initial learning rate / step size for learning
        -- Default is 1e-2
    decay_function()?: (learning_rate: float, iteration: int, max_iterations: int, **kwargs) -> float
        Function which decays the learning rate over the iterations of learning
        -- Default is lambda lr, i, m, **kwargs = lr (Identity on initial_learning_rate [lr])
    max_iterations?: int
        Maximum iterations to complete an optimisation cycle over - stopping condition for learning
        -- Default is 1000
    stop_condition()?: (iteration?: int, max_iterations?: int, **kwargs) -> bool 
        Extra stopping conditions for the optimiser / learning cycle
        -- Default is lambda *args, **kwargs = False [Identity]

    Methods
    -------
    objective() : (X: np.ndarray, Y: np.npdarray, predict(): Callable[[np.ndarray], np.ndarray], **kwargs) -> np.ndarray
        The objective function the optimiser is trying to solve. Accepts any other arguments that may be required as **kwargs
        (e.g. delta, beta, gamma, c, autoencoder for CEM)
    optimise() : (objective: Callable[..., np.ndarray], max_iterations: int, initial_learning_rate: Union[float, int], 
                    decay_function: Callable[[float, int, int], float]) -> np.ndarray
        Optimises the supplied objective function using a supplied learning rate and optional decay function, 
        or using those set in the Optimiser object
    """
    def __init__(self, **kwargs) -> None:
        if kwargs.get("estimator"):
            self.estimator = kwargs.get("estimator")
            try:
                if callable(getattr(self.estimator, "fit")):
                    pass
            except:
                raise ValueError("Invalid argument in __init__: estimator does not have function fit")
            try:
                if callable(getattr(self.estimator, "score_samples")):
                    pass
            except:
                raise ValueError("Invalid argument in __init__: estimator does not have function score_samples")
            self._fit = self.estimator.fit
            self._score_samples = self.estimator.score_samples
            self._score = None
        else:
            self._fit = self.base_fit
            self._score = self.base_score
            self._score_samples = self.base_score_samples

        if kwargs.get("distance_function"): 
            self._distance_function = check_type(kwargs.get("distance_function"), "__init__", Callable)
        else:
            self._distance_function = lambda x, y: np.linalg.norm(x.reshape(-1, 1) - y.reshape(-1, 1))
            
        if kwargs.get("transformation_function"): 
            self._transformation_function = check_type(kwargs.get("transformation_function"), "__init__", Callable)
        else:
            self._transformation_function = lambda x: -np.log(x)
        
    def base_fit(self, X):
        self.X = X
        self.n_samples = X.shape[0]   
    
    def base_score(self, X: np.ndarray, K: int=10):
        distances = np.zeros(self.n_samples)
        for idx in range(self.n_samples):
            distances[idx] = self.distance_function(X, self.X[idx, :])
        return self.transformation_function(np.sort(distances)[K])
    
    def base_score_samples(self, X: np.ndarray, K: int=10):
        n_samples_test = X.shape[0]
        if n_samples_test == 1:
            return self.score_samples_single(X)
        else:
            scores = np.zeros((n_samples_test, 1))
            for idx in range(n_samples_test):
                scores[idx] = self.score(X[idx, :], K)
            return scores

    @property
    def distance_function(self) -> Callable:
        """
        Sets and changes the distance_function method of the density estimator
        -------
        Callable
        """
        
        return self._distance_function

    @distance_function.setter
    def distance_function(self, distance_function) -> None:
        self._distance_function = check_type(distance_function, "distance_function.setter", Callable)
        
    @property
    def transformation_function(self) -> Callable:
        """
        Sets and changes the transformation_function method of the density estimator
        -------
        Callable
        """
        
        return self._transformation_function

    @transformation_function.setter
    def transformation_function(self, transformation_function) -> None:
        self._transformation_function = check_type(transformation_function, "transformation_function.setter", Callable)

    @property
    def fit(self) -> Callable:
        """
        Sets and changes the fit method of the density estimator
        -------
        Callable
        """
        return self._fit

    @fit.setter
    def fit(self, fit) -> None:
        self._fit = check_type(fit, "fit.setter", Callable)

    @property
    def score(self) -> Callable:
        """
        Sets and changes the score method of the density estimator
        -------
        Callable
        """
        return self._score

    @score.setter
    def score(self, score) -> None:
        self._score = check_type(score, "score.setter", Callable)

    @property
    def score_samples(self) -> Callable:
        """
        Sets and changes the score_samples method of the density estimator
        -------
        Callable
        """
        return self._score_samples

    @score_samples.setter
    def score_samples(self, score_samples) -> None:
        self._score_samples = check_type(score_samples, "score_samples.setter", Callable)