from fatapi.model import Model
from fatapi.model.estimators import Transformer, DensityEstimator
from fatapi.optimisers import Optimiser, FISTAOptimiser
from fatapi.data import Data
from fatapi.helpers import dijkstra, sigmoid
from typing import Callable, Tuple, Union, List
from fatapi.methods import ExplainabilityMethod
from fatapi.helpers import check_type, get_volume_of_sphere
import numpy as np

class CEMMethod(ExplainabilityMethod):
    """
    Abstract class for the CEM algorithm as an AI explainability method.
    
    Contains methods for generating counterfactuals and the data / model to apply FACE to, 
    as well as extracting internal representations
    
    Parameters
    ----------
    data? : fatapi.data.Data
        Data object containing data and feature columns
        -- Only required if X and Y are supplied to explain()
    target? : fatapi.data.Data
        Data object containing target data and target feature columns
        -- Only required if X and Y are supplied to explain()
    factuals? : fatapi.data.Data
        Data object containing features of datapoints to be used in ExplainabilityMethod methods
        -- Only required if factuals and factuals_target are supplied to explain()
    factuals_target? : fatapi.data.Data
        Data object containing target features of datapoints to be used in ExplainabilityMethod methods
        -- Only required if factuals and factuals_target are supplied to explain()
    predict()? : (X: np.ndarray) -> np.ndarray
        Method for predicting the class label of X
        -- Only required if model not supplied
    predict_proba()? : (X: np.ndarray) -> np.ndarray
        Method for getting the probability of X being the predicted class label
        -- Only required if model not supplied
    model? : fatapi.model.Model
        Model object used to get prediction values and class predictions
        -- Only required if predict not supplied
    mode? : String
        String specifying which elements of the counterfactual should be gathered - default is "pn" but can be "pp"/"pn"
        -- Default is "pn"
    autoencoder?: (X: np.ndarray, **kwargs) -> np.ndarray
        Autoencoder which transforms datapoint X to get a more useful counterfactual result by making X closer to a data manifold
        -- Default returns X
    kappa? : Union[float, int]
        Confidence parameter which controls the seperation between the predicted class of the modified input and tbe
         next-highest-prediction predicted class of said modified input
        -- Default is 1
    c? : Union[float, int]
        Regularisation parameter for the loss function
        -- Default is 1
    beta? : Union[float, int]
        Regularisation parameter for the elastic net regulariser
        -- Default is 1
    gamma? : Union[float, int]
        Regularisation parameter for the reconstruction error of autoencoding the input
        -- Default is 1, 0 if no autoencoder
    initial_learning_rate? : Union[float, int]
        Learning rate for the optimiser for the CEM loss function (initial learning rate if decay_function supplied to optimiser)
        -- Default is 1e-2
    decay_function? : (initial_learning_rate: float, iteration: int, max_iterations: int, **kwargs) -> (learning_rate: float)
        Function for decaying the learning_rate over iterations of the optimiser
        -- Default is lambda lr, i, m, **kwargs: lr (Identity)
    max_iterations? : int
        Maximum iterations for the optimiser on the CEM loss function
        -- Default is 1000
    search_c? : bool
        Whether to do a binary search on 'c' to explore a larger spread of the feature space, or not
        -- Default is False
    search_c_max_iterations? : int
        Maximum iterations of a binary search on 'c'
        -- Default is 10
    search_c_upperbound? : Union[float, int]
        Upper bound for the value of 'c' during binary search on 'c'
        -- Default is 1e10
    search_c_lowerbound? : Union[float, int]
        Lower bound for the value of 'c' during binary search on 'c'
        -- Default is 0
    initial_deltas? : np.ndarray[num_features]
        Initial value for the permutation of the factual (delta)
        -- Default is np.zeros((1,X.shape[1]))
       
    Methods
    -------
    explain() : (X?: np.ndarray, Y?: np.ndarray, predict()?: Callable) -> np.ndarray
        Generates counterfactual datapoints from X and Y using predict function or model predict function or argument predict function
        Returns permutations for items of X in shape of X
        -- Uses factuals and factuals_target from preprocess_factuals if no X and Y given
    preprocess_factuals() : (factuals?: fatapi.data.Data, factuals_target?: fatapi.data.Data, model?: fatapi.model.Model, 
                                            scaler?: fatapi.model.estimators.Transformer, encoder?: fatapi.model.estimators.Transformer) -> np.ndarray
        Uses encoder and scaler from black-box-model or argument to preprocess data as needed.
    build_graph() : (X: np.ndarray, Y: np.ndarray, t_distance?: float, t_density?: float, 
                                    t_prediction?: float, conditions()?: Callable) -> np.ndarray
        Builds graph for distances between nodes in the feature space - returns adjacency matrix of weights between [i,j]; 
        i, j are indicies of datapoints in X (rows)
    get_permutations() : () -> np.ndarray
        Returns the graph which build_graph() produces
    get_explain_targets() : () -> List[float]
        Returns the classifications of X after explain (PP: Y[i] for X[i] / PN: class after X[i]+permutations[i])
    get_counterfactuals(as_indexes?: bool) : () -> np.ndarray
        Returns the counterfactuals for the supplied factuals (permutations of X)
        -- Default is as data (not as_indexes)
    get_counterfactuals_as_data() : () -> np.ndarray
        Returns the counterfactuals as data in the same form X and Y were supplied as tuple (data, target)
        -- target is the same as get_counterfactuals()
    """
    def __init__(self, *args, **kwargs) -> None:
        super(CEMMethod, self).__init__(*args, **kwargs)
        if not ('factuals' in kwargs and 'factuals_target' in kwargs):
            print("Warning in __init__: factuals and factuals_target not supplied - need to provide factuals and factuals_target to explain()")
        else:
            self._factuals = check_type(kwargs.get("factuals"), "__init__", Data)
            self._factuals_target = check_type(kwargs.get("factuals_target"), "__init__", Data)
        if not 'autoencoder' in kwargs:
            print("Warning in __init__: no autoencoder supplied - X will not be put closer to data manifold and results may be inaccurate")
        noAE = True
        self._autoencoder = lambda X, **kwargs: X
        self._mode = "pn"
        self._search_c = False
        self._search_c_max_iterations = 10
        self._search_c_upperbound = 1e10
        self._search_c_lowerbound = 0
        self._kappa = 1
        self._c = 1
        self._beta = 1
        self._gamma = 0
        self._initial_deltas = np.zeros((1, np.shape(self._factuals)[1]))
        self._initial_learning_rate = 1e-2
        self._decay_function = lambda lr, i, m, **kwargs : lr
        self._max_iterations = 1000
        if 'autoencoder' in kwargs:
            self._autoencoder = check_type(kwargs.get("autoencoder"), "__init__", Callable[[np.ndarray], np.ndarray])
            self._gamma = 1
            noAE = False
        if 'mode' in kwargs:
            m: str = check_type(kwargs.get("mode"), "__init__", str)
            if not (m.lower()=="pp" or m.lower()=="pn"):
                raise ValueError(f"Invalid arguments in __init__: mode must be 'pn' or 'pp'")
            else:
                self._mode = check_type(kwargs.get("mode"), "__init__", str)
        if 'search_c' in kwargs:
            self._search_c = check_type(kwargs.get("search_c"), "__init__", bool)
        if self._search_c:
            self._c = 10
        if 'search_c_max_iterations' in kwargs:
            if kwargs.get('search_c_max_iterations') >= 1:
                self._search_c_max_iterations = check_type(kwargs.get("search_c_max_iterations"), "__init__", int)
            else:
                raise ValueError(f"Invalid argument in __init__: search_c_max_iterations must be >= 1")
        if 'search_c_upperbound' in kwargs:
            self._search_c_upperbound = check_type(kwargs.get("search_c_upperbound"), "__init__", float, int)
        if 'search_c_lowerbound' in kwargs:
            self._search_c_lowerbound = check_type(kwargs.get("search_c_lowerbound"), "__init__", float, int)
        if self._search_c_lowerbound >= self._search_c_upperbound:
            raise ValueError(f"Invalid argument in __init__: search_c_lowerbound must be less than search_c_upperbound")
        if 'max_iterations' in kwargs:
            if kwargs.get('max_iterations') >= 1:
                self._max_iterations = check_type(kwargs.get("max_iterations"), "__init__", int)
            else:
                raise ValueError(f"Invalid argument in __init__: max_iterations must be >= 1")
        if 'kappa' in kwargs:
            if kwargs.get('kappa') >= 0:
                self._kappa = check_type(kwargs.get("kappa"), "__init__", float, int)
            else:
                raise ValueError(f"Invalid argument in __init__: kappa must be >= 0")
        if 'c' in kwargs:
            if kwargs.get('c') >= 0:
                self._c = check_type(kwargs.get("c"), "__init__", float, int)
            else:
                raise ValueError(f"Invalid argument in __init__: c must be >= 0")
        if 'beta' in kwargs:
            if kwargs.get('beta') >= 0:
                self._beta = check_type(kwargs.get("beta"), "__init__", float, int)
            else:
                raise ValueError(f"Invalid argument in __init__: beta must be >= 0")
        if 'gamma' in kwargs:
            if (noAE) or (not noAE and kwargs.get('gamma') >= 0):
                self._gamma = check_type(kwargs.get("gamma"), "__init__", float, int)
            else:
                raise ValueError(f"Invalid argument in __init__: gamma must be >= 0")
        if 'initial_learning_rate' in kwargs:
            if kwargs.get('initial_learning_rate') >= 0:
                self._initial_learning_rate = check_type(kwargs.get("initial_learning_rate"), "__init__", float, int)
            else:
                raise ValueError(f"Invalid argument in __init__: initial_learning_rate must be >= 0")
        if 'decay_function' in kwargs:
            self._decay_function = check_type(kwargs.get("decay_function"), "__init__", Callable[[float, int, int], float])
        if 'initial_deltas' in kwargs:
            self._initial_deltas = check_type(kwargs.get("initial_deltas"), "__init__", np.ndarray)
            if not np.shape(self._initial_deltas[1])==np.shape(self._factuals[1]):
                raise ValueError(f"Invalid argument in __init__: initial_deltas must be the same shape as features")
        # The Fast Iterative Shrinkage-Thresholding Algorithm [https://www.ceremade.dauphine.fr/~carlier/FISTA] optimiser
        self._optimiser = FISTAOptimiser(objective=self.objective_function, autoencoder=self._autoencoder, predict=self._predict, predict_proba=self._predict_proba, 
                                         initial_deltas=self._initial_deltas, initial_learning_rate=self._initial_learning_rate, 
                                         max_iterations=self._max_iterations, beta=self._beta, decay_function=self._decay_function)
        if 'optimiser' in kwargs:
            self._optimiser = check_type(kwargs.get("optimiser"), "__init__", Optimiser)
            self._optimiser.objective = self.objective_function
            self._optimiser.predict = self._predict
            self._optimiser.predict_proba = self._predict_proba
            self._optimiser.decay_function = self._decay_function
            self._optimiser.initial_learning_rate = self._initial_learning_rate
            self._optimiser.max_iterations = self._max_iterations
            self._optimiser.autoencoder = self._autoencoder
            self._optimiser.beta = self._beta
            self._optimiser.initial_deltas = self._initial_deltas
            
    @property
    def autoencoder(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Sets and changes the autoencoder which transforms X to be closer to the data manifold

        """
        
        return self._autoencoder

    @autoencoder.setter
    def autoencoder(self, autoencoder) -> None:
        self._autoencoder = check_type(autoencoder, "__init__", Callable[[np.ndarray], np.ndarray])

    @property
    def mode(self) -> str:
        """
        Sets and changes the mode of the CEM algorithm - pertient positives or pertient negatives

        """
        
        return self._mode
    
    @mode.setter
    def mode(self, mode) -> None:
        if mode.lower()=="pn".lower() or mode.lower()=="pp".lower():
            self._mode = mode
        else:
            raise ValueError("Invalid argument in kernel.setter: mode is not 'pp' or 'pn'") 

    @property
    def kappa(self) -> Union[float, int]:
        """
        Sets and changes the kappa variable of the CEM algorithm

        """
        
        return self._kappa
        
    @kappa.setter
    def kappa(self, kappa) -> None:
        if kappa >= 0:
            self._kappa = check_type(kappa, "kappa.setter", float, int)
        else:
            raise ValueError("Invalid argument in kappa.setter: kappa must be >= 0")

    @property
    def c(self) -> Union[float, int]:
        """
        Sets and changes the c variable of the CEM algorithm

        """
        
        return self._c

    @c.setter
    def c(self, c) -> None:
        if c >= 0:
            self._c = check_type(c, "c.setter", float, int)
        else:
            raise ValueError("Invalid argument in c.setter: c must be >= 0")

    @property
    def beta(self) -> Union[float, int]:
        """
        Sets and changes the beta variable of the CEM algorithm

        """
        
        return self._beta

    @beta.setter
    def beta(self, beta) -> None:
        if beta >= 0:
            self._beta = check_type(beta, "beta.setter", float, int)
        else:
            raise ValueError("Invalid argument in beta.setter: beta must be >= 0")

    @property
    def gamma(self) -> Union[float, int]:
        """
        Sets and changes the gamma variable of the CEM algorithm

        """
        
        return self._gamma

    @gamma.setter
    def gamma(self, gamma) -> None:
        if gamma >= 0:
            self._gamma = check_type(gamma, "gamma.setter", float, int)
        else:
            raise ValueError("Invalid argument in gamma.setter: gamma must be >= 0")

    @property
    def search_c(self) -> bool:
        """
        Sets and changes whether the 'c' regularisation parameter gets found via binary search

        """
        
        return self._search_c

    @search_c.setter
    def search_c(self, search_c) -> None:
        self._search_c = check_type(search_c, "__init__", bool)
        
    @property
    def search_c_max_iterations(self) -> int:
        """
        Sets and changes the maximum iterations over the binary search over 'c'

        """
        
        return self._search_c_max_iterations

    @search_c_max_iterations.setter
    def search_c_max_iterations(self, search_c_max_iterations) -> None:
        if search_c_max_iterations >= 1:
            self._search_c_max_iterations = check_type(search_c_max_iterations, "search_c_max_iterations.setter", int)
        else:
            raise ValueError("Invalid argument in search_c_max_iterations.setter: search_c_max_iterations must be >= 1")

    @property
    def max_iterations(self) -> int:
        """
        Sets and changes the maximum iterations over the optimiser

        """
        
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, max_iterations) -> None:
        if max_iterations >= 1:
            self._max_iterations = check_type(max_iterations, "max_iterations.setter", int)
        else:
            raise ValueError("Invalid argument in max_iterations.setter: max_iterations must be >= 1")

    @property
    def decay_function(self) -> Callable[[float, int, int], float]:
        """
        Sets and changes the decay_function which decays the learning_rate over iterations of the optimiser

        """
        
        return self._decay_function

    @decay_function.setter
    def decay_function(self, decay_function) -> None:
        self._decay_function = check_type(decay_function, "__init__", Callable[[float, int, int], float])
        
    @property
    def search_c_lowerbound(self) -> Union[float, int]:
        """
        Sets and changes the lower bound of the 'c' parameter for search

        """
        
        return self._search_c_lowerbound

    @search_c_lowerbound.setter
    def search_c_lowerbound(self, search_c_lowerbound) -> None:
        if search_c_lowerbound >= self.search_c_upperbound:
            self._search_c_lowerbound = check_type(search_c_lowerbound, "search_c_lowerbound.setter", float, int)
        else:
            raise ValueError("Invalid argument in search_c_lowerbound.setter: search_c_lowerbound must be < search_c_upperbound")
        
    @property
    def search_c_upperbound(self) -> Union[float, int]:
        """
        Sets and changes the upper bound of the 'c' parameter for search

        """
        
        return self._search_c_upperbound

    @search_c_upperbound.setter
    def search_c_upperbound(self, search_c_upperbound) -> None:
        if search_c_upperbound <= self.search_c_lowerbound:
            self._search_c_upperbound = check_type(search_c_upperbound, "search_c_upperbound.setter", float, int)
        else:
            raise ValueError("Invalid argument in search_c_upperbound.setter: search_c_upperbound must be > search_c_lowerbound")
    
    @property
    def initial_learning_rate(self) -> Union[float, int]:
        """
        Sets and changes the learning rate of the optimiser

        """
        
        return self._initial_learning_rate

    @initial_learning_rate.setter
    def initial_learning_rate(self, initial_learning_rate) -> None:
        if initial_learning_rate > 0:
            self._initial_learning_rate = check_type(initial_learning_rate, "initial_learning_rate.setter", float, int)
        else:
            raise ValueError("Invalid argument in initial_learning_rate.setter: initial_learning_rate must be > 0")
    
    @property
    def initial_initial_deltas(self):
        """
        Sets and changes the initial values of delta for the CEM algorithm

        """
        
        return self._initial_initial_deltas

    @initial_initial_deltas.setter
    def initial_initial_deltas(self, initial_initial_deltas) -> None:
        self._initial_initial_deltas = check_type(initial_initial_deltas, "initial_initial_deltas.setter", np.ndarray)
        if not np.shape(initial_initial_deltas)[1] == np.shape(self._factuals)[1]:
            raise ValueError("Invalid argument in initial_initial_deltas.setter: initial_initial_deltas must have same number of features as factuals")
    
    def kernel_KDE(self, X: np.ndarray, t_density: float, t_distance: float, density_estimator: DensityEstimator, weight_function: Callable, samples=None):
        density_estimator.fit(X)
        n_samples = X.shape[0]
        length = n_samples
        if samples:
            length = samples
        g = np.zeros([n_samples, length], dtype=float)
        for i in range(n_samples):
            for j in range(i):
                X_1=X[i, :]
                X_2=X[j, :]
                v0 = X_1.reshape(-1, 1)
                v1 = X_2.reshape(-1, 1)
                dist = np.linalg.norm(v0 - v1)
                if dist <= t_distance:
                    midpoint = (v0 + v1)/2
                    density = density_estimator.score_samples(midpoint.reshape(1, -1))
                    density_X_1 = density_estimator.score_samples(X_1.reshape(1, -1))
                    if np.exp(density_X_1) >= t_density:
                        g[i,j] = weight_function(np.exp(density)) * dist
                    else:
                        g[i,j] = 0
                else:
                    g[i,j] = 0
        g = g + g.T - np.diag(np.diag(g))
        return g
        
    def kernel_GS(self, X: np.ndarray, t_density: float, t_distance: float, K: int, radius_limit: float, density_estimator:  DensityEstimator, weight_function: Callable, samples=None):
        density_estimator.fit(X)
        n_samples = X.shape[0]
        length = n_samples
        if samples:
            length = samples
        g = np.zeros([n_samples, length], dtype=float)
        for i in range(n_samples):
            for j in range(i):
                X_1=X[i, :]
                X_2=X[j, :]
                v0 = X_1.reshape(-1, 1)
                v1 = X_2.reshape(-1, 1)
                dist = np.linalg.norm(v0 - v1)
                if dist <= t_distance:
                    midpoint = (v0 + v1)/2
                    density = density_estimator.score_samples(midpoint.reshape(1, -1), K)
                    gs_score = density_estimator.score_samples(X_1.reshape(1, -1), K+1)
                    if gs_score >= radius_limit and density >= t_density:
                        g[i,j] = weight_function(sigmoid(density)) * dist
                    else:
                        g[i,j] = 0
                else:
                    g[i,j] = 0
        g = g + g.T - np.diag(np.diag(g))
        return g

    def kernel_KNN(self, X: np.ndarray, n_neighbours: int, weight_function: Callable, samples=None):
        n_samples = X.shape[0]
        length = n_samples
        if samples:
            length = samples
        volume_sphere = get_volume_of_sphere(X.shape[1])
        const = (n_neighbours / (n_samples * volume_sphere))**(1/X.shape[1])
        g = np.zeros((n_samples, length))
        for i in range(n_samples):
            v0 = X[i, :].reshape(-1, 1)
            counter = 0
            for j in range(length):
                v1 = X[j, :].reshape(-1, 1)
                g[i, j] = np.linalg.norm(v0 - v1)
            t = np.argsort(g[i, :])[(1+counter+n_neighbours):]
            mask = np.ix_(t)
            g[i, mask] = 0
        for i in range(n_samples):
            v0 = X[i, :].reshape(-1, 1)
            for j in range(length):
                v1 = X[j, :].reshape(-1, 1)
                if g[i, j] != 0:
                    current_value = g[i, j]
                    g[i, j] = current_value * weight_function(const / (current_value**n_samples))
        return g
        
    def kernel_E(self, X: np.ndarray, t_distance: float, epsilon: float, samples=None):
        n_samples = X.shape[0]
        length = n_samples
        if samples:
            length = samples
        g = np.zeros((n_samples, length))
        for i in range(n_samples):
            v0 = X[i, :].reshape(-1, 1)
            for j in range(i):
                v1 = X[j, :].reshape(-1, 1)
                dist = np.linalg.norm(v0 - v1)
                if dist <= epsilon and dist <= t_distance:
                    g[i, j] = epsilon
        g = g + g.T - np.diag(np.diag(g))
        return g

    def check_edge(self, **kwargs):
        if not ('weight' in kwargs and 'X_1' in kwargs and 'X_2' in kwargs and len(kwargs.get("X_1"))>0 and len(kwargs.get("X_2"))>0):
            raise ValueError(f"Missing arguments in check_edge: {'' if 'X_1' in kwargs else 'X_1'} {'' if 'X_2' in kwargs else 'X_2'} {'' if 'weight' in kwargs else 'weight'} {'' if 'conditions' in kwargs else 'conditions'}")
        X_1 = check_type(kwargs.get("X_1"), np.ndarray, "check_edge")
        X_2 = check_type(kwargs.get("X_2"), np.ndarray, "check_edge")
        weight = check_type(kwargs.get("weight"), float, "check_edge")
        conditions = check_type(kwargs.get("conditions"), Callable, "check_edge")
        if conditions(X_1=X_1, X_2=X_2, weight=weight):
            return True
        else:
            return False

    def get_kernel_image(self, X: np.ndarray, kernel, t_prediction: float, t_density: float, t_distance: float, epsilon: float, n_neighbours: int, K: int, radius_limit: float, density_estimator: DensityEstimator, weight_function: Callable, samples=None):
        if self.kernel_type.lower()=="kde":
            temp = kernel(X=X, t_density=t_density, t_distance=t_distance, density_estimator=density_estimator, weight_function=weight_function, samples=samples)
        elif self.kernel_type.lower()=="knn":
            temp = kernel(X=X, n_neighbours=n_neighbours, weight_function=weight_function, samples=samples)
        elif self.kernel_type.lower()=="e":
            temp = kernel(X=X, t_distance=t_distance, epsilon=epsilon, samples=samples)
        elif self.kernel_type.lower()=="gs":
            temp = kernel(X=X, t_density=t_density, t_distance=t_distance, K=K, radius_limit=radius_limit, density_estimator=density_estimator, weight_function=weight_function, samples=samples)
        else:
            temp = kernel(X=X, t_density=t_density, t_distance=t_distance, epsilon=epsilon, t_prediction=t_prediction, K=K, weight_function=weight_function, samples=samples)
        return temp

    def build_graph(self, **kwargs):
        if not ('conditions' in kwargs and 'X' in kwargs and 'kernel_image' in kwargs):
            raise ValueError(f"Missing arguments in build_graph: {'' if 'X' in kwargs else 'X'} {'' if 'kernel_image' in kwargs else 'kernel_image'} {'' if 'conditions' in kwargs else 'conditions'}")
        if 'X' in kwargs:
            if len(kwargs.get("X"))>0:
                X = check_type(kwargs.get("X"), np.ndarray, "build_graph")
            else:
                raise ValueError("Invalid argument in build_graph: X is empty")
        if 'kernel_image' in kwargs:
            if len(kwargs.get("kernel_image"))>0:
                kernel_image = check_type(kwargs.get("kernel_image"), np.ndarray, "build_graph")
            else:
                raise ValueError("Invalid argument in build_graph: kernel_image is empty")
        if 'conditions' in kwargs:
            conditions = check_type(kwargs.get("conditions"), Callable, "build_graph")
        n_samples = X.shape[0]
        g = np.zeros([n_samples, n_samples], dtype=float)
        for i in range(n_samples):
            for j in range(i):
                if self.check_edge(X_1=X[i, :], X_2=X[j, :], weight=float(kernel_image[i, j]), conditions=conditions):
                    g[i, j] = kernel_image[i, j]
                else:
                    g[i, j] = 0
        g = g + g.T - np.diag(np.diag(g))
        self.graph = g
        return g

    def explain_FACE(self, X: np.ndarray, Y: np.ndarray, factuals: np.ndarray, factuals_target: np.ndarray, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if not ((len(X)>0 and len(Y)>0) or (self.data and self.target)):
            raise ValueError("Invalid arguments in explain_FACE: (self.data and self.target) or (X and Y) needed")
        if len(X)>0 and not len(Y)>0:
            raise ValueError("Invalid arguments in explain_FACE: target needed for data; X supplied but not Y")
        if self.factuals and not self.factuals_target:
            raise ValueError("Invalid arguments in explain_FACE: self.factuals_target expected for self.factuals; factuals_target not supplied")
        if not ((self.factuals and self.factuals_target) or (len(factuals)>0 and len(factuals_target)>0)):
            raise ValueError("Invalid arguments in explain_FACE: factuals and factuals_target or self.factuals and self.factuals_target expected")
        if not (X.shape[0]==Y.shape[0]):
            raise ValueError("Invalid argument in explain_FACE: different number of points in X and Y")
        if len(factuals)>0 and len(factuals_target)>0:
            facts = factuals
            facts_target = factuals_target
        else:
            facts = self.factuals.dataset
            facts_target = self.factuals_target.dataset
        if not (facts.shape[0]==facts_target.shape[0]):
            raise ValueError("Invalid argument in explain_FACE: different number of points in factuals and factuals_target")
        target_classes = []
        if 'target_classes' in kwargs:
            target_classes = check_type(kwargs.get("target_classes"), np.ndarray, "explain_FACE")
            if not target_classes.shape == facts_target.shape:
                raise ValueError("Invalid argument in explain_face: target_classes must have the same shape as factuals_target")
        t_den = self.t_density
        t_dist = self.t_distance
        epsilon = self.epsilon
        t_pred = self.t_prediction
        k_n = self.n_neighbours
        K_ = self.K
        t_radius = self.t_radius
        kern = self.kernel
        density_estimator = self.density_estimator
        weight_function = self.weight_function
        if 'kernel' in kwargs:
            kern = check_type(kwargs.get("kernel"), Callable, "explain_FACE")
        cs_f = self.conditions
        if 'conditions' in kwargs:
            cs_f = check_type(kwargs.get("conditions"), Callable, "explain_FACE")
        pred_f = self.predict
        if 'predict' in kwargs:
            pred_f = check_type(kwargs.get("predict"), Callable, "explain_FACE")
        pred_proba_f = self.predict_proba
        if 'predict_proba' in kwargs:
            pred_proba_f = check_type(kwargs.get("predict_proba"), Callable, "explain_FACE")
        sp_f = self.shortest_path
        if 'shortest_path' in kwargs:
            sp_f = check_type(kwargs.get("shortest_path"), Callable, "explain_FACE")
        if 'density_estimator' in kwargs:
            density_estimator = check_type(kwargs.get("density_estimator"), DensityEstimator, "explain_FACE")
        if 'weight_function' in kwargs:
            weight_function = check_type(kwargs.get("weight_function"), Callable, "explain_FACE")
        if 't_density' in kwargs:
            t_den = check_type(kwargs.get("t_density"), float, "explain_FACE")
        if 't_distance' in kwargs:
            t_dist = check_type(kwargs.get("t_distance"), float, "explain_FACE")
        if 'epsilon' in kwargs:
            t_dist = check_type(kwargs.get("epsilon"), float, "explain_FACE")
        if 't_prediction' in kwargs:
            t_pred = check_type(kwargs.get("t_prediction"), float, "explain_FACE")
        if 'n_neighbours' in kwargs:
            k_n = check_type(kwargs.get("n_neighbours"), int, "explain_FACE")
        if 'K' in kwargs:
            K_ = check_type(kwargs.get("K"), int, "explain_FACE")
        if 't_radius' in kwargs:
            t_radius = check_type(kwargs.get("t_radius"), float, "explain_FACE")
            
        predictions = pred_proba_f(X)
        kernel_image = self.get_kernel_image(X, kern, t_pred, t_den, t_dist, epsilon, k_n, K_, t_radius, density_estimator, weight_function)
        graph = self.build_graph(X=X, kernel_image=kernel_image, conditions=cs_f)
        self.graph = graph
        #if not ((X == fac).all(1).any() for fac in factuals):
        #        print("Warning in explain_FACE: factuals are not a subset of X")
        #if not ((Y == fac).all(1).any() for fac in factuals_target):
        #    raise ValueError("Invalid argument in explain_FACE: factuals_target are not a subset of Y")
        classes = np.unique(Y)
        count = 0
        # 2D - List of List[int]
        candidate_targets_all = []
        # 2D - List of List[int]; final distances for each candidate_target
        dists_all = []
        # 3D - List of List of List[int]; List of paths for each candidate_target
        paths_all = []
        dists_all_best = []
        paths_all_best = []
        counterfactual_indexes = []
        counterfactuals = []
        counterfactual_targets = []
        for fac in factuals:
            if not ((X == fac).all(1).any() for fac in factuals):
                ind = -(count+1)
            else:
                ind = np.where((X == fac).all(axis=1))[0][0]
            #if (factuals_target[count]==Y[ind]):
            #    raise ValueError(f"Invalid class for factuals_target[{count}]: same class as Y[{ind}]")
            if len(target_classes)>0:
                target_class = target_classes[count]
                predictions_target_class = predictions[:, target_class]
            else:
                target_class = factuals_target[count]
                predictions_target_class = predictions[:, np.delete(classes, target_class)]
            start_node_edges = []
            if not ((X == fac).all(1).any() for fac in factuals):
                print("Warning in explain_FACE: factuals are not a subset of X")
                start_node_edges = self.get_kernel_image(fac, kern, t_pred, t_den, t_dist, epsilon, k_n, K_, density_estimator, weight_function, X.shape[0])
            t0 = np.where(predictions_target_class >= t_pred)[0]
            if len(target_classes)>0:
                t1 = np.where(Y == target_class)[0]
            else:
                t1 = [x for x in range(Y.shape[0]) if x not in np.where(Y == target_class)[0]]
            candidate_targets = list(set(t0).intersection(set(t1)))
            dists = []
            paths = []
            for candidate in candidate_targets:
                dist, path = dijkstra(graph=graph, start=ind, end=candidate, start_edges=start_node_edges)
                dists.append(dist)
                paths.append(path)
            zipped_lists = zip(dists, paths)
            sorted_zipped_lists = sorted(zipped_lists)
            paths_ = [element for _, element in sorted_zipped_lists]
            dists_ = sorted(dists)

            candidate_targets_all.append(candidate_targets)
            dists_all.append(dists_)
            paths_all.append(paths_)
            dists_all_best.append(dists_[0])
            paths_all_best.append(paths_[0])
            counterfactual_indexes.append(paths_[0][1])
            counterfactuals.append(X[paths_[0][1]])
            counterfactual_targets.append(Y[paths_[0][1]])
            count += 1

        self.graph = graph
        self.distances = dists_all
        self.distances_best = dists_all_best
        self.paths = paths_all
        self.paths_best = paths_all_best
        self.counterfactual_indexes = counterfactual_indexes
        self.counterfactuals = counterfactuals
        self.counterfactual_targets = counterfactual_targets
        self.candidate_targets = candidate_targets_all
        return (graph, dists_all_best, paths_all_best)

    def get_graph(self):
        return self.graph
        
    def get_explain_candidates(self):
        return self.candidate_targets
        
    def get_explain_distances(self):
        return self.distances_best
        
    def get_explain_paths(self):
        return self.paths_best
        
    def get_counterfactuals(self, as_indexes=False):
        if as_indexes:
            return self.counterfactual_indexes
        else:
            return self.counterfactual_targets

    def get_counterfactuals_as_data(self):
        return self.counterfactuals, self.counterfactual_targets

    def __str__(self):
        return f"Factuals: {self.factuals}, Factual Targets: {self.factuals_target}, Kernel Type: {self.kernel_type}, K-Neighbours: {self.n_neighbours}, Epsilon: {self.epsilon}, Distance Threshold: {self.t_distance}, Density Threshold: {self.t_density}, Prediction Threshold: {self.t_prediction}"