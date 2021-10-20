from fatapi.model import Model
from fatapi.model.estimators import Transformer, DensityEstimator
from fatapi.data import Data
from fatapi.helpers import dijkstra, sigmoid
from typing import Callable, Tuple, Union, List
from fatapi.methods import ExplainabilityMethod
from fatapi.helpers import check_type, get_volume_of_sphere
import numpy as np

class FACEMethod(ExplainabilityMethod):
    """
    Abstract class for the FACE algorithm as an AI explainability method.
    
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
        Method for predicting the class label of X
        -- Only required if model not supplied
    model? : fatapi.model.Model
        Model object used to get prediction values and class predictions
        -- Only required if predict not supplied
    kernel_type? : str
        String specifying which kernel to use to build weights from data - default is "kde" but can be "kde"/"knn"/"e"/"gs"
        -- Only required if kernel not supplied
    kernel()? : (X: np.ndarray, weight_function: Callable[Union[float, int], float], **kwargs) -> np.ndarray
        Kernel function to build weights using two datapoints
        -- Only required if kernel_type is not supplied or to override kernel function
    n_neighbours? : int
        Number of neighbours to take into account when using knn kernel
        -- Only required if kernel_type is "knn"
        -- Default is 20
    K? : int
        Number of neighbours to take into account when using gs kernel
        -- Only required if kernel_type is "gs"
        -- Default is 10
    t_radius? : Union[float, int]
        Number of neighbours to take into account when using gs kernel
        -- Only required if kernel_type is "gs"
        -- Default is 1.10
    t_distance? : Union[float, int]
        Threshold of distance between nodes to constitute a non-zero weight
        -- Default is 10000
    epsilon? : Union[float, int]
        Value of epsilon to set when using E kernel
        -- Default is 0.75
    t_density? : Union[float, int]
        Threshold of density value between nodes to constitute a non-zero weight when building feasible paths (number of neighbours when kernel_type=="knn")
        -- Only required if kernel_type is "kde"
        -- Default is 0.001 
    t_prediction? : Union[float, int] [0-1]
        Threshold of prediction value from predict function to warrant 
        -- Default is 0.5
    density_estimator?: DensityEstimator
        Density estimator used for KDE and GS kernels; must have fit, score_samples methods
    shortest_path()? : (X: np.ndarray, start: int, end: int, start_edges: np.ndarray) -> Tuple[float, List[int]]
        Shortest path algorithm to find the path between each instance (row) of X using data 
        (datapoints corresponding to indexes [i, j] of graph) and graph (adjacency matrix)
        -- Default is dijkstra
    conditions()? : (X_1: np.ndarray, X_2: np.ndarray, weight: Union[float, int]) -> Boolean
        Additional conditions which check for feasible paths between nodes - must return a 
        -- Default is lambda **kwargs: True
    weight_function()? : (x: Union[float, int]) -> float
        Weighting function for kernels when processing kernel result
        -- Default is lambda x: -numpy.log(x)

    Methods
    -------
    explain() : (X?: np.ndarray, Y?: np.ndarray, factuals?: np.ndarray, factuals_target?: np.ndarray, 
                    kernel()?: Callable[[np.ndarray, Callable], np.ndarray],
                    conditions()?: Callable[[np.ndarray, np.ndarray, Union[float, int]], bool],
                    predict()?: Callable[[np.ndarray], np.ndarray], predict_proba()?: Callable[[np.ndarray], np.ndarray], 
                    shortest_path()?: Callable[[np.ndarray, int, int, np.ndarray], Tuple[float, List[int]]], 
                    density_estimator?: DensityEstimator, weight_function()?: Callable[[Union[float, int]], float],
                    t_distance?: Union[float, int], t_density?: Union[float, int], t_prediction?: Union[float, int],
                    epsilon?: Union[float, int], n_neighbours?: int, K?: int, t_radius?: Union[float, int]) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        Generates counterfactual datapoints from X and Y using predict function or model predict function or argument predict function
        Returns counterfactual target classes as array
        -- Uses factuals and factuals_target from preprocess_factuals if no X and Y given
    preprocess_factuals() : (factuals?: fatapi.data.Data, factuals_target?: fatapi.data.Data, model?: fatapi.model.Model, 
                                scaler?: fatapi.model.estimators.Transformer, encoder?: fatapi.model.estimators.Transformer) -> np.ndarray
        Uses encoder and scaler from black-box-model or argument to preprocess data as needed.
    build_graph() : (X: np.ndarray, kernel_image: np.ndarray, conditions(): Callable[[np.ndarray, np.ndarray, Union[float, int]], Boolean]) -> np.ndarray
        Builds graph for distances between nodes in the feature space - returns adjacency matrix of weights between [i,j]; 
        i, j are indicies of datapoints in X (rows)
    get_graph() : () -> np.ndarray
        Returns the graph which build_graph() produces
    get_explain_distances() : () -> List[float]
        Returns the distances to the counterfactuals from the starting datapoint
    get_explain_candidates() : () -> np.ndarray
        Returns the valid candidate indexes of datapoints that satisfy edge conditions from each factual datapoint
    get_explain_paths() : () -> np.ndarray
        Returns the best paths from the factuals to the counterfactuals
    get_counterfactuals(as_indexes?: bool) : () -> np.ndarray
        Returns the counterfactuals for the supplied factuals (classifications / Y values)
        -- Default is as data (not as_indexes)
    get_counterfactuals_as_data() : () -> np.ndarray
        Returns the counterfactual datapoints as data in the same form X and Y were supplied as tuple (data, target)
        -- target is the same as get_counterfactuals()
    """
    def __init__(self, *args, **kwargs) -> None:
        super(FACEMethod, self).__init__(*args, **kwargs)
        if not ('factuals' in kwargs and 'factuals_target' in kwargs):
            print("Warning in __init__: factuals and factuals_target not supplied - need to provide factuals and factuals_target to explain()")
        else:
            self._factuals = check_type(kwargs.get("factuals"), "__init__", Data)
            self._factuals_target = check_type(kwargs.get("factuals_target"), "__init__", Data)
        if not ('kernel' in kwargs or 'kernel_type' in kwargs):
            raise ValueError(f"Invalid arguments in __init__: please provide kernel or kernel_type")
        if 'data' in kwargs and not 'target' in kwargs:
            raise ValueError(f"Invalid arguments in __init__: please provide data and target or provide X and Y to explain()")
        ktype = ""
        if 'kernel_type' in kwargs:
            ktype = check_type(kwargs.get("kernel_type"), "__init__", str)
        if not (ktype.lower()=="kde" or ktype.lower()=="knn" or ktype.lower()=="e" or ktype.lower()=="gs"):
            raise ValueError(f"Invalid arguments in __init__: kernel_type must be 'kde', 'knn', 'e' or 'gs'")
        else:
            self._kernel_type = ktype
            ks = {"kde" : self.kernel_KDE, "knn" : self.kernel_KNN, "e" : self.kernel_E, "gs" : self.kernel_GS}
            self._kernel = ks[self._kernel_type.lower()]
        if 'kernel' in kwargs:
            self._kernel = check_type(kwargs.get("kernel"), "__init__", Callable[[np.ndarray, Callable], np.ndarray])
        if 'data' in kwargs:
            self.data = check_type(kwargs.get("data"), "__init__", Data)
        if 'target' in kwargs:
            self.target = check_type(kwargs.get("target"), "__init__", Data)
        self._t_distance = 10000
        self._epsilon = 0.75
        self._t_prediction = 0.5
        self._t_density = 0.001
        self._n_neighbours = 20
        self._K = 10
        self._t_radius = 1.10
        self._shortest_path = dijkstra
        self._conditions = lambda **kwargs: True
        self._weight_function = lambda x: -np.log(x)
        self._density_estimator = DensityEstimator()
        if 'shortest_path' in kwargs:
            self._shortest_path = check_type(kwargs.get("shortest_path"), "__init__", Callable[[np.ndarray, int, int, np.ndarray], Tuple[float, List[int]]])
        if self._kernel_type=="kde" and not 'density_estimator' in kwargs:
            raise ValueError("Missing argument in __init__: density_estimator required when kernel_type is 'kde'; recommended to use methods fit, score_samples from sklearn.neighbors.KernelDensity")
        if 'density_estimator' in kwargs:
            self._density_estimator = check_type(kwargs.get("density_estimator"), "__init__", DensityEstimator)
        if 't_distance' in kwargs:
            self._t_distance = check_type(kwargs.get("t_distance"), "__init__", float, int)
        if 'epsilon' in kwargs:
            self._epsilon = check_type(kwargs.get("epsilon"), "__init__", float, int)
        if 't_prediction' in kwargs:
            self._t_prediction = check_type(kwargs.get("t_prediction"), "__init__", float, int)
        if 't_density' in kwargs:
            if not self._kernel_type=="kde":
                print("Warning in __init__: t_density supplied but kernel may not be kde")
            if kwargs.get('t_density') >= 0 and kwargs.get('t_density') <= 1:
                self._t_density = check_type(kwargs.get("t_density"), "__init__", float, int)
            else:
                raise ValueError(f"Invalid argument in __init__: t_density must be between 0 and 1")
        if 'n_neighbours' in kwargs:
            self._n_neighbours = check_type(kwargs.get("n_neighbours"), "__init__", int)
        if 'K' in kwargs:
            self._K = check_type(kwargs.get("K"), "__init__", int)
        if 't_radius' in kwargs:
            self._t_radius = check_type(kwargs.get("t_radius"), "__init__", float, int)
        if 'conditions' in kwargs:
            self._conditions = check_type(kwargs.get("conditions"), "__init__", Callable[[np.ndarray, np.ndarray, Union[float, int]], bool])
        if 'weight_function' in kwargs:
            self._weight_function = check_type(kwargs.get("weight_function"), "__init__", Callable[[Union[float, int]], float])
        if self._kernel_type=="knn":
            if not 'n_neighbours' in kwargs:
                print("Warning in __init__: n_neighbours not supplied - default (20) being used")
        if self.K=="gs":
            if not 'K' in kwargs:
                print("Warning in __init__: K not supplied - default (10) being used")
        self._explain = self.explain_FACE
        self.graph = None

    @property
    def shortest_path(self) -> Callable[[np.ndarray, int, int, np.ndarray], Tuple[float, List[int]]]:
        """
        Sets and changes the shortest_path algorithm
        -------
        Callable
        """
        
        return self._shortest_path

    @shortest_path.setter
    def shortest_path(self, shortest_path) -> None:
        self._shortest_path = check_type(shortest_path, "__init__", Callable[[np.ndarray, int, int, np.ndarray], Tuple[float, List[int]]])

    @property
    def density_estimator(self) -> DensityEstimator:
        """
        Sets and changes the density_estimator attribute
        -------
        Callable
        """
        
        return self._density_estimator

    @density_estimator.setter
    def density_estimator(self, density_estimator) -> None:
        self._density_estimator = check_type(density_estimator, "__init__", DensityEstimator)

    @property
    def conditions(self) -> Callable[[np.ndarray, np.ndarray, Union[float, int]], bool]:
        """
        Sets and changes the extra conditions feasible paths must pass
        -------
        Callable
        """
        
        return self._conditions

    @conditions.setter
    def conditions(self, conds) -> None:
        self._conditions = check_type(conds, "conditions.setter", Callable[[np.ndarray, np.ndarray, Union[float, int]], bool])

    @property
    def weight_function(self) -> Callable[[Union[float, int]], float]:
        """
        Sets and changes the extra weight_function feasible paths must pass
        -------
        Callable
        """
        
        return self._weight_function

    @weight_function.setter
    def weight_function(self, weight_function) -> None:
        self._weight_function = check_type(weight_function, "weight_function.setter", Callable[[Union[float, int]], float])

    @property
    def kernel(self) -> Callable[[np.ndarray, Callable], np.ndarray]:
        """
        Sets and changes the kernel algorithm
        -------
        Callable
        """
        
        return self._kernel

    @kernel.setter
    def kernel(self, kernel) -> None:
        self._kernel = check_type(kernel, "kernel.setter", Callable[[np.ndarray, Callable], np.ndarray])

    @property
    def t_prediction(self) -> Union[float, int]:
        """
        Sets and changes the prediction threshold
        -------
        Callable
        """
        
        return self._t_prediction

    @t_prediction.setter
    def t_prediction(self, t_prediction) -> None:
        if t_prediction >= 0 and t_prediction <= 1:
            self._t_prediction = check_type(t_prediction, "t_prediction.setter", float, int)
        else:
            raise ValueError("Invalid argument in t_prediction.setter: t_prediction must be between 0 and 1 (inclusive)")

    @property
    def t_distance(self) -> Union[float, int]:
        """
        Sets and changes the distance threshold
        -------
        Callable
        """
        
        return self._t_distance

    @t_distance.setter
    def t_distance(self, t_distance) -> None:
        self._t_distance = check_type(t_distance, "t_distance.setter", float, int)
    
    @property
    def t_density(self) -> Union[float, int]:
        """
        Sets and changes the density threshold
        -------
        Callable
        """
        
        return self._t_density

    @t_density.setter
    def t_density(self, t_density) -> None:
        self._t_density = check_type(t_density, "t_density.setter", float, int)

    @property
    def t_radius(self) -> Union[float, int]:
        """
        Sets and changes the radius threshold
        -------
        Callable
        """
        
        return self._t_radius

    @t_radius.setter
    def t_radius(self, t_radius) -> None:
        self._t_radius = check_type(t_radius, "t_radius.setter", float, int)

    @property
    def epsilon(self) -> Union[float, int]:
        """
        Sets and changes the epsilon threshold
        -------
        Callable
        """
        
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon) -> None:
        self._epsilon = check_type(epsilon, "epsilon.setter", float, int)
        
    @property
    def K(self) -> int:
        """
        Sets and changes K (GS kernel neighbours)
        -------
        Callable
        """
        
        return self._K

    @K.setter
    def K(self, K) -> None:
        self._K = check_type(K, int, "K.setter")

    @property
    def n_neighbours(self) -> int:
        """
        Sets and changes n_neighbours (KNN kernel neighbours)
        -------
        Callable
        """
        
        return self._n_neighbours

    @n_neighbours.setter
    def n_neighbours(self, n_neighbours) -> None:
        self._n_neighbours = check_type(n_neighbours, int, "n_neighbours.setter")

    @property
    def kernel_type(self) -> str:
        """
        Sets and changes the kernel_type
        -------
        Callable
        """
        
        return self._kernel_type

    @kernel_type.setter
    def kernel_type(self, kernel_type) -> None:
        if kernel_type.lower()=="kde".lower() or kernel_type.lower()=="knn".lower() or kernel_type.lower()=="e".lower() or kernel_type.lower()=="gs".lower():
            self._kernel_type = kernel_type
        else:
            raise ValueError("Invalid argument in kernel.setter: kernel_type is not 'kde', 'knn', 'e' or 'gs'") 

    def kernel_KDE(self, X: np.ndarray, t_density, t_distance, density_estimator: DensityEstimator, weight_function: Callable[[Union[float, int]], float], samples=None):
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
                dist = density_estimator.distance_function(X_1, X_2)
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
        
    def kernel_GS(self, X: np.ndarray, t_density, t_distance, K: int, radius_limit, density_estimator: DensityEstimator, weight_function: Callable[[Union[float, int]], float], samples=None):
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
                dist = density_estimator.distance_function(X_1, X_2)
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

    def kernel_KNN(self, X: np.ndarray, n_neighbours: int, weight_function: Callable[[Union[float, int]], float], samples=None):
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
        
    def kernel_E(self, X: np.ndarray, t_distance: Union[float, int], epsilon: Union[float, int], samples=None):
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
        X_1 = check_type(kwargs.get("X_1"), "check_edge", np.ndarray)
        X_2 = check_type(kwargs.get("X_2"), "check_edge", np.ndarray)
        weight = check_type(kwargs.get("weight"), "check_edge", float, int)
        conditions = check_type(kwargs.get("conditions"), "check_edge", Callable[[np.ndarray, np.ndarray, Union[float, int]], bool])
        if conditions(X_1=X_1, X_2=X_2, weight=weight):
            return True
        else:
            return False

    def get_kernel_image(self, X: np.ndarray, kernel: Callable[[np.ndarray, Callable], np.ndarray], t_prediction: Union[float, int], t_density: Union[float, int], t_distance: Union[float, int], epsilon: Union[float, int], n_neighbours: int, K: int, radius_limit: Union[float, int], density_estimator: DensityEstimator, weight_function: Callable[[Union[float, int]], float], samples=None):
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
                X = check_type(kwargs.get("X"), "build_graph", np.ndarray)
            else:
                raise ValueError("Invalid argument in build_graph: X is empty")
        if 'kernel_image' in kwargs:
            if len(kwargs.get("kernel_image"))>0:
                kernel_image = check_type(kwargs.get("kernel_image"), "build_graph", np.ndarray)
            else:
                raise ValueError("Invalid argument in build_graph: kernel_image is empty")
        if 'conditions' in kwargs:
            conditions = check_type(kwargs.get("conditions"), "build_graph", Callable[[np.ndarray, np.ndarray, Union[float, int]], bool])
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
        if kwargs.get("target_classes"):
            target_classes = check_type(kwargs.get("target_classes"), "explain_FACE", np.ndarray)
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
            kern = check_type(kwargs.get("kernel"), "explain_FACE", Callable[[np.ndarray, Callable], np.ndarray])
        cs_f = self.conditions
        if 'conditions' in kwargs:
            cs_f = check_type(kwargs.get("conditions"), "explain_FACE", Callable[[np.ndarray, np.ndarray, Union[float, int]], bool])
        pred_f = self.predict
        if 'predict' in kwargs:
            pred_f = check_type(kwargs.get("predict"), "explain_FACE", Callable[[np.ndarray], np.ndarray])
        pred_proba_f = self.predict_proba
        if 'predict_proba' in kwargs:
            pred_proba_f = check_type(kwargs.get("predict_proba"), "explain_FACE", Callable[[np.ndarray], np.ndarray])
        sp_f = self.shortest_path
        if 'shortest_path' in kwargs:
            sp_f = check_type(kwargs.get("shortest_path"), "explain_FACE", Callable[[np.ndarray, int, int, np.ndarray], Tuple[float, List[int]]])
        if 'density_estimator' in kwargs:
            density_estimator = check_type(kwargs.get("density_estimator"), "explain_FACE", DensityEstimator)
        if 'weight_function' in kwargs:
            weight_function = check_type(kwargs.get("weight_function"), "explain_FACE", Callable[[Union[float, int]], float])
        if 't_density' in kwargs:
            t_den = check_type(kwargs.get("t_density"), "explain_FACE", float, int)
        if 't_distance' in kwargs:
            t_dist = check_type(kwargs.get("t_distance"), "explain_FACE", float, int)
        if 'epsilon' in kwargs:
            t_dist = check_type(kwargs.get("epsilon"), "explain_FACE", float, int)
        if 't_prediction' in kwargs:
            t_pred = check_type(kwargs.get("t_prediction"), "explain_FACE", float, int)
        if 'n_neighbours' in kwargs:
            k_n = check_type(kwargs.get("n_neighbours"), "explain_FACE", int)
        if 'K' in kwargs:
            K_ = check_type(kwargs.get("K"), "explain_FACE", int)
        if 't_radius' in kwargs:
            t_radius = check_type(kwargs.get("t_radius"), "explain_FACE", float, int)
            
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
                dist, path = sp_f(graph=graph, start=ind, end=candidate, start_edges=start_node_edges)
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
        return (counterfactual_targets)

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
        return f"Factuals: {self.factuals}, Factual Targets: {self.factuals_target}, Kernel Type: {self.kernel_type}, K (GS): {self.K}, N-Neighbours: {self.n_neighbours}, Epsilon: {self.epsilon}, Distance Threshold: {self.t_distance}, Density Threshold: {self.t_density}, Prediction Threshold: {self.t_prediction}, Radius Threshold (GS): {self.t_radius}"