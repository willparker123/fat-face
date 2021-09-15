from fatapi.model import Model
from fatapi.model.estimators import Transformer, DensityEstimator
from fatapi.data import Data
from fatapi.helpers import dijkstra, sigmoid
from typing import Callable, Tuple, Union, List
from fatapi.methods import ExplainabilityMethod
from fatapi.helpers import check_type
import numpy as np

class FACEMethod(ExplainabilityMethod):
    """
    Abstract class for the FACE algorithm as an AI explainability method.
    
    Contains methods for generating counterfactuals and the data / model to apply FACE to, 
    as well as visualisation and extracting internal representations
    
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
    predict()? : (X: np.ndarray) -> np.ndarray)
        Method for predicting the class label of X
        -- Only required if model not supplied
    model? : fatapi.model.Model
        Model object used to get prediction values and class predictions
        -- Only required if predict not supplied
    explain()? : (self, X: numpy.array, Y?: numpy.array, predict()?: Callable) -> numpy.array
        Generates counterfactual datapoints from X and Y using predict function or model predict function or argument predict function
    kernel_type? : String
        String specifying which kernel to use to build weights from data - default is "kde" but can be "kde"/"knn"/"e"/"gs"
        -- Only required if kernel not supplied
    kernel()? : (self: FACEMethod, X_1: numpy.array, X_2: numpy.array, Y_1?: numpy.array, Y_2?: numpy.array) -> Float
        Kernel function to build weights using two datapoints
        -- Only required if kernel_type is not supplied or to override kernel function
    n_neighbours? : Int
        Number of neighbours to take into account when using knn kernel
        -- Only required if kernel_type is "knn"
    K? : Int
        Number of neighbours to take into account when using gs kernel
        -- Only required if kernel_type is "gs"
    t_distance? : Float
        Threshold of distance between nodes to constitute a non-zero weight
        -- Default is 10000
    epsilon? : Float
        Value of epsilon to set when using E kernel
        -- Default is 0.75
    t_density? : Float
        Threshold of density value between nodes to constitute a non-zero weight when building feasible paths (number of neighbours when kernel_type=="knn")
        -- Only required if kernel_type is "kde"
        -- Default is 0.001 
    t_prediction? : Float [0-1]
        Threshold of prediction value from predict function to warrant 
        -- Default is 0.5
    density_estimator?: DensityEstimator
        Density estimator used for KDE and GS kernels; must have fit, score_samples methods
    shortest_path()? : (X: numpy.array, graph: numpy.array, data: numpy.array)
        Shortest path algorithm to find the path between each instance (row) of X using data 
        (datapoints corresponding to indexes [i, j] of graph) and graph (adjacency matrix)
        -- Default is dijkstra
    conditions()? : (X_1: numpy.array, X_2: numpy.array, Y_1?: numpy.array, Y_2?: numpy.array) -> Boolean
        Additional conditions which check for feasible paths between nodes - must return a 
        -- Default is lambda **kwargs: True
    weight_function()? : (x: float) -> float
        Weighting function for kernels when processing kernel result
        -- Default is lambda x: -numpy.log(x)

    Methods
    -------
    explain() : (X?: numpy.array, Y?: numpy.array, predict()?: Callable) -> numpy.array
        Generates counterfactual datapoints from X and Y using predict function or model predict function or argument predict function
        -- Uses factuals and factuals_target from preprocess_factuals if no X and Y given
    preprocess_factuals() : (factuals?: fatapi.data.Data, factuals_target?: fatapi.data.Data, model?: fatapi.model.Model, 
                                            scaler?: fatapi.model.estimators.Transformer, encoder?: fatapi.model.estimators.Transformer) -> numpy.array
        Uses encoder and scaler from black-box-model or argument to preprocess data as needed.
    build_graph() : (X: numpy.array, Y: numpy.array, t_distance?: Float, t_density?: Float, 
                                    t_prediction?: Float, conditions()?: Callable) -> numpy.array
        Builds graph for distances between nodes in the feature space - returns adjacency matrix of weights between [i,j]; 
        i, j are indicies of datapoints in X (rows)
    get_graph() : () -> numpy.array
        Returns the graph which build_graph() produces
    """
    def __init__(self, *args, **kwargs) -> None:
        super(FACEMethod, self).__init__(*args, **kwargs)
        if not kwargs.get("factuals") and kwargs.get("factuals_target"):
            print("Warning in __init__: factuals and factuals_target not supplied - need to provide X and Y to explain()")
        if not (kwargs.get('kernel') or kwargs.get('kernel_type')):
            raise ValueError(f"Invalid arguments in __init__: please provide kernel or kernel_type")
        if kwargs.get('data') and not kwargs.get('target'):
            raise ValueError(f"Invalid arguments in __init__: please provide data and target or provide X and Y to explain()")
        ktype = ""
        if kwargs.get('kernel_type'):
            ktype = check_type(kwargs.get("kernel_type"), str, "__init__")
        if not (ktype.lower()=="kde".lower() or ktype.lower()=="knn".lower() or ktype.lower()=="e".lower() or ktype.lower()=="gs".lower()):
            raise ValueError(f"Invalid arguments in __init__: kernel_type must be 'kde', 'knn', 'e' or 'gs'")
        else:
            self._kernel_type = ktype
            ks = {"kde" : self.kernel_KDE, "knn" : self.kernel_KNN, "e" : self.kernel_E, "gs" : self.kernel_GS}
            self._kernel = ks[self._kernel_type.lower()]
        if kwargs.get('kernel'):
            self._kernel = check_type(kwargs.get("kernel"), Callable, "__init__")
        if kwargs.get('data'):
            self.data = check_type(kwargs.get("data"), Data, "__init__")
        if kwargs.get('target'):
            self.target = check_type(kwargs.get("target"), Data, "__init__")
        self.t_distance = 10000
        self.epsilon = 0.75
        self.t_prediction = 0.5
        self.t_density = 0.001
        self.n_neighbours = 20
        self.K = 10
        self._shortest_path = dijkstra
        self._conditions = lambda **kwargs: True
        self._weight_function = lambda x: -np.log(x)
        self._density_estimator = DensityEstimator()
        if kwargs.get('shortest_path'):
            self._shortest_path = check_type(kwargs.get("shortest_path"), Callable, "__init__")
        if self._kernel_type=="kde" and not kwargs.get('density_estimator'):
            raise ValueError("Missing argument in __init__: density_estimator required when kernel_type is 'kde'; recommended to use methods fit, score_samples from sklearn.neighbors.KernelDensity")
        if kwargs.get('density_estimator'):
            self._density_estimator = check_type(kwargs.get("density_estimator"), DensityEstimator, "__init__")
        if kwargs.get('t_distance'):
            self.t_distance = check_type(kwargs.get("t_distance"), float, "__init__")
        if kwargs.get('epsilon'):
            self.epsilon = check_type(kwargs.get("epsilon"), float, "__init__")
        if kwargs.get('t_prediction'):
            self.t_prediction = check_type(kwargs.get("t_prediction"), float, "__init__")
        if kwargs.get('t_density'):
            if not self._kernel_type=="kde":
                print("Warning in __init__: t_density supplied but kernel may not be kde")
            if kwargs.get('t_density') >= 0 and kwargs.get('t_density') < 1:
                self.t_density = check_type(kwargs.get("t_density"), float, "__init__")
            else:
                raise ValueError(f"Invalid argument in __init__: t_density must be between 0 and 1")
        if kwargs.get('n_neighbours'):
            self.n_neighbours = check_type(kwargs.get("n_neighbours"), int, "__init__")
        if kwargs.get('K'):
            self.K = check_type(kwargs.get("K"), int, "__init__")
        if kwargs.get('conditions'):
            self._conditions = check_type(kwargs.get("conditions"), Callable, "__init__")
        if kwargs.get('weight_function'):
            self._weight_function = check_type(kwargs.get("weight_function"), Callable, "__init__")
        if self._kernel_type=="knn":
            if not kwargs.get('n_neighbours'):
                print("Warning in __init__: n_neighbours not supplied - default (20) being used")
        if self.K=="gs":
            if not kwargs.get('K'):
                print("Warning in __init__: K not supplied - default (10) being used")
        self._explain = self.explain_FACE
        self.graph = None

    @property
    def shortest_path(self) -> Callable:
        """
        Sets and changes the shortest_path algorithm
        -------
        Callable
        """
        
        return self._shortest_path

    @shortest_path.setter
    def shortest_path(self, shortest_path) -> None:
        self._shortest_path = check_type(shortest_path, Callable, "__init__")

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
        self._density_estimator = check_type(density_estimator, DensityEstimator, "__init__")

    @property
    def conditions(self) -> bool:
        """
        Sets and changes the extra conditions feasible paths must pass
        -------
        Callable
        """
        
        return self._conditions

    @conditions.setter
    def conditions(self, conds) -> None:
        self._conditions = check_type(conds, Callable, "conditions.setter")

    @property
    def weight_function(self) -> float:
        """
        Sets and changes the extra weight_function feasible paths must pass
        -------
        Callable
        """
        
        return self._weight_function

    @weight_function.setter
    def weight_function(self, weight_function) -> None:
        self._weight_function = check_type(weight_function, Callable, "weight_function.setter")

    @property
    def kernel(self) -> Callable:
        """
        Sets and changes the kernel algorithm
        -------
        Callable
        """
        
        return self._kernel

    @kernel.setter
    def kernel(self, kernel) -> None:
        self._kernel = check_type(kernel, Callable, "kernel.setter")

    @property
    def kernel_type(self) -> Callable:
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

    def kernel_KDE(self, X: np.ndarray, t_density: float, t_distance: float, density_estimator: DensityEstimator, weight_function: Callable):
        density_estimator.fit(X)
        n_samples = X.shape[0]
        g = np.zeros([n_samples, n_samples], dtype=float)
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
                    if density <= t_density:
                        g[i,j] = weight_function(np.exp(density)) * dist
                    else:
                        g[i,j] = 0
                else:
                    g[i,j] = 0
        g = g + g.T - np.diag(np.diag(g))
        return g
        
    def kernel_GS(self, X: np.ndarray, t_density: float, t_distance: float, K: int, density_estimator:  DensityEstimator, weight_function: Callable):
        density_estimator.fit(X)
        n_samples = X.shape[0]
        g = np.zeros([n_samples, n_samples], dtype=float)
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
                    if density <= t_density:
                        g[i,j] = weight_function(sigmoid(density)) * dist
                    else:
                        g[i,j] = 0
                else:
                    g[i,j] = 0
        g = g + g.T - np.diag(np.diag(g))
        return g

    def kernel_KNN(self, X: np.ndarray, n_neighbours: int, weight_function: Callable):
        volume_sphere = get_volume_of_sphere(self.n_features)
        const = (self.n_neighbours / (self.n_samples * volume_sphere))**(1/self.n_features)
        k = npo.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            counter = 0
            for j in range(self.n_samples):
                v1 = self.X[j, :].reshape(-1, 1)
                k[i, j] = npo.linalg.norm(v0 - v1)
# =============================================================================
#                 if self.edge_coditions(v0, v1):
#                     dist = npo.linalg.norm(v0 - v1)
#                     k[i, j] = dist
#                 else:
#                     counter += 1
# =============================================================================
            t = npo.argsort(k[i, :])[(1+counter+self.n_neighbours):]
            mask = npo.ix_(t)
            k[i, mask] = 0
            
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            for j in range(self.n_samples):
                v1 = self.X[j, :].reshape(-1, 1)
                if k[i, j] != 0:
                    current_value = k[i, j]
                    k[i, j] = current_value * self.weight_function(const / (current_value**self.n_features))
        return k
        
    def kernel_E(self, X: np.ndarray, t_distance: float, epsilon: float):
        return 0

    def check_edge(self, **kwargs):
        if not kwargs.get("weight")==None and kwargs.get("prediction_X_1")==None and kwargs.get("prediction_X_2")==None and len(kwargs.get("X_1"))<=0 and len(kwargs.get("X_2"))<=0 and kwargs.get("t_prediction"):
            raise ValueError(f"Missing arguments in check_edge: {'' if len(kwargs.get('X_1'))>0 else 'X_1'} {'' if len(kwargs.get('X_2'))>0 else 'X_2'} {'' if kwargs.get('weight') else 'weight'} {'' if kwargs.get('prediction_X_1') else 'prediction_X_1'} {'' if kwargs.get('prediction_X_2') else 'prediction_X_2'} {'' if kwargs.get('t_prediction') else 't_prediction'} {'' if kwargs.get('conditions') else 'conditions'}")
        X_1 = check_type(kwargs.get("X_1"), np.ndarray, "check_edge")
        X_2 = check_type(kwargs.get("X_2"), np.ndarray, "check_edge")
        weight = check_type(kwargs.get("weight"), float, "check_edge")
        t_prediction = check_type(kwargs.get("t_prediction"), float, "check_edge")
        prediction_X_1 = kwargs.get("prediction_X_1")
        prediction_X_2 = kwargs.get("prediction_X_2")
        conditions = check_type(kwargs.get("conditions"), Callable, "check_edge")
        if prediction_X_1 <= t_prediction and prediction_X_2 <= t_prediction and conditions(X_1=X_1, X_2=X_2, weight=weight):
            return True
        else:
            return False

    def get_kernel_image(self, X: np.ndarray, kernel, t_prediction: float, t_density: float, t_distance: float, epsilon: float, n_neighbours: int, K: int, density_estimator: DensityEstimator, weight_function: Callable):
        if self.kernel_type.lower()=="kde":
            temp = kernel(X=X, t_density=t_density, t_distance=t_distance, density_estimator=density_estimator, weight_function=weight_function)
        elif self.kernel_type.lower()=="knn":
            temp = kernel(X=X, n_neighbours=n_neighbours, weight_function=weight_function)
        elif self.kernel_type.lower()=="e":
            temp = kernel(X=X, t_distance=t_distance, epsilon=epsilon)
        elif self.kernel_type.lower()=="gs":
            temp = kernel(X=X, t_density=t_density, t_distance=t_distance, K=K, density_estimator=density_estimator, weight_function=weight_function)
        else:
            temp = kernel(X=X, t_density=t_density, t_distance=t_distance, epsilon=epsilon, t_prediction=t_prediction, K=K, weight_function=weight_function)
        return temp
    
    def get_predictions(self, X: np.ndarray, predict: Callable):
        n_samples = X.shape[0]
        g = []
        for x in range(n_samples):
            g.append(predict(X[x, :].reshape(1, -1))[0])
        return g

    def build_graph(self, **kwargs):
        if not (len(kwargs.get("X"))>0 and len(kwargs.get("kernel_image"))>0 and len(kwargs.get("predict_image"))>0 and kwargs.get("conditions") and kwargs.get("t_prediction")):
            raise ValueError(f"Missing arguments in build_graph: {'' if kwargs.get('X') else 'X'} {'' if kwargs.get('kernel_image') else 'kernel_image'} {'' if kwargs.get('predict_image') else 'predict_image'} {'' if kwargs.get('t_prediction') else 't_prediction'} {'' if kwargs.get('conditions') else 'conditions'}")
        if len(kwargs.get("X"))>0:
            X = check_type(kwargs.get("X"), np.ndarray, "build_graph")
        if len(kwargs.get("kernel_image"))>0:
            kernel_image = check_type(kwargs.get("kernel_image"), np.ndarray, "build_graph")
        if len(kwargs.get("predict_image"))>0:
            predict_image = check_type(kwargs.get("predict_image"), list, "build_graph")
        if kwargs.get("conditions"):
            conditions = check_type(kwargs.get("conditions"), Callable, "build_graph")
        if kwargs.get("t_prediction"):
            t_prediction = check_type(kwargs.get("t_prediction"), float, "build_graph")
        if kwargs.get("shortest_path"):
            shortest_path = check_type(kwargs.get("shortest_path"), Callable, "build_graph")
        n_samples = X.shape[0]
        g = np.zeros([n_samples, n_samples], dtype=float)
        for i in range(n_samples):
            for j in range(i):
                if self.check_edge(X_1=X[i, :], X_2=X[j, :], weight=float(kernel_image[i, j]), prediction_X_1=predict_image[i], prediction_X_2=predict_image[j], t_prediction=t_prediction, conditions=conditions):
                    g[i, j] = kernel_image[i, j]
                else:
                    g[i, j] = 0
        g = g + g.T - np.diag(np.diag(g))
        return g

    def explain_FACE(self, X: np.ndarray, Y: np.ndarray, factuals: np.ndarray, factuals_target: np.ndarray, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if not (len(X)>0 and len(Y)>0):
            raise ValueError("Invalid arguments in explain_FACE: (self.data and self.target) or (X and Y) needed")
        if len(X)>0 and not len(Y)>0:
            raise ValueError("Invalid arguments in explain_FACE: target needed for data; X supplied but not Y")
        if self.factuals and not self.factuals_target:
            raise ValueError("Invalid arguments in explain_FACE: self.factuals_target expected for self.factuals; factuals_target not supplied")
        if not ((self.factuals and self.factuals_target) or (len(factuals)>0 and len(factuals_target)>0)):
            raise ValueError("Invalid arguments in explain_FACE: factuals and factuals_target or self.factuals and self.factuals_target expected")
        if not (X.shape[0]==Y.shape[0]):
            raise ValueError("Invalid argument in explain_FACE: different number of points in X and Y")
        t_den = self.t_density
        t_dist = self.t_distance
        epsilon = self.epsilon
        t_pred = self.t_prediction
        k_n = self.n_neighbours
        K_ = self.K
        kern = self.kernel
        density_estimator = self.density_estimator
        weight_function = self.weight_function
        if kwargs.get("kernel"):
            kern = check_type(kwargs.get("kernel"), Callable, "explain_FACE")
        cs_f = self.conditions
        if kwargs.get("conditions"):
            cs_f = check_type(kwargs.get("conditions"), Callable, "explain_FACE")
        pred_f = self.predict
        if kwargs.get("predict"):
            pred_f = check_type(kwargs.get("predict"), Callable, "explain_FACE")
        sp_f = self.shortest_path
        if kwargs.get("shortest_path"):
            sp_f = check_type(kwargs.get("shortest_path"), Callable, "explain_FACE")
        if kwargs.get("density_estimator"):
            density_estimator = check_type(kwargs.get("density_estimator"), DensityEstimator, "explain_FACE")
        if kwargs.get("weight_function"):
            weight_function = check_type(kwargs.get("weight_function"), Callable, "explain_FACE")
        if kwargs.get("t_density"):
            t_den = check_type(kwargs.get("t_density"), float, "explain_FACE")
        if kwargs.get("t_distance"):
            t_dist = check_type(kwargs.get("t_distance"), float, "explain_FACE")
        if kwargs.get("epsilon"):
            t_dist = check_type(kwargs.get("epsilon"), float, "explain_FACE")
        if kwargs.get("t_prediction"):
            t_pred = check_type(kwargs.get("t_prediction"), float, "explain_FACE")
        if kwargs.get("n_neighbours"):
            k_n = check_type(kwargs.get("n_neighbours"), int, "explain_FACE")
        if kwargs.get("K"):
            K_ = check_type(kwargs.get("K"), int, "explain_FACE")
        
        predict_image = self.get_predictions(X, pred_f)
        kernel_image = self.get_kernel_image(X, kern, t_pred, t_den, t_dist, epsilon, k_n, K_, density_estimator, weight_function)

        graph = self.build_graph(X=X, kernel_image=kernel_image, predict_image=predict_image, shortest_path=sp_f, t_prediction=t_pred, conditions=cs_f)
        self.graph = graph
        return graph

    def __str__(self):
        return f"Factuals: {self.factuals}, Factual Targets: {self.factuals_target}, Kernel Type: {self.kernel_type}, K-Neighbours: {self.n_neighbours}, Epsilon: {self.epsilon}, Distance Threshold: {self.t_distance}, Density Threshold: {self.t_density}, Prediction Threshold: {self.t_prediction}"