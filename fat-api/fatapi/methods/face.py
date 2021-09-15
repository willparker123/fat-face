from fatapi.model import Model
from fatapi.model.estimators import Transformer
from fatapi.data import Data
from fatapi.helpers import dijkstra
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
        String specifying which kernel to use to build weights from data - default is "KDE" but can be "KDE"/"KNN"/"Epsilon"/"GS"
        -- Only required if kernel not supplied
    kernel()? : (self: FACEMethod, X_1: numpy.array, X_2: numpy.array, Y_1?: numpy.array, Y_2?: numpy.array) -> Float
        Kernel function to build weights using two datapoints
        -- Only required if kernel_type is not supplied or to override kernel function
    n_neighbours? : Int
        Number of neighbours to take into account when using KNN kernel
        -- Only required if kernel_type is "KNN"
    K? : Int
        Number of neighbours to take into account when using GS kernel
        -- Only required if kernel_type is "GS"
    t_distance? : Float
        Threshold of distance (epsilon) between nodes to constitute a non-zero weight
        -- Default is 0.25
    t_density? : Float
        Threshold of density value between nodes to constitute a non-zero weight when building feasible paths (number of neighbours when kernel_type=="KNN")
        -- Only required if kernel_type is "KDE"
        -- Default is 0.001 
    t_prediction? : Float [0-1]
        Threshold of prediction value from predict function to warrant 
        -- Default is 0.5
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
        if not (ktype.lower()=="KDE".lower() or ktype.lower()=="KNN".lower() or ktype.lower()=="E".lower() or ktype.lower()=="GS".lower()):
            raise ValueError(f"Invalid arguments in __init__: kernel_type must be 'KDE', 'KNN', 'E' or 'GS'")
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
        self.t_distance = 0.25
        self.t_prediction = 0.5
        self.t_density = 0.001
        self.n_neighbours = 20
        self.K = 10
        self._shortest_path = dijkstra
        self._conditions = lambda **kwargs: True
        self._weight_function = lambda x: -np.log(x)
        if kwargs.get('shortest_path'):
            self.shortest_path = check_type(kwargs.get("shortest_path"), Callable, "__init__")
        if kwargs.get('t_distance'):
            self.t_distance = check_type(kwargs.get("t_distance"), float, "__init__")
        if kwargs.get('t_prediction'):
            self.t_prediction = check_type(kwargs.get("t_prediction"), float, "__init__")
        if kwargs.get('t_density'):
            if not self._kernel_type=="KDE":
                print("Warning in __init__: t_density supplied but kernel may not be KDE")
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
        if self._kernel_type=="KNN":
            if not kwargs.get('n_neighbours'):
                print("Warning in __init__: n_neighbours not supplied - default (20) being used")
        if self.K=="GS":
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
        if kernel_type.lower()=="KDE".lower() or kernel_type.lower()=="KNN".lower() or kernel_type.lower()=="E".lower() or kernel_type.lower()=="GS".lower():
            self._kernel_type = kernel_type
        else:
            raise ValueError("Invalid argument in kernel.setter: kernel_type is not 'kde', 'knn', 'e' or 'gs'") 

    def kernel_KDE(self, **kwargs):
        return 0
        
    def kernel_KNN(self, **kwargs):
        return 0
        
    def kernel_E(self, **kwargs):
        return 0

    def kernel_GS(self, **kwargs):
        return 0

    def check_edge(self, **kwargs):
        if len(kwargs.get("X_1"))>0:
            X_1 = check_type(kwargs.get("X_1"), np.ndarray, "check_edge")
        if len(kwargs.get("X_2"))>0:
            X_2 = check_type(kwargs.get("X_2"), np.ndarray, "check_edge")
        if kwargs.get("weight"):
            weight = check_type(kwargs.get("weight"), float, "check_edge")
        if kwargs.get("prediction"):
            prediction = kwargs.get("prediction")
        if kwargs.get("conditions"):
            conditions = check_type(kwargs.get("conditions"), Callable, "check_edge")
        if not kwargs.get("weight")==None and kwargs.get("prediction")==None and len(kwargs.get("X_1"))<=0 and len(kwargs.get("X_2"))<=0:
            raise ValueError(f"Missing arguments in check_edge: {'' if len(kwargs.get('X_1'))>0 else 'X_1'} {'' if len(kwargs.get('X_2'))>0 else 'X_2'} {'' if kwargs.get('weight') else 'weight'} {'' if kwargs.get('prediction') else 'prediction'} {'' if kwargs.get('conditions') else 'conditions'}")
        return True

    def get_kernel_image(self, X: np.ndarray, kernel, t_prediction: float, t_density: float, t_distance: float, n_neighbours: int, K: int):
        n_samples = X.shape[0]
        g = np.zeros([n_samples, n_samples], dtype=float)
        for i in range(n_samples):
            for j in range(i):
                if self.kernel_type=="kde":
                    temp = kernel(X_1=X[i, :], X_2=X[j, :], t_density=t_density, t_distance=t_distance)
                if self.kernel_type=="knn":
                    temp = kernel(X_1=X[i, :], X_2=X[j, :], n_neighbours=n_neighbours)
                if self.kernel_type=="e":
                    temp = kernel(X_1=X[i, :], X_2=X[j, :], t_distance=t_distance)
                if self.kernel_type=="gs":
                    temp = kernel(X_1=X[i, :], X_2=X[j, :], t_density=t_density, t_distance=t_distance, K=K)
                else:
                    temp = kernel(X_1=X[i, :], X_2=X[j, :], t_density=t_density, t_distance=t_distance, t_prediction=t_prediction, K=K)
                g[i,j] = temp
        g = g + g.T - np.diag(np.diag(g))
        return g
    
    def get_predictions(self, X: np.ndarray, predict: Callable):
        n_samples = X.shape[0]
        g = []
        for x in range(n_samples):
            g.append(predict(X[x, :].reshape(1, -1))[0])
        return g

    def build_graph(self, **kwargs):
        if not (len(kwargs.get("X"))>0 and len(kwargs.get("kernel_image"))>0 and len(kwargs.get("predict_image"))>0 and kwargs.get("conditions")):
            raise ValueError(f"Missing arguments in build_graph: {'' if kwargs.get('X') else 'X'} {'' if kwargs.get('kernel_image') else 'kernel_image'} {'' if kwargs.get('predict_image') else 'predict_image'} {'' if kwargs.get('conditions') else 'conditions'}")
        if len(kwargs.get("X"))>0:
            X = check_type(kwargs.get("X"), np.ndarray, "build_graph")
        if len(kwargs.get("kernel_image"))>0:
            kernel_image = check_type(kwargs.get("kernel_image"), np.ndarray, "build_graph")
        if len(kwargs.get("predict_image"))>0:
            predict_image = check_type(kwargs.get("predict_image"), list, "build_graph")
        if kwargs.get("conditions"):
            conditions = check_type(kwargs.get("conditions"), Callable, "build_graph")
        if kwargs.get("shortest_path"):
            shortest_path = check_type(kwargs.get("shortest_path"), Callable, "build_graph")
        n_samples = X.shape[0]
        g = np.zeros([n_samples, n_samples], dtype=float)
        for i in range(n_samples):
            for j in range(i):
                if self.check_edge(X_1=X[i, :], X_2=X[j, :], weight=kernel_image[i, j], prediction=predict_image[i], conditions=conditions):
                    g[i, j] = kernel_image[i, j]
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
        t_pred = self.t_prediction
        k_n = self.n_neighbours
        K_ = self.K
        kern = self.kernel
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
        if kwargs.get("t_density"):
            t_den = check_type(kwargs.get("t_density"), float, "explain_FACE")
        if kwargs.get("t_distance"):
            t_dist = check_type(kwargs.get("t_distance"), float, "explain_FACE")
        if kwargs.get("t_prediction"):
            t_pred = check_type(kwargs.get("t_prediction"), float, "explain_FACE")
        if kwargs.get("n_neighbours"):
            k_n = check_type(kwargs.get("n_neighbours"), int, "explain_FACE")
        if kwargs.get("K"):
            K_ = check_type(kwargs.get("K"), int, "explain_FACE")
        
        predict_image = self.get_predictions(X, pred_f)
        kernel_image = self.get_kernel_image(X, kern, t_pred, t_den, t_dist, k_n, K_)

        graph = self.build_graph(X=X, kernel_image=kernel_image, predict_image=predict_image, shortest_path=sp_f, conditions=cs_f)
        self.graph = graph
        return graph

    def __str__(self):
        return f"Factuals: {self.factuals}, Factual Targets: {self.factuals_target}, Kernel Type: {self.kernel_type}, K-Neighbours: {self.n_neighbours}, Distance Threshold: {self.t_distance}, Density Threshold: {self.t_density}, Prediction Threshold: {self.t_prediction}"