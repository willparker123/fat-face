import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
import networkx as nx


class BaseFACE:
    """Base class for CE methods based on shortest path over a graph

    Attributes:
        data: pandas DataFrame containing data (n_samples, n_features)
        clf: binary classifier with a predict method
        dist_metric: metric for scipy.spatial.distance.cdist function
        dist_threshold: maximum distance between nodes for edge to be added
        pred_threshold: prediction threshold for classification of CE
    """
    def __init__(self, data, clf, dist_metric='euclidean', dist_threshold=20, pred_threshold=None):
        """ Inits BaseFACE class
        """
        self.data = data
        self.clf = clf
        self.dist_metric = dist_metric
        self.dist_threshold = dist_threshold
        self.pred_threshold = pred_threshold
        self.G = nx.Graph()

    def _threshold_function(self, x, y):
        """Distance function that determines whether to add edge to graph

        Args:
            x: array
            y: array

        Returns:
            distance between points
        """
        dist = cdist(x, y, metric=self.dist_metric)
        print(dist)
        return dist

    def _weight_function(self, x, y):
        """Distance or density function that calculates weights for graph

        Default uses distance measure for weights but can be overridden for subclasses.

        Args:
            x: point 1
            y: point 2

        Returns:
            weight between points
        """
        return self._threshold_function(x, y)

    def _calculate_weights(self, x, y):
        """Calculate edge weights using on weight function if it meets distance and weight thresholds

        Args:
            x: point 1
            y: point 2

        Returns:
            weight for edge between x and y
        """
        if self._threshold_function(x, y) < self.dist_threshold:
            weight = self._weight_function(x, y)
            return weight
        else:
            return None

    def prune_nodes(self):
        """Method to remove nodes that do not meet a condition or threshold

        Returns:

        """
        pass

    def prune_edges(self):
        """Method to remove edges that do not meet a threshold

        Returns:

        """
        pass

    def add_nodes_and_edges(self, new_point=False):
        """Creates nodes and edges with weights.

        If new_point is False then creates nodes and edges (if threshold is met) for all data points.
        If now_point is True then creates 1 extra node and adds edges to all other nodes.

        Args:
            new_point: boolean whether a new point has just been added to data

        Returns:

        """

        edge_weights = []

        if new_point is False:
            self.G.add_nodes_from(list(self.data.index))
            print(f'{self.G.number_of_nodes()} nodes have been added to graph.')
            self.prune_nodes()

            dist_matrix = pd.DataFrame(self._threshold_function(self.data.values, self.data.values),
                                       index=self.data.index, columns=self.data.index)
            weight_matrix = pd.DataFrame(self._weight_function(self.data.values, self.data.values),
                                         index=self.data.index, columns=self.data.index)

            nodes = list(self.G.nodes)
            for i, node_from in enumerate(nodes[:-1]):
                for node_to in nodes[i + 1:]:
                    dist = dist_matrix.loc[node_from][node_to]
                    if dist < self.dist_threshold:
                        weight = weight_matrix.loc[node_from][node_to]
                        edge_weights.append((node_from, node_to, {'weight': weight}))

            self.G.add_edges_from(edge_weights)
            print(f'{self.G.number_of_edges()} edges have been added to graph.')
            self.prune_edges()

        else:
            new_node = list(self.G.nodes)[-1] + 1
            self.G.add_node(new_node)
            print(f'1 node has been added to graph.')
            self.prune_nodes()

            dist_matrix = pd.DataFrame(self._threshold_function(self.data.values[-1].reshape(1, -1),
                                                                self.data.values[:-1]),
                                       index=[self.data.index[-1]], columns=self.data.index[:-1])
            weight_matrix = pd.DataFrame(self._weight_function(self.data.values[-1].reshape(1, -1),
                                                               self.data.values[:-1]),
                                         index=[self.data.index[-1]], columns=self.data.index[:-1])

            for node_to in list(self.G.nodes)[:-1]:
                dist = dist_matrix.loc[new_node][node_to]
                if dist < self.dist_threshold:
                    weight = weight_matrix.loc[new_node][node_to]
                    edge_weights.append((new_node, node_to, {'weight': weight}))

            self.G.add_edges_from(edge_weights)
            print(f'{len(edge_weights)} edges have been added.')
            self.prune_edges()

    def generate_counterfactual(self, instance):
        """Generates counterfactual to flip prediction of example using dijkstra shortest path for the graph created

        Args:
            instance: instance to generate CE

        Returns:
            path to CE as instances from data as a pandas DataFrame
            probability of CE (if prediction_prob specified)

        """
        if self.G.number_of_nodes() == 0:
            self.add_nodes_and_edges()

        instance = instance.reshape(1, -1)
        target = int(abs(self.clf.predict(instance) - 1))
        target_data = self.data[self.clf.predict(self.data) == target]
        target_nodes = list(set(list(self.G.nodes())).intersection(target_data.index))

        example_in_data = self.data[self.data.eq(instance)].dropna()
        if len(example_in_data) > 0:
            start_node = example_in_data.iloc[0].name
        else:
            start_node = self.data.iloc[-1].name + 1
            self.data = self.data.append(pd.Series(instance.squeeze(), index=list(self.data), name=start_node))

            self.add_nodes_and_edges(new_point=True)

        assert start_node in list(self.G.nodes), "Instance does not meet thresholds"

        _, path = nx.multi_source_dijkstra(self.G, target_nodes, target=start_node)
        path = path[::-1]

        if self.pred_threshold is not None:
            pred_prob = self.clf.predict_proba(target_data.loc[path[-1]].values.reshape(1, -1)).squeeze()[target]

            while pred_prob < self.pred_threshold:
                target_data = target_data.drop(path[-1])
                target_nodes = list(set(list(self.G.nodes())).intersection(target_data.index))
                assert len(target_nodes) > 0, "Target nodes do not meet thresholds"
                _, path = nx.multi_source_dijkstra(self.G, target_nodes, target=start_node)
                path = path[::-1]
                pred_prob = self.clf.predict_proba(target_data.loc[path[-1]].values.reshape(1, -1)).squeeze()[target]

            path_df = self.data.loc[path].reset_index(drop=True)
            pred = self.clf.predict(path_df).astype(int).reshape(-1, 1)
            prob = np.take_along_axis(self.clf.predict_proba(path_df), pred, axis=1)
            prob_df = pd.DataFrame(np.concatenate((pred, prob), axis=1), columns=['prediction', 'probability'])

            return path_df, prob_df

        path_df = self.data.loc[path].reset_index(drop=True)
        pred = self.clf.predict(path_df).astype(int).reshape(-1, 1)
        prob_df = pd.DataFrame(pred, columns=['prediction'])

        return path_df, prob_df


class FACE(BaseFACE):
    """
    Implementation of Poyiadzi et al (2020) FACE: Feasible and Actionable Counterfactual Explanations

    Attributes:
        kde: kernel-density estimator
        density_threshold: low density threshold to prune nodes from graph

    """
    def __init__(self, data, clf, dist_metric='euclidean', dist_threshold=1, pred_threshold=None, kde=gaussian_kde,
                 density_threshold=0.01):
        super().__init__(data, clf, dist_metric, dist_threshold, pred_threshold)
        """Inits FACE class
        """
        self.kde = kde(data.T)
        self.density_threshold = density_threshold

    def _weight_function(self, x, y):
        """Weights based on kernel-density estimator

        Args:
            x: point 1
            y: point 2

        Returns:
            weight for edge between points
        """
        dist = self._threshold_function(x, y)
        weight = -np.log(self.kde((x.T + y.T) / 2) * (dist + 1e-10))
        return weight

    def prune_nodes(self):
        """removes nodes that do not meet density threshold

        Returns:

        """
        low_density = self.data[self.kde(self.data.T) < self.density_threshold].index
        self.G.remove_nodes_from(low_density)
        self.data = self.data[self.data.index.isin(list(self.G.nodes()))]
        if len(low_density) > 0:
            print(f'{len(low_density)} nodes removed due to low density.')
