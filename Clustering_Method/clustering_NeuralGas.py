# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Grid_search import Grid_search_all
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from Clustering_Method.gng_replacement import NeuralGasWithParamsSimple   # replaced Neupy

# Replacing Neupy-based function with custom implementation
def clustering_NeuralGas_clustering(data, X, n_start_nodes, max_nodes, step, max_edge_age, num_processes_for_algo=1):
    model = NeuralGasWithParamsSimple(
        n_start_nodes=n_start_nodes,
        max_nodes=max_nodes,
        step=step,
        max_edge_age=max_edge_age
    )
    model.fit(X)
    clusters = model.labels_
    num_clusters = len(np.unique(clusters))
    return clusters, num_clusters


def clustering_NeuralGas(data, X, aligned_original_labels, global_known_normal_samples_pca=None, threshold_value=0.3, num_processes_for_algo=1):
    # Grid_search_all returns a parameter_dict with best_params already applied.
    parameter_dict = Grid_search_all(X, 'NeuralGas', num_processes_for_algo=num_processes_for_algo)
    # print('parameter_dict from Grid_search_all: ', parameter_dict)

    # Get the values directly from the parameter_dict
    n_start_nodes_val = parameter_dict.get('n_start_nodes', 2)
    max_nodes_val = parameter_dict.get('max_nodes', 50)
    step_val = parameter_dict.get('step', 0.2)
    max_edge_age_val = parameter_dict.get('max_edge_age', 50)

    clusters, num_clusters = clustering_NeuralGas_clustering(
        data, X,
        n_start_nodes=n_start_nodes_val,
        max_nodes=max_nodes_val,
        step=step_val,
        max_edge_age=max_edge_age_val,
        num_processes_for_algo=num_processes_for_algo
    )

    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG NeuralGas main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG NeuralGas main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")

    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, num_clusters, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value, num_processes_for_algo=num_processes_for_algo)

    return {
        'Cluster_labeling': final_cluster_labels_from_cni,
        'Best_parameter_dict': parameter_dict
    }


# For Grid Search compatibility – use the simple class
class NeuralGasWithParams(BaseEstimator, ClusterMixin):
    def __init__(self, n_start_nodes=2, max_nodes=50, step=0.2, max_edge_age=50, num_processes_for_algo=1):
        self.n_start_nodes = n_start_nodes
        self.max_nodes = max_nodes
        self.step = step
        self.max_edge_age = max_edge_age
        self.num_processes_for_algo = num_processes_for_algo
        self.model = None
        self.clusters = None

    def fit(self, X, y=None):
        self.model = NeuralGasWithParamsSimple(
            n_start_nodes=self.n_start_nodes,
            max_nodes=self.max_nodes,
            step=self.step,
            max_edge_age=self.max_edge_age
        )
        self.model.fit(X)
        self.clusters = self.model.labels_
        self.labels_ = self.clusters
        return self

    def predict(self, X):
        if self.clusters is None:
            raise RuntimeError("Model must be fit before calling predict()")
        return self.clusters


def pre_clustering_NeuralGas(data, X, n_start_nodes, max_nodes, step, max_edge_age, num_processes_for_algo=1):
    # cluster_labels are model-generated labels, num_clusters_actual is the count of unique labels found by NeuralGas
    cluster_labels, num_clusters_actual = clustering_NeuralGas_clustering(data, X, n_start_nodes, max_nodes, step, max_edge_age, num_processes_for_algo=num_processes_for_algo)
    
    # predict_NeuralGas = clustering_nomal_identify(data, cluster_labels, num_clusters_actual)
    # num_clusters = len(np.unique(predict_NeuralGas))  # Counting the number of clusters

    # For NeuralGas, the 'before_labeling' might be the model itself if its state is useful, or just cluster_labels.
    # Here, returning cluster_labels for consistency with other pre_clustering functions that return labels or simple model objects.
    # If the NeuralGasWithParamsSimple model object is needed by tuning methods, this might need adjustment.
    neural_gas_model_placeholder = cluster_labels # Or potentially the model from clustering_NeuralGas_clustering if it's serializable and useful

    return {
        'model_labels' : cluster_labels,
        'n_clusters': num_clusters_actual, # Actual n_clusters from NeuralGas
        'before_labeling': neural_gas_model_placeholder 
    }
