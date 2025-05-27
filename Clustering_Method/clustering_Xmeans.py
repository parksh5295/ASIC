# input 'X' is X_reduced or X rows
# Clustering Algorithm: X-means; Autonomously tuning n_clusters in k-means
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import silhouette_score
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Grid_search import Grid_search_all
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
import numpy as np

# X-Means Clustering Function
def x_means_clustering(X, random_state, max_clusters, n_init=30):
    best_score = -1
    best_model = None
    best_k = 2
    for k in range(2, max_clusters + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        
        silhouette_avg = silhouette_score(X, labels)
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_model = model
            best_k = k
    return best_model, best_k

def clustering_Xmeans_clustering(data, X, random_state, max_clusters, n_init=30):  # Fundamental Xmeans clustering, Added n_init
    # default; max=clusters=10,
    # Perform Y-Means Clustering
    model, optimal_k = x_means_clustering(X, random_state, max_clusters, n_init=n_init) # Pass n_init
    if model is None: # x_means_clustering might return None if no valid k is found
        print("[Warning clustering_Xmeans_clustering] x_means_clustering returned no model. Defaulting to k=2 or empty labels.")
        # Handle this case: maybe run a simple KMeans with k=2 as a fallback or return error indicating labels
        # For now, let's assume if model is None, optimal_k might also be problematic, return empty or default.
        # This depends on how x_means_clustering handles the case where all k fail.
        # Based on current x_means_clustering, best_model could remain None if all silhouette_scores are <= -1 (initial best_score)
        # Or if all k iterations are skipped due to < 2 unique labels.
        # A more robust x_means_clustering would ensure it always returns *some* model or raises error.
        # Assuming if model is None, we can't get labels. Returning empty labels and default k=0 or 2.
        return np.array([]), 2 # Or raise an error

    clusters = model.labels_

    return clusters, optimal_k


def clustering_Xmeans(data, X, aligned_original_labels, global_known_normal_samples_pca=None, threshold_value=0.3):
    # Grid_search_all returns a parameter_dict with best_params already applied.
    parameter_dict = Grid_search_all(X, 'Xmeans')

    # Get the values directly from the parameter_dict
    random_state_val = parameter_dict.get('random_state', 42)
    max_clusters_val = parameter_dict.get('max_clusters', 1000) # Default value or desired value
    n_init_val = parameter_dict.get('n_init', 30) # Default value or desired value

    clusters, num_clusters = clustering_Xmeans_clustering(data, X, 
                                                        random_state=random_state_val, 
                                                        max_clusters=max_clusters_val,
                                                        n_init=n_init_val) # Pass n_init

    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG XMeans main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG XMeans main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")

    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, num_clusters, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value)

    # predict_Xmeans = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': parameter_dict
    }


# Auxiliary class for Grid Search
class XMeansWrapper(BaseEstimator, ClusterMixin):
    def __init__(self, random_state=42, max_clusters=10, n_init=30):
        # Automatically assign a value for __init__ if no input value is present
        self.random_state = random_state
        self.max_clusters = max_clusters
        self.n_init = n_init
        self.model = None
        self.best_k = None

    def fit(self, X, y=None):
        self.model, self.best_k = x_means_clustering(X, self.random_state, self.max_clusters, n_init=self.n_init)
        return self

    def predict(self, X):
        return self.model.predict(X) if self.model else None

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)
    

def pre_clustering_Xmeans(data, X, random_state, max_clusters, n_init=30):
    # clusters are model-generated labels before CNI, num_clusters_optimal is the k found by x_means_clustering
    cluster_labels, num_clusters_optimal = clustering_Xmeans_clustering(data, X, random_state, max_clusters, n_init=n_init)

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : num_clusters_optimal, # Optimal n_clusters from XMeans
        'before_labeling' : cluster_labels # or the model from x_means_clustering if needed, but usually labels are sufficient for pre_clustering
    }