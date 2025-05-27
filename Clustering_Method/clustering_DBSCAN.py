# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.cluster import DBSCAN
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Grid_search import Grid_search_all
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_DBSCAN_clustering(data, X, eps, count_samples):  # Fundamental DBSCAN clustering
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=count_samples) # default; eps=0.5, min_samples=5
    # eps: Radius, min_samples: Minimum number of samples to qualify as a cluster
    clusters = dbscan.fit_predict(X)

    num_clusters = len(np.unique(clusters))  # Counting the number of clusters

    return clusters, num_clusters, dbscan


def clustering_DBSCAN(data, X_reduced_features, original_labels_aligned, global_known_normal_samples_pca=None, threshold_value=0.3):
    # Using internal defaults due to not passing parameter_dict when calling Grid_search_all
    parameter_dict = Grid_search_all(X_reduced_features, 'DBSCAN') 
    
    # Using internal defaults due to not passing parameter_dict when calling Grid_search_all
    # Default parameter_dict inside Grid_search_all in Grid_search.py: 'eps': 0.5, 'count_samples': 5
    eps_val = parameter_dict.get('eps', 0.5) 
    # Grid_search_all is optimized for 'min_samples', so check that key first.
    # 'count_samples' is a key of the default value inside Grid_search_all.
    min_samples_val = parameter_dict.get('min_samples', parameter_dict.get('count_samples', 5))

    print(f"DBSCAN: Using parameters eps={eps_val}, min_samples={min_samples_val}")
    
    # Perform DBSCAN Clustering
    predict_DBSCAN, num_clusters_actual, dbscan_model = clustering_DBSCAN_clustering(data, X_reduced_features, eps_val, min_samples_val)
    
    # Identify Clustering results as normal/abnormal
    # clustering_nomal_identify function takes X, original_labels_aligned, cluster_labels, n_clusters as arguments.
    final_cluster_labels_from_cni = clustering_nomal_identify(X_reduced_features, original_labels_aligned, predict_DBSCAN, num_clusters_actual, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value)
    num_clusters_after_cni = len(np.unique(final_cluster_labels_from_cni))

    return {
        'Cluster_labeling': final_cluster_labels_from_cni,
        'Best_parameter_dict': parameter_dict
    }


def pre_clustering_DBSCAN(data, X, eps, count_samples):
    # cluster_labels are model-generated labels, num_clusters_actual is the count of unique labels found by DBSCAN
    cluster_labels, num_clusters_actual, dbscan = clustering_DBSCAN_clustering(data, X, eps, count_samples)
    
    return {
        'model_labels' : cluster_labels,
        'n_clusters' : num_clusters_actual, # Actual n_clusters from DBSCAN
        'before_labeling' : dbscan
    }