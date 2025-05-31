# Clustering Method = Spherical GMM
# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.mixture import GaussianMixture
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Elbow_method import Elbow_method
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from Clustering_Method.clustering_GMM import fit_gmm_with_retry


def clustering_SGMM(data, X, max_clusters, original_labels_aligned, global_known_normal_samples_pca=None, threshold_value=0.3, num_processes_for_algo=1):
    # Define an initial parameter_dict that includes reg_covar
    # This will be passed to Elbow_method, which might use its own default if this is None,
    # but it's good practice to define it here for clarity and control.
    initial_parameter_dict = {
        'random_state': 42,
        'reg_covar': 1e-6, # Default initial reg_covar for SGMM
        'n_init': 30      # Changed default n_init for SGMM to 30
    }

    # Elbow_method now expects a parameter_dict and will use reg_covar from it for SGMM
    sgmm_specific_max_clusters = min(max_clusters, 50) # SGMM only tests up to 50
    after_elbow = Elbow_method(data, X, 'SGMM', sgmm_specific_max_clusters, parameter_dict=initial_parameter_dict.copy(), num_processes_for_algo=num_processes_for_algo)
    n_clusters = after_elbow['optimal_cluster_n']
    
    # The parameter_dict returned by Elbow_method should contain the used random_state and reg_covar
    parameter_dict_from_elbow = after_elbow['best_parameter_dict']

    # Extract parameters for fit_gmm_with_retry, providing defaults
    random_state_val = parameter_dict_from_elbow.get('random_state', 42)
    reg_covar_init_val = parameter_dict_from_elbow.get('reg_covar', 1e-6) # Initial reg_covar for retry
    n_init_val = parameter_dict_from_elbow.get('n_init', 1) # n_init for GMM
    # max_reg_covar_val can be a fixed value or also from elbow if tuned
    max_reg_val = parameter_dict_from_elbow.get('max_reg_covar', 100) 

    # Apply Spherical GMM (SGMM) Clustering using the retry mechanism
    sgmm, cluster_labels = fit_gmm_with_retry(
        X,
        n_components=n_clusters, 
        covariance_type='spherical', 
        random_state=random_state_val,
        reg_covar_init=reg_covar_init_val,
        max_reg_covar_val=max_reg_val,
        n_init_val=n_init_val,
        num_processes_for_algo=num_processes_for_algo
    )
    
    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG SGMM main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG SGMM main_clustering] Param for CNI 'aligned_original_labels' - Shape: {original_labels_aligned.shape}")
    
    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, original_labels_aligned, cluster_labels, n_clusters, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value, num_processes_for_algo=num_processes_for_algo)
    num_clusters_after_cni = len(np.unique(final_cluster_labels_from_cni))

    return {
        'Cluster_labeling': final_cluster_labels_from_cni,
        'Best_parameter_dict': parameter_dict_from_elbow
    }


def pre_clustering_SGMM(data, X, n_clusters, random_state, reg_covar=1e-6, n_init=1, num_processes_for_algo=1):
    # Apply Spherical GMM (SGMM) Clustering using the retry mechanism
    # For pre_clustering, usually a fixed n_init is fine, and default max_reg_covar
    sgmm, cluster_labels = fit_gmm_with_retry(
        X,
        n_components=n_clusters, 
        covariance_type='spherical', 
        random_state=random_state,
        reg_covar_init=reg_covar, # Use the reg_covar passed to this function as initial for retry
        n_init_val=n_init,       # Use the n_init passed to this function
        num_processes_for_algo=num_processes_for_algo
    )

    # predict_SGMM = clustering_nomal_identify(data, cluster_labels, n_clusters)
    # num_clusters = len(np.unique(predict_SGMM))  # Counting the number of clusters

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : n_clusters, # n_clusters requested
        'before_labeling' : sgmm
    }