# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.mixture import GaussianMixture
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Elbow_method import Elbow_method
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_GMM_normal(data, X, max_clusters, aligned_original_labels, global_known_normal_samples_pca=None):
    gmm_specific_max_clusters = min(max_clusters, 50) # GMM/SGMM is tested up to 50
    after_elbow = Elbow_method(data, X, 'GMM', gmm_specific_max_clusters)
    n_clusters = after_elbow['optimal_cluster_n']
    parameter_dict = after_elbow['best_parameter_dict']

    n_init_val = parameter_dict.get('n_init', 1)
    reg_covar_init_val = parameter_dict.get('reg_covar', 1e-6)

    gmm, clusters = fit_gmm_with_retry(X, n_clusters, 
                                       random_state=parameter_dict.get('random_state'),
                                       n_init_val=n_init_val,
                                       reg_covar_init=reg_covar_init_val)
    
    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG GMM-normal main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG GMM-normal main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")

    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, n_clusters, global_known_normal_samples_pca=global_known_normal_samples_pca)

    # predict_GMM = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': parameter_dict
    }


def clustering_GMM_full(data, X, max_clusters, aligned_original_labels, global_known_normal_samples_pca=None):
    gmm_specific_max_clusters = min(max_clusters, 50) # GMM/SGMM is tested up to 50
    after_elbow = Elbow_method(data, X, 'GMM', gmm_specific_max_clusters)
    n_clusters = after_elbow['optimal_cluster_n']
    parameter_dict = after_elbow['best_parameter_dict']

    n_init_val = parameter_dict.get('n_init', 1)
    reg_covar_init_val = parameter_dict.get('reg_covar', 1e-6)

    gmm, clusters = fit_gmm_with_retry(X, n_clusters, covariance_type='full', 
                                       random_state=parameter_dict.get('random_state'),
                                       n_init_val=n_init_val,
                                       reg_covar_init=reg_covar_init_val)
    
    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG GMM-full main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG GMM-full main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")
        
    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, n_clusters, global_known_normal_samples_pca=global_known_normal_samples_pca)

    # predict_GMM = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': parameter_dict
    }


def clustering_GMM_tied(data, X, max_clusters, aligned_original_labels, global_known_normal_samples_pca=None):
    gmm_specific_max_clusters = min(max_clusters, 50) # GMM/SGMM is tested up to 50
    after_elbow = Elbow_method(data, X, 'GMM', gmm_specific_max_clusters)
    n_clusters = after_elbow['optimal_cluster_n']
    parameter_dict = after_elbow['best_parameter_dict']

    n_init_val = parameter_dict.get('n_init', 1)
    reg_covar_init_val = parameter_dict.get('reg_covar', 1e-6)

    gmm, clusters = fit_gmm_with_retry(X, n_clusters, covariance_type='tied', 
                                       random_state=parameter_dict.get('random_state'),
                                       n_init_val=n_init_val,
                                       reg_covar_init=reg_covar_init_val)
    
    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG GMM-tied main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG GMM-tied main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")
        
    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, n_clusters, global_known_normal_samples_pca=global_known_normal_samples_pca)

    # predict_GMM = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': parameter_dict
    }


def clustering_GMM_diag(data, X, max_clusters, aligned_original_labels, global_known_normal_samples_pca=None):
    gmm_specific_max_clusters = min(max_clusters, 50) # GMM/SGMM is tested up to 50
    after_elbow = Elbow_method(data, X, 'GMM', gmm_specific_max_clusters)
    n_clusters = after_elbow['optimal_cluster_n']
    parameter_dict = after_elbow['best_parameter_dict']

    n_init_val = parameter_dict.get('n_init', 1)
    reg_covar_init_val = parameter_dict.get('reg_covar', 1e-6)

    gmm, clusters = fit_gmm_with_retry(X, n_clusters, covariance_type='diag', 
                                       random_state=parameter_dict.get('random_state'),
                                       n_init_val=n_init_val,
                                       reg_covar_init=reg_covar_init_val)

    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG GMM-diag main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG GMM-diag main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")
    
    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, n_clusters, global_known_normal_samples_pca=global_known_normal_samples_pca)

    # predict_GMM = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': parameter_dict
    }


def clustering_GMM(data, X, max_clusters, GMM_type, aligned_original_labels, global_known_normal_samples_pca=None):
    if GMM_type == 'normal':
        predict_GMM_dict = clustering_GMM_normal(data, X, max_clusters, aligned_original_labels, global_known_normal_samples_pca=global_known_normal_samples_pca)
    elif GMM_type == 'full':
        predict_GMM_dict = clustering_GMM_full(data, X, max_clusters, aligned_original_labels, global_known_normal_samples_pca=global_known_normal_samples_pca)
    elif GMM_type == 'tied':
        predict_GMM_dict = clustering_GMM_tied(data, X, max_clusters, aligned_original_labels, global_known_normal_samples_pca=global_known_normal_samples_pca)
    elif GMM_type == 'diag':
        predict_GMM_dict = clustering_GMM_diag(data, X, max_clusters, aligned_original_labels, global_known_normal_samples_pca=global_known_normal_samples_pca)
    else:
        print("GMM type Error!! -In Clustering")
        # Consider raising an error or returning a specific structure indicating failure
        return None 

    # predict_GMM = predict_GMM_dict['Cluster_labeling']
    # parameter_dict = predict_GMM_dict['Best_parameter_dict']
    
    # return {
    #     'Cluster_labeling': predict_GMM,
    #     'Best_parameter_dict': parameter_dict
    # }
    return predict_GMM_dict # Return the whole dictionary from the specific GMM type function


# Precept Function for Clustering Count Tuning Loop

def pre_clustering_GMM_normal(data, X, n_clusters, random_state, reg_covar=1e-6, n_init=1):
    gmm, cluster_labels = fit_gmm_with_retry(X, n_clusters, 
                                           random_state=random_state, 
                                           reg_covar_init=reg_covar,
                                           n_init_val=n_init)

    # predict_GMM_normal = clustering_nomal_identify(data, cluster_labels, n_clusters)
    # num_clusters = len(np.unique(predict_GMM_normal))  # Counting the number of clusters

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : n_clusters, # n_clusters requested
        'before_labeling' : gmm
    }


def pre_clustering_GMM_full(data, X, n_clusters, random_state, reg_covar=1e-6, n_init=1):
    gmm, cluster_labels = fit_gmm_with_retry(X, n_clusters, covariance_type='full', 
                                           random_state=random_state, 
                                           reg_covar_init=reg_covar,
                                           n_init_val=n_init)

    # predict_GMM_full = clustering_nomal_identify(data, cluster_labels, n_clusters)
    # num_clusters = len(np.unique(predict_GMM_full))  # Counting the number of clusters

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : n_clusters,
        'before_labeling' : gmm
    }


def pre_clustering_GMM_tied(data, X, n_clusters, random_state, reg_covar=1e-6, n_init=1):
    gmm, cluster_labels = fit_gmm_with_retry(X, n_clusters, covariance_type='tied', 
                                           random_state=random_state, 
                                           reg_covar_init=reg_covar,
                                           n_init_val=n_init)

    # predict_GMM_tied = clustering_nomal_identify(data, cluster_labels, n_clusters)
    # num_clusters = len(np.unique(predict_GMM_tied))  # Counting the number of clusters

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : n_clusters,
        'before_labeling' : gmm
    }


def pre_clustering_GMM_diag(data, X, n_clusters, random_state, reg_covar=1e-6, n_init=1):
    gmm, cluster_labels = fit_gmm_with_retry(X, n_clusters, covariance_type='diag', 
                                           random_state=random_state, 
                                           reg_covar_init=reg_covar,
                                           n_init_val=n_init)

    # predict_GMM_diag = clustering_nomal_identify(data, cluster_labels, n_clusters)
    # num_clusters = len(np.unique(predict_GMM_diag))  # Counting the number of clusters

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : n_clusters,
        'before_labeling' : gmm
    }


def pre_clustering_GMM(data, X, n_clusters, random_state, GMM_type, n_init=30):
    if GMM_type == 'normal':
        clustering_gmm = pre_clustering_GMM_normal(data, X, n_clusters, random_state, reg_covar=1e-6, n_init=n_init)
    elif GMM_type == 'full':
        clustering_gmm = pre_clustering_GMM_full(data, X, n_clusters, random_state, reg_covar=1e-6, n_init=n_init)
    elif GMM_type == 'tied':
        clustering_gmm = pre_clustering_GMM_tied(data, X, n_clusters, random_state, reg_covar=1e-6, n_init=n_init)
    elif GMM_type == 'diag':
        clustering_gmm = pre_clustering_GMM_diag(data, X, n_clusters, random_state, reg_covar=1e-6, n_init=n_init)
    else:
        print("GMM type Error!! -In Clustering")
        return None # Or raise an error
    
    return clustering_gmm


# Functions to automatically update reg_covar to avoid errors
def fit_gmm_with_retry(X, n_components, covariance_type='full', random_state=None, reg_covar_init=1e-6, max_reg_covar_val=100, n_init_val=1):
    reg_covar = reg_covar_init
    while reg_covar <= max_reg_covar_val:
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                random_state=random_state,
                reg_covar=reg_covar,
                n_init=n_init_val
            )
            cluster_labels = gmm.fit_predict(X)
            return gmm, cluster_labels
        except ValueError as e:
            print(f"[Warning] GMM ({covariance_type}, n_init={n_init_val}) failed with reg_covar={reg_covar:.1e}: {e}")
            reg_covar *= 10
    # If loop finishes, all retries failed
    print(f"[ERROR] GMM ({covariance_type}, n_init={n_init_val}) ultimately failed after trying reg_covar up to {reg_covar / 10:.1e} (max was {max_reg_covar_val})")
    raise ValueError(f"GMM ({covariance_type}, n_init={n_init_val}) failed after trying reg_covar up to {max_reg_covar_val}")
