# Clustering Algorithm: Custafson-Kessel (Similarly to Fuzzy Algorithm)
# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Elbow_method import Elbow_method
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from sklearn.metrics import silhouette_score


# Gustafson-Kessel Clustering Implementation
def ck_cluster(X, c, m=2, error=0.01, maxiter=500, epsilon_scale=1e-5): # Fix. Origin; error=0.005, maxiter=1000
    """
    Gustafson-Kessel Clustering Algorithm.
    
    Parameters:
        X: ndarray
            Input data of shape (n_samples, n_features).
        c: int
            Number of clusters.
        m: float
            Fuzziness coefficient (default=2).
        error: float
            Stopping criterion threshold (default=0.005).
        maxiter: int
            Maximum number of iterations (default=1000).
            
    Returns:
        cntr: ndarray
            Cluster centers of shape (c, n_features).
        u: ndarray
            Final membership matrix of shape (c, n_samples).
        d: ndarray
            Distance matrix of shape (c, n_samples).
        fpc: float
            Final fuzzy partition coefficient.
    """
    n_samples, n_features = X.shape
    u = np.random.dirichlet(np.ones(c), size=n_samples).T  # Random initialization of membership matrix

    um = u ** m

    denom = np.sum(um, axis=1, keepdims=True)
    denom = np.fmax(denom, np.finfo(np.float64).eps)  # Prevent 0
    cntr = np.dot(um, X) / denom

    cov_matrices = np.array([np.eye(n_features) for _ in range(c)])  # Initial covariance matrices
    d = np.zeros((c, n_samples))

    for iteration in range(maxiter):
        # Calculate cluster centers
        cntr = np.dot(um, X) / um.sum(axis=1, keepdims=True)

        # Update covariance matrices
        for i in range(c):
            diff = X - cntr[i]
            cov = np.dot((um[i][:, np.newaxis] * diff).T, diff) / um[i].sum()

            # Normalize by determinant
            det = np.linalg.det(cov)
            if not np.isfinite(det) or det <= 0:    # Exception handling
                det = np.finfo(float).eps
            cov /= det ** (1 / n_features)

            # Regularize covariance
            cov = regularize_covariance(cov, epsilon_scale)

            cov_matrices[i] = cov
        '''
        # Checking matrix dimensions
        print("um[i].shape:", um[i].shape)
        print("diff.shape:", diff.shape)
        print("cov_matrices[i].shape:", cov_matrices[i].shape)
        '''
        
        # Calculate distances and update membership
        for i in range(c):
            diff = X - cntr[i]
            
            try:
                inv_cov = np.linalg.inv(cov_matrices[i])
            except np.linalg.LinAlgError:
                print(f"[WARNING] Singular matrix at cluster {i}, using pseudo-inverse.")
                inv_cov = np.linalg.pinv(cov_matrices[i])

            val = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
            val = np.clip(val, 0, None) # Prevent negative numbers before SQRT
            d[i] = np.sqrt(val)

        d = np.fmax(d, np.finfo(np.float64).eps)  # Avoid division by zero
        
        ratio = d / d[:, np.newaxis]
        ratio = np.clip(ratio, np.finfo(np.float64).eps, 1e10)  # Preventing very small to too large numbers
        u_new = 1.0 / np.sum(ratio ** (2 / (m - 1)), axis=0)

        # Check for convergence
        if np.linalg.norm(u_new - u) < error:
            break
        u = u_new

    fpc = np.sum(u ** m) / n_samples
    return cntr, u, d, fpc, cov_matrices


def ck_predict(X_new, cntr, cov_matrices, m=2):
    """
    Predict membership for new data points in Gustafson-Kessel clustering.

    Parameters:
        X_new: ndarray
            New data of shape (n_samples, n_features).
        cntr: ndarray
            Cluster centers of shape (c, n_features).
        cov_matrices: ndarray
            Covariance matrices of shape (c, n_features, n_features).
        m: float
            Fuzziness coefficient.

    Returns:
        membership: ndarray
            Membership matrix of shape (c, n_samples).
    """
    c = cntr.shape[0]
    n_samples = X_new.shape[0]
    d = np.zeros((c, n_samples))

    for i in range(c):
        diff = X_new - cntr[i]
        inv_cov = np.linalg.inv(cov_matrices[i])
        d[i] = np.sqrt(np.sum(np.dot(diff, inv_cov) * diff, axis=1))

    d = np.fmax(d, np.finfo(np.float64).eps)  # Avoid divide-by-zero
    u = 1.0 / np.sum((d / d[:, np.newaxis]) ** (2 / (m - 1)), axis=0)

    return u


def clustering_CK(data, X, max_clusters, aligned_original_labels, global_known_normal_samples_pca=None):
    after_elbow = Elbow_method(data, X, 'CK', max_clusters)
    n_clusters = after_elbow['optimal_cluster_n']
    parameter_dict = after_elbow['best_parameter_dict'] # This dict might have 'n_init' from Elbow's base params

    # Perform Gustafson-Kessel Clustering; Performing with auto-tuned epsilon included
    # Pass the specific n_init for CK here
    ck_results = tune_epsilon_for_ck(X, c=n_clusters, n_init=n_init_for_ck)
    
    if ck_results[0] is None: # Check if tune_epsilon_for_ck failed
        print("[ERROR clustering_CK] tune_epsilon_for_ck returned no valid result. Cannot proceed.")
        # Handle error appropriately, e.g., return a dict indicating failure or raise exception
        return {
            'Cluster_labeling': np.array([]), # Empty or error indicator
            'Best_parameter_dict': parameter_dict, # Might still be useful
            'Error': 'CK clustering failed due to no valid result from tuning.'
        }

    cntr, u, d, fpc, cov_matrices, best_epsilon = ck_results
    parameter_dict['epsilon_scale'] = best_epsilon  # Save selected values
    parameter_dict['n_init_ck_actual'] = n_init_for_ck # Save actual n_init used for CK

    # Assign clusters based on maximum membership
    cluster_labels = np.argmax(u, axis=0)

    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG CK main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG CK main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")
    
    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, cluster_labels, n_clusters, global_known_normal_samples_pca=global_known_normal_samples_pca)

    # predict_CK = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': parameter_dict
    }


def pre_clustering_CK(data, X, n_clusters, n_init_for_ck=30):
    # Tune epsilon_scale for better stability, now also with n_init
    ck_results = tune_epsilon_for_ck(X, c=n_clusters, n_init=n_init_for_ck)

    if ck_results[0] is None: # Check if tune_epsilon_for_ck failed
        print(f"[ERROR pre_clustering_CK] tune_epsilon_for_ck returned no valid result for n_clusters={n_clusters}. Cannot proceed.")
        # For pre_clustering, it must return the expected dict structure or raise error
        # Returning a structure that indicates failure but matches expected keys where possible
        return {
            'model_labels': np.array([]), 
            'n_clusters': n_clusters,
            'before_labeling': None, # No model
            'Error': 'CK pre_clustering failed.' 
        }
        
    cntr, u, d, fpc, cov_matrices, best_epsilon = ck_results

    # Assign clusters based on maximum membership
    cluster_labels = np.argmax(u, axis=0)

    # predict_CK = clustering_nomal_identify(data, cluster_labels, n_clusters)
    # num_clusters = len(np.unique(predict_CK))  # Counting the number of clusters

    # Wrapping to write like a model
    ck_model = CKFakeModel(cntr, cov_matrices, fpc)

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : n_clusters, # n_clusters requested
        'before_labeling' : ck_model
    }

# Make it look like a model for Elbow/tuning
class CKFakeModel:
    def __init__(self, cntr, cov_matrices, fpc):
        self.cntr = cntr
        self.cov_matrices = cov_matrices
        self.fpc = fpc
        self.inertia_ = 1 - fpc  # Just for Elbow_method compatibility

    def predict(self, X_new):
        u = ck_predict(X_new, self.cntr, self.cov_matrices)
        return np.argmax(u, axis=0)

    def fit(self, X):
        pass  # Already fitted


# Functions for stabilizing CK covariance
def regularize_covariance(cov, epsilon_scale=1e-5):
    """
    Regularize covariance matrix if it is near-singular.
    
    Parameters:
        cov: ndarray
            Covariance matrix.
        epsilon_scale: float
            Scale of epsilon added to the diagonal, relative to mean of diagonal.

    Returns:
        cov_reg: ndarray
            Regularized covariance matrix.
    """
    try:
        # Check determinant for near-singularity
        cond = np.linalg.cond(cov)
        if cond > 1e10 or np.isnan(cond):
            avg_diag = np.mean(np.diag(cov))
            if avg_diag == 0 or np.isnan(avg_diag):
                avg_diag = 1.0
            epsilon = epsilon_scale * avg_diag
            cov += np.eye(cov.shape[0]) * epsilon
    except np.linalg.LinAlgError:
        # If condition number fails
        cov += np.eye(cov.shape[0]) * epsilon_scale
    return cov

def tune_epsilon_for_ck(X, c, epsilon_candidates=[1e-7, 1e-6, 1e-5, 1e-4], n_init=1, m=2, error=0.01, maxiter=500):
    best_overall_score = -np.inf # Initialize with a very small number
    best_overall_result = None
    best_overall_epsilon = None

    for eps in epsilon_candidates:
        current_epsilon_best_score = -np.inf
        current_epsilon_best_result_for_init = None
        
        for init_iter in range(n_init):
            # print(f"[DEBUG CK TuneEpsilon] Epsilon: {eps}, Init: {init_iter + 1}/{n_init}") # Optional Debug
            try:
                # For ck_cluster to have different random initializations, 
                # we might need to ensure its internal np.random.dirichlet behaves differently.
                # Explicitly seeding here for each init_iter could be an option if ck_cluster itself doesn't vary enough.
                # However, np.random.dirichlet itself should produce different results if called multiple times without a fixed global seed.
                
                cntr_iter, u_iter, d_iter, fpc_iter, cov_matrices_iter = ck_cluster(X, c=c, m=m, error=error, maxiter=maxiter, epsilon_scale=eps)
                
                # Ensure u_iter is valid for silhouette_score
                if u_iter is None or u_iter.shape[0] != c or u_iter.shape[1] != X.shape[0]:
                    print(f"[Warning CK TuneEpsilon] Invalid membership matrix u_iter for eps {eps}, init {init_iter+1}. Skipping.")
                    continue

                labels_iter = np.argmax(u_iter, axis=0)
                
                if len(np.unique(labels_iter)) < 2: # Silhouette score requires at least 2 labels
                    # print(f"[Warning CK TuneEpsilon] Not enough unique labels ({len(np.unique(labels_iter))}) for eps {eps}, init {init_iter+1} to calculate silhouette. Using FPC or skipping.")
                    # Fallback to FPC or skip this iteration
                    # For now, let's use FPC if silhouette isn't possible. Or simply skip.
                    # Using FPC: score_iter = fpc_iter 
                    # Skipping:
                    continue

                score_iter = silhouette_score(X, labels_iter)

                if score_iter > current_epsilon_best_score:
                    current_epsilon_best_score = score_iter
                    current_epsilon_best_result_for_init = (cntr_iter, u_iter, d_iter, fpc_iter, cov_matrices_iter)

            except np.linalg.LinAlgError as e:
                print(f"[Warning CK TuneEpsilon] LinAlgError for eps {eps}, init {init_iter+1}: {e}")
                continue # Skip this iteration
            except ValueError as e: # Might be raised by silhouette_score if labels are problematic
                print(f"[Warning CK TuneEpsilon] ValueError for eps {eps}, init {init_iter+1}: {e}")
                continue # Skip this iteration
            except Exception as e:
                print(f"[Warning CK TuneEpsilon] Unexpected error for eps {eps}, init {init_iter+1}: {e}")
                continue

        # After all n_init for current epsilon
        if current_epsilon_best_result_for_init is not None and current_epsilon_best_score > best_overall_score:
            best_overall_score = current_epsilon_best_score
            best_overall_result = current_epsilon_best_result_for_init
            best_overall_epsilon = eps

    if best_overall_result is None:
        # Fallback if no valid result was found after all eps and inits (e.g., all singular matrices or <2 labels)
        print("[ERROR CK TuneEpsilon] No valid clustering result found after all trials. Returning None.")
        # Or raise an error, or return a dummy/default result
        # For now, let's return Nones, which will likely cause issues upstream, highlighting the problem.
        return None, None, None, None, None, None


    # Unpack the best overall result
    cntr, u, d, fpc, cov_matrices = best_overall_result
    return cntr, u, d, fpc, cov_matrices, best_overall_epsilon
