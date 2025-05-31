# Clustering Algorithm: Custafson-Kessel (Similarly to Fuzzy Algorithm)
# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Elbow_method import Elbow_method
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from sklearn.metrics import silhouette_score
import multiprocessing # For Pool
import itertools # For product


# Gustafson-Kessel Clustering Implementation
def ck_cluster(X, c, m=2, error=0.01, maxiter=500, epsilon_scale=1e-5, random_state_seed=None): # Added random_state_seed
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
            Maximum number of iterations (default=500).
        epsilon_scale: float
            Scale of epsilon added to the diagonal, relative to mean of diagonal.
        random_state_seed: int or None
            Seed for random number generation.
            
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
    if random_state_seed is not None:
        np.random.seed(random_state_seed) # Set seed for this specific call if provided
        
    n_samples, n_features = X.shape
    u = np.random.dirichlet(np.ones(c), size=n_samples).T  # Random initialization of membership matrix

    um = u ** m

    denom = np.sum(um, axis=1, keepdims=True)
    denom = np.fmax(denom, np.finfo(np.float64).eps)  # Prevent 0
    cntr = np.dot(um, X) / denom

    cov_matrices = np.array([np.eye(n_features) for _ in range(c)])  # Initial covariance matrices
    d = np.zeros((c, n_samples))

    for iteration in range(maxiter):
        cntr_old = cntr.copy() # For convergence check if needed, or u comparison is enough
        u_old = u.copy()

        cntr = np.dot(um, X) / np.fmax(um.sum(axis=1, keepdims=True), np.finfo(np.float64).eps) # Avoid div by zero

        for i in range(c):
            diff = X - cntr[i]
            cov_sum_um_i = um[i].sum()
            if cov_sum_um_i == 0: # Avoid division by zero if a cluster is empty
                # Option 1: Keep previous cov_matrix (or re-initialize)
                # cov_matrices[i] = np.eye(n_features) # Re-initialize
                # Option 2: Skip update for this cluster (might lead to issues if it stays empty)
                # print(f"[WARN ck_cluster] Cluster {i} is empty. Skipping covariance update.")
                # For now, let's try re-initializing, but this needs careful thought
                cov_matrices[i] = np.eye(n_features) * epsilon_scale # Small regularized identity
                continue 
                
            cov = np.dot((um[i][:, np.newaxis] * diff).T, diff) / cov_sum_um_i
            det = np.linalg.det(cov)
            if not np.isfinite(det) or det <= 0:
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
                # print(f"[WARNING ck_cluster] Singular matrix at cluster {i} for inv, using pseudo-inverse.")
                inv_cov = np.linalg.pinv(cov_matrices[i])

            val = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
            val = np.clip(val, 0, None) # Prevent negative numbers before SQRT
            d[i] = np.sqrt(val)

        d_clipped = np.fmax(d, np.finfo(np.float64).eps)
        
        # Check if any column in d_clipped is all zeros (or very small) which means a point is equidistant (or nearly) to all centers with distance 0
        # This can happen if a point is exactly at a center and all cov_matrices are identity for example.
        if np.any(np.all(d_clipped < np.finfo(np.float64).eps, axis=0)):
            # print("[WARN ck_cluster] At least one point has zero distance to all clusters. Handling u_new calculation.")
            # Handle this case to prevent division by zero in u_new calculation for such points
            # For points with all zero distances, assign equal membership or to the first cluster.
            # This is a rare edge case.
            pass # Let u_new calculation proceed, 1.0/np.sum(...) might handle it if d_clipped is non-zero due to fmax

        # u_new calculation - careful with d_power term if m=1 (though m is usually >1)
        if m == 1:
            # Hard clustering: assign to closest cluster
            u_new = np.zeros_like(d_clipped)
            closest_cluster_indices = np.argmin(d_clipped, axis=0)
            u_new[closest_cluster_indices, np.arange(n_samples)] = 1.0
        else:
            dist_power = 2. / (m - 1.)
            # Add small epsilon to d_clipped in denominator to prevent issues if any d_clipped is exactly zero after fmax
            # This is less likely due to fmax but as a safeguard.
            # However, np.sum might become zero if all distances for a point are huge, leading to 1.0/0
            # The previous d_clipped = np.fmax(d, np.finfo(np.float64).eps) should handle most zero issues for individual distances
            temp_dist = d_clipped ** (-dist_power) # (c, n_samples)
            sum_temp_dist = np.sum(temp_dist, axis=0, keepdims=True) # (1, n_samples)
            u_new = temp_dist / np.fmax(sum_temp_dist, np.finfo(np.float64).eps) # Avoid division by zero if sum_temp_dist is zero
            u_new = np.fmax(u_new, np.finfo(np.float64).eps) # Ensure u_new is not zero, which could make um zero

        if np.linalg.norm(u_new - u_old) < error:
            u = u_new # Update u to the latest before breaking
            break
        u = u_new
        um = u ** m

    fpc = np.sum(u ** 2) / n_samples # Standard FPC uses u**2, not u**m for m != 2
    return cntr, u, d, fpc, cov_matrices


def _ck_cluster_worker_args(args_tuple):
    # Unpack arguments and call the original ck_cluster or a slightly modified one if needed
    X_data, c_val, m_val, error_val, maxiter_val, epsilon_val, random_s_seed = args_tuple
    # Call ck_cluster with the random seed for this specific run
    cntr, u, d, fpc, cov_matrices = ck_cluster(X_data, c_val, m_val, error_val, maxiter_val, epsilon_val, random_state_seed=random_s_seed)
    
    if u is not None and u.shape[1] > 0: # Check if u is valid and has samples
        labels = np.argmax(u, axis=0)
        if len(np.unique(labels)) >= 2 and len(labels) >=2 : # Silhouette score needs at least 2 labels and 2 samples
            try:
                score = silhouette_score(X_data, labels)
                return score, cntr, u, d, fpc, cov_matrices, epsilon_val
            except ValueError:
                 # print(f"[WARN _ck_cluster_worker_args] Silhouette score failed for eps={epsilon_val}, seed={random_s_seed}. Not enough unique labels or samples.")
                 return -1.0, cntr, u, d, fpc, cov_matrices, epsilon_val # Return params even on score fail
        else:
            # print(f"[WARN _ck_cluster_worker_args] Not enough unique labels for Silhouette score. Eps={epsilon_val}, Seed={random_s_seed}, UniqueLabels={np.unique(labels)}")
            return -1.0, cntr, u, d, fpc, cov_matrices, epsilon_val # Return params with bad score
    # print(f"[WARN _ck_cluster_worker_args] u is None or empty. Eps={epsilon_val}, Seed={random_s_seed}")
    return -1.0, None, None, None, None, None, epsilon_val


def tune_epsilon_for_ck(X, c, epsilon_candidates=[1e-7, 1e-6, 1e-5, 1e-4], n_init=1, m=2, error=0.01, maxiter=500, num_processes_for_algo=1):
    best_score = -float('inf')
    best_params_tuple = (None, None, None, None, None, None) # cntr, u, d, fpc, cov_matrices, epsilon

    tasks = []
    for epsilon_val_cand in epsilon_candidates:
        for i in range(n_init):
            # Create a unique seed for each init and epsilon combination
            # This ensures that if n_init > 1, each initialization is different
            current_seed = hash((epsilon_val_cand, i)) % (2**32 -1) # Simple hash based seed
            tasks.append((X, c, m, error, maxiter, epsilon_val_cand, current_seed))

    if num_processes_for_algo > 1 and len(tasks) > 1 and X.shape[0] > 10: # Add check for X.shape[0] to avoid overhead for small data
        # print(f"[DEBUG tune_epsilon_for_ck] Using {num_processes_for_algo} processes for {len(tasks)} tasks.")
        try:
            with multiprocessing.Pool(processes=num_processes_for_algo) as pool:
                results = pool.map(_ck_cluster_worker_args, tasks)
        except Exception as e:
            # print(f"[ERROR tune_epsilon_for_ck] Multiprocessing Pool failed: {e}. Falling back to sequential.")
            results = [_ck_cluster_worker_args(task) for task in tasks] # Fallback
    else:
        # print(f"[DEBUG tune_epsilon_for_ck] Running sequentially for {len(tasks)} tasks.")
        results = [_ck_cluster_worker_args(task) for task in tasks]

    valid_results_found = False
    for score, cntr_res, u_res, d_res, fpc_res, cov_m_res, eps_val_res in results:
        if score > -1.0 : # Consider score > -1 as potentially valid (silhouette is [-1, 1])
            valid_results_found = True
            if score > best_score:
                best_score = score
                best_params_tuple = (cntr_res, u_res, d_res, fpc_res, cov_m_res, eps_val_res)
    
    if not valid_results_found:
        # print(f"[WARN tune_epsilon_for_ck] No valid CK configuration found with score > -1.0. Best score was {best_score}.")
        # If all scores were -1, best_params_tuple would still hold params from one of those -1 score runs if any results had non-None components.
        # Ensure we return None if truly no valid model was found (e.g. all components were None from worker)
        if best_params_tuple[0] is None: # Check if center is None as proxy for failed run
            return None, None, None, None, None, None
    
    # print(f"[DEBUG tune_epsilon_for_ck] Selected Epsilon: {best_params_tuple[5]} with Silhouette: {best_score}")
    return best_params_tuple


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


def clustering_CK(data, X, max_clusters, aligned_original_labels, global_known_normal_samples_pca=None, threshold_value=0.3, num_processes_for_algo=1):
    after_elbow = Elbow_method(data, X, 'CK', max_clusters, num_processes_for_algo=num_processes_for_algo)
    n_clusters = after_elbow['optimal_cluster_n']
    parameter_dict = after_elbow['best_parameter_dict']
    n_init_for_ck = parameter_dict.get('n_init', 10) # Use n_init from elbow or default to 10 for CK

    ck_results = tune_epsilon_for_ck(X, c=n_clusters, n_init=n_init_for_ck, num_processes_for_algo=num_processes_for_algo)
    
    if ck_results[0] is None: # Check if tune_epsilon_for_ck failed
        print("[ERROR clustering_CK] tune_epsilon_for_ck returned no valid result. Cannot proceed.")
        # Handle error appropriately, e.g., return a dict indicating failure or raise exception
        return {
            'Cluster_labeling': np.array([]),
            'Best_parameter_dict': parameter_dict,
            'Error': 'CK clustering failed due to no valid result from tuning.'
        }

    cntr, u, d, fpc, cov_matrices, best_epsilon = ck_results
    parameter_dict['epsilon_scale'] = best_epsilon
    parameter_dict['n_init_ck_actual'] = n_init_for_ck

    # Assign clusters based on maximum membership
    cluster_labels = np.argmax(u, axis=0)

    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG CK main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG CK main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")
    
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, cluster_labels, n_clusters, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value, num_processes_for_algo=num_processes_for_algo)

    return {
        'Cluster_labeling': final_cluster_labels_from_cni,
        'Best_parameter_dict': parameter_dict
    }


def pre_clustering_CK(data, X, n_clusters, n_init_for_ck=30, num_processes_for_algo=1):
    ck_results = tune_epsilon_for_ck(X, c=n_clusters, n_init=n_init_for_ck, num_processes_for_algo=num_processes_for_algo)

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
        if cond > 1e10 or np.isnan(cond) or not np.isfinite(cond):
            # print(f"[DEBUG regularize_covariance] High condition number: {cond}. Regularizing.")
            avg_diag = np.mean(np.diag(cov))
            if avg_diag == 0 or not np.isfinite(avg_diag):
                avg_diag = 1.0 # Default if diag is zero or invalid
            epsilon = epsilon_scale * avg_diag
            # Ensure epsilon is not zero if avg_diag was zero and epsilon_scale is small
            epsilon = max(epsilon, np.finfo(float).eps) 
            cov_reg = cov + np.eye(cov.shape[0]) * epsilon
            # print(f"[DEBUG regularize_covariance] Added epsilon: {epsilon}")
            return cov_reg
    except np.linalg.LinAlgError:
        # print("[DEBUG regularize_covariance] LinAlgError during cond check. Regularizing by default.")
        avg_diag = np.mean(np.diag(cov))
        if avg_diag == 0 or not np.isfinite(avg_diag):
            avg_diag = 1.0
        epsilon = epsilon_scale * avg_diag
        epsilon = max(epsilon, np.finfo(float).eps)
        cov_reg = cov + np.eye(cov.shape[0]) * epsilon
        return cov_reg
    return cov


'''
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
'''
