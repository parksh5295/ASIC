# with Elbow method
# input 'data' is X or X_reduced
# 'clustering' is 'clustering_algorithm'.fit_predict(data)
# output(optimal_k): 'Only' Optimal number of cluster by data

# Some Clustering Algorihtm; Kmeans, Kmedians, GMM, SGMM, FCM, CK requires additional work to tune the number of clusters.

import numpy as np
from Clustering_Method.common_clustering import get_clustering_function
import multiprocessing # Added for parallel processing
import os # For os.cpu_count()


def Elbow_choose_clustering_algorithm(data, X, clustering_algorithm, n_clusters, parameter_dict, GMM_type="normal", num_processes_for_algo=1):   # X: Encoding and embedding, post-PCA, post-delivery
    pre_clustering_func = get_clustering_function(clustering_algorithm)

    # Parameters are expected to be in parameter_dict passed from Elbow_method
    if clustering_algorithm == 'Kmeans':
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            # 'n_init': parameter_dict['n_init'],
            'n_init': 10, # Use a smaller, fixed n_init for faster k selection in Elbow for K-Means
            'num_processes_for_algo': num_processes_for_algo
        }
        clustering = pre_clustering_func(data, X, n_clusters, **algorithm_params)
    elif clustering_algorithm == 'GMM': # General GMM, distinct from SGMM
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'GMM_type': GMM_type, # GMM_type is specific to pre_clustering_GMM
            # 'reg_covar': parameter_dict['reg_covar'],
            'n_init': parameter_dict.get('n_init', 1), # Ensure n_init is passed for general GMM too
            'num_processes_for_algo': num_processes_for_algo
        }
        clustering = pre_clustering_func(data, X, n_clusters, **algorithm_params)
    elif clustering_algorithm == 'SGMM': # Spherical GMM
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'reg_covar': parameter_dict['reg_covar'],
            'n_init': parameter_dict.get('n_init', 1), # Added n_init from parameter_dict
            'num_processes_for_algo': num_processes_for_algo
        }
        clustering = pre_clustering_func(data, X, n_clusters, **algorithm_params)
    elif clustering_algorithm in ['FCM', 'CK']:
        if clustering_algorithm == 'CK':
            # For CK, ensure n_init from the parameter_dict (which is 30 from Elbow_method's base_parameter_dict)
            # is passed as n_init_for_ck to pre_clustering_CK
            algorithm_params_ck = {
                'n_init_for_ck': parameter_dict.get('n_init', 30), # Default to 30 if not in dict somehow
                'num_processes_for_algo': num_processes_for_algo
            }
            clustering = pre_clustering_func(data, X, n_clusters, **algorithm_params_ck)
        else: # FCM or any other in this list that doesn't take n_init explicitly here, but pre_clustering_FCM should take num_processes_for_algo
            algorithm_params_fcm = {
                'num_processes_for_algo': num_processes_for_algo
            }
            clustering = pre_clustering_func(data, X, n_clusters, **algorithm_params_fcm)
    elif clustering_algorithm == 'Gmeans':
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'max_clusters': parameter_dict['max_clusters'], # GMeans uses max_clusters from dict
            'tol': parameter_dict['tol'],
            'n_init': parameter_dict.get('n_init', 30), # Added n_init for Gmeans
            'num_processes_for_algo': num_processes_for_algo
        }
        # n_clusters (k from loop) is not passed directly to pre_clustering_Gmeans
        clustering = pre_clustering_func(data, X, **algorithm_params)
    elif clustering_algorithm == 'Xmeans':
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'max_clusters': parameter_dict['max_clusters'], # XMeans uses max_clusters from dict
            'n_init': parameter_dict.get('n_init', 30), # Added n_init for Xmeans
            'num_processes_for_algo': num_processes_for_algo
        }
        # n_clusters (k from loop) is not passed directly to pre_clustering_Xmeans
        clustering = pre_clustering_func(data, X, **algorithm_params)
    elif clustering_algorithm == 'DBSCAN':
        algorithm_params = {
            # DBSCAN does not use random_state in its sklearn implementation
            'eps': parameter_dict['eps'],
            'count_samples': parameter_dict['count_samples'], # maps to min_samples
            'num_processes_for_algo': num_processes_for_algo
        }
        # n_clusters (k from loop) is not passed to pre_clustering_DBSCAN
        clustering = pre_clustering_func(data, X, **algorithm_params)
    elif clustering_algorithm == 'MShift':
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'quantile': parameter_dict['quantile'],
            'n_samples': parameter_dict['n_samples'],
            'num_processes_for_algo': num_processes_for_algo
        }
        # n_clusters (k from loop) is not passed to pre_clustering_MShift
        clustering = pre_clustering_func(data, X, **algorithm_params)
    elif clustering_algorithm == 'NeuralGas':
        algorithm_params = {
            # NeuralGas pre_clustering doesn't take random_state in its current signature
            'n_start_nodes': parameter_dict['n_start_nodes'],
            'max_nodes': parameter_dict['max_nodes'],
            'step': parameter_dict['step'],
            'max_edge_age': parameter_dict['max_edge_age'],
            'num_processes_for_algo': num_processes_for_algo
        }
        # n_clusters (k from loop) is not passed to pre_clustering_NeuralGas
        clustering = pre_clustering_func(data, X, **algorithm_params)
    else: # KMedians and any other algorithm that takes n_clusters and random_state by default
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'num_processes_for_algo': num_processes_for_algo
        }
        clustering = pre_clustering_func(data, X, n_clusters, **algorithm_params)

    return clustering


def Elbow_method(data, X, clustering_algorithm, max_clusters, parameter_dict=None, num_processes_for_algo=None):
    # Maintain complete parameter_dict for compatibility and to ensure all necessary params are available
    base_parameter_dict = {
        'random_state': 42,
        'n_init': 30, # For KMeans primarily
        'max_clusters': 1000, # For GMeans, XMeans primarily
        'tol': 1e-4, # For GMeans primarily
        'eps': 0.5, # For DBSCAN primarily
        'count_samples': 5, # For DBSCAN primarily (as min_samples)
        'quantile': 0.2, # For MShift primarily
        'n_samples': 500, # For MShift primarily
        'n_start_nodes': 2, # For NeuralGas primarily
        'max_nodes': 50, # For NeuralGas primarily
        'step': 0.2, # For NeuralGas primarily
        'max_edge_age': 50, # For NeuralGas primarily
        'epochs': 300, # For CANNwKNN (if used here, but usually tuned in Grid_search_all)
        'batch_size': 256, # For CANNwKNN
        'n_neighbors': 5, # For CANNwKNN
        'reg_covar': 1e-6 # For GMM and SGMM
    }
    if parameter_dict is not None:
        base_parameter_dict.update(parameter_dict) # Update with any user-provided params
    
    current_parameter_dict = base_parameter_dict.copy()

    wcss_or_bic = []  # Store WCSS (inertia) or BIC by number of clusters
    cluster_range = range(2, max_clusters + 1) # Start from 2 clusters for GMM/SGMM BIC calculation and most algos
    if not cluster_range:
        # Handle edge case where max_clusters < 2
        print("Warning: max_clusters is less than 2. Elbow method requires at least 2 clusters. Returning default k=2.")
        return {
            'optimal_cluster_n': 2,
            'best_parameter_dict': current_parameter_dict
        }

    # Prepare tasks for parallel execution
    tasks = []
    for k_task in cluster_range:
        # Pass num_processes_for_algo to _calculate_score_for_k via args_tuple
        tasks.append((k_task, data, X, clustering_algorithm, current_parameter_dict, num_processes_for_algo))

    # This will store (k, score) tuples, possibly out of order from parallel execution
    k_score_pairs = [] 

    if tasks:
        # Determine number of processes for the Pool
        if num_processes_for_algo == 0: # Use all available CPUs
            pool_processes = os.cpu_count()
            if pool_processes is None: # Fallback if os.cpu_count() returns None
                 pool_processes = 1 
        elif num_processes_for_algo is not None and num_processes_for_algo > 0: # User specified positive number
            pool_processes = num_processes_for_algo
        else: # Default: None or invalid value, use half CPUs or multiprocessing's default, ensure at least 1
            cpu_cores = os.cpu_count()
            if cpu_cores:
                pool_processes = max(1, cpu_cores // 2)
            else: # Fallback if os.cpu_count() returns None
                pool_processes = max(1, multiprocessing.cpu_count() // 2 if multiprocessing.cpu_count() else 1)
        
        # Ensure pool_processes does not exceed the number of tasks
        pool_processes = min(pool_processes, len(tasks))
        if pool_processes == 0 and len(tasks) > 0: pool_processes = 1 # Ensure at least 1 process if tasks exist

        print(f"[Elbow_method] Using {pool_processes} processes for {len(tasks)} k-values (num_processes_for_algo={num_processes_for_algo}).")
        try:
            with multiprocessing.Pool(processes=pool_processes) as pool:
                # pool.map will preserve order if tasks are ordered, 
                # but _calculate_score_for_k returns (k, score) so we can sort later if needed.
                # Using map as _calculate_score_for_k takes a single tuple argument.
                k_score_pairs = pool.map(_calculate_score_for_k, tasks)
        except Exception as e:
            print(f"Error during parallel Elbow method processing: {e}. Falling back to sequential.")
            # Fallback to sequential execution
            k_score_pairs = [] # Clear partially filled results from try block
            for k_seq in cluster_range:
                # Call helper directly, passing num_processes_for_algo
                _, score_seq = _calculate_score_for_k((k_seq, data, X, clustering_algorithm, current_parameter_dict, num_processes_for_algo)) 
                k_score_pairs.append((k_seq, score_seq))
    else:
        print("[Elbow_method] No tasks to run for Elbow method (max_clusters might be < 2).")
        # This case is also handled by the `if not cluster_range:` block earlier, 
        # but good to have a log if tasks list is empty for other reasons.

    # Sort results by k to ensure wcss_or_bic is in the correct order for diff calculations
    k_score_pairs.sort(key=lambda x: x[0])
    
    # Populate wcss_or_bic from sorted results
    wcss_or_bic = [score for k_val, score in k_score_pairs]
    
    if len(wcss_or_bic) < 2: # Need at least 2 points to calculate differences
        print("Warning: Not enough valid scores to determine elbow. Returning default k=2 or max_clusters if 1.")
        optimal_k = 2 if max_clusters >= 2 else max_clusters if max_clusters ==1 else 1 # Ensure k is at least 1
        if max_clusters == 0 : optimal_k =1 # avoid k=0
        return {
            'optimal_cluster_n': optimal_k,
            'best_parameter_dict': current_parameter_dict
        }
    
    # Rate of change of slope; For GMM/SGMM, lower BIC is better → so reverse slope logic by finding max decrease (min -diff)
    # For WCSS (inertia), lower is better, so we want to find where the decrease starts to slow down (max second derivative)
    diff_scores = np.diff(wcss_or_bic)
    if len(diff_scores) < 1:
        print("Warning: Not enough score differences to determine elbow. Returning default k=2 or max_clusters if 1.")
        optimal_k = 2 if max_clusters >= 2 else max_clusters if max_clusters ==1 else 1
        if max_clusters == 0 : optimal_k =1
        return {
            'optimal_cluster_n': optimal_k,
            'best_parameter_dict': current_parameter_dict
        }

    second_diff = np.diff(diff_scores) 
    if len(second_diff) == 0:
        # If only two k values were tested (e.g., k=2, k=3), second_diff will be empty.
        # Default to the k with the better score or a predefined k.
        # For GMM/SGMM (BIC, lower is better), choose k with min score.
        # For WCSS (inertia, lower is better), choose k with min score.
        optimal_k_index = np.argmin(wcss_or_bic)
        optimal_k = cluster_range[optimal_k_index] 
        print("Warning: Only two k values tested or not enough points for second derivative. Optimal k chosen by min score.")
    else:
        if clustering_algorithm in ['GMM', 'SGMM']:
            # For BIC, we look for the point where the BIC score starts to increase after decreasing, or decreases less steeply.
            # The elbow point is where the second derivative of BIC (w.r.t k) is maximized (most positive change in slope).
            # Since we want lower BIC, np.argmin(wcss_or_bic) would give the best k if no elbow.
            # Kneedle algorithm might be more robust here. For now, using second derivative approach.
            # We want the point before a significant increase, or where decrease significantly slows.
            # argmax(second_diff) means the largest increase in slope (from more negative to less negative, or from negative to positive slope of BIC diffs)
            optimal_k_index = np.argmax(second_diff) + 1 # +1 because second_diff is shorter by 1 than diff_scores
        else:
            # For WCSS (inertia), we look for the point where the rate of decrease slows down.
            # This corresponds to the maximum of the second derivative (largest positive value).
            optimal_k_index = np.argmax(second_diff) + 1 # +1 as above
        
        optimal_k = cluster_range[optimal_k_index] # Convert index back to k value

    return {
        'optimal_cluster_n': optimal_k,
        'best_parameter_dict': current_parameter_dict # Return the potentially updated dict
    }

# Helper function for Elbow_method parallel execution
def _calculate_score_for_k(args_tuple):
    """Calculates WCSS/BIC score for a single k value."""
    # Unpack num_processes_for_algo from args_tuple
    k, data_local, X_local, clustering_algorithm_local, current_parameter_dict_local, num_processes_for_algo_local = args_tuple
    score_val = np.inf # Default to a bad score

    try:
        # print(f"[DEBUG Elbow Worker] Processing k={k} for {clustering_algorithm_local} with num_processes_for_algo={num_processes_for_algo_local}") # Optional worker debug
        # Pass num_processes_for_algo_local to Elbow_choose_clustering_algorithm
        clustering_result = Elbow_choose_clustering_algorithm(data_local, X_local, clustering_algorithm_local, k, current_parameter_dict_local, num_processes_for_algo=num_processes_for_algo_local)
        
        if clustering_result is None or 'before_labeling' not in clustering_result or clustering_result['before_labeling'] is None:
            print(f"Warning: Clustering failed or returned no model for k={k} with {clustering_algorithm_local}. Using inf score.")
            return k, np.inf # score_val is already np.inf or explicitly set
            
        clustering_model = clustering_result['before_labeling']
        
        # The original code had a try-except for fit here, but pre_clustering usually handles fit/fit_predict.
        # We rely on inertia_ or bic() being available after Elbow_choose_clustering_algorithm.

        temp_score = None
        if clustering_algorithm_local in ['GMM', 'SGMM']:
            if hasattr(clustering_model, 'bic') and callable(getattr(clustering_model, 'bic')):
                try:
                    temp_score = clustering_model.bic(X_local)
                except Exception as e_bic:
                    print(f"Warning: Error calling bic() for {clustering_algorithm_local} model at k={k}: {e_bic}")
            else:
                print(f"Warning: bic() method not available or not callable for {clustering_algorithm_local} model at k={k}.")
        elif clustering_algorithm_local == 'CK':
            if hasattr(clustering_model, 'silhouette_score_'):
                silhouette_val = clustering_model.silhouette_score_
                if silhouette_val is not None and np.isfinite(silhouette_val):
                    # Silhouette Score is in [-1, 1], higher is better.
                    # To make it compatible with Elbow (lower is better & positive slope for elbow point):
                    # Transform to [0, 2] where 0 is best.
                    temp_score = 1.0 - silhouette_val 
                else:
                    print(f"Warning: Silhouette score is None or invalid for CK model at k={k} (value: {silhouette_val}). Using inf score for this k.")
            else:
                print(f"Warning: silhouette_score_ attribute not available for CK model at k={k}. Using inf score for this k.")
        else: # K-Means, etc.
            if hasattr(clustering_model, 'inertia_'):
                inertia_val = clustering_model.inertia_
                if inertia_val is not None and np.isfinite(inertia_val):
                    temp_score = inertia_val
                else:
                    print(f"Warning: Inertia is None or invalid for {clustering_algorithm_local} model at k={k} (value: {inertia_val}). Using inf score for this k.")
            else:
                print(f"Warning: inertia_ attribute not available for {clustering_algorithm_local} model at k={k}. Using inf score for this k.")
        
        if temp_score is not None and np.isfinite(temp_score):
            score_val = temp_score
        else:
            score_val = np.inf # Ensure bad score if temp_score ended up None or non-finite
            
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError for k={k} with {clustering_algorithm_local}: {e}. Using inf score.")
    except ValueError as e:
        print(f"ValueError for k={k} with {clustering_algorithm_local}: {e}. Using inf score.")
    except Exception as e:
        # Catch any other unexpected errors from a specific k iteration
        print(f"Unexpected error for k={k} with {clustering_algorithm_local}: {e}. Using inf score.")
        
    # print(f"[DEBUG Elbow Worker] Finished k={k}, score={score_val}") # Optional
    return k, score_val