# Input and output is parameter_dict
# The GridSearchCV model is manually implemented: Evaluate based on silhouette score
'''
Output: Dictionaries by Clustering algorithm
xmeans_result = best_results['Xmeans']  ->  {'best_params': {'max_clusters': 50}, 'all_params': {parameter_dict}, 'silhouette_score': 0.78, 'davies_bouldin_score': 0.42}
best_xmeans_params = best_results['Xmeans']['best_params']  ->  {'max_clusters': 50}
'''

import numpy as np
import importlib
import multiprocessing # Added for parallel processing
from sklearn.model_selection import GridSearchCV
from itertools import product
from sklearn.metrics import make_scorer, silhouette_score, davies_bouldin_score, f1_score, accuracy_score
from sklearn.cluster import KMeans, DBSCAN
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from utils.class_row import nomal_class_data


# Moved helper for Kmeans parallel processing to top level
def _evaluate_kmeans_n_init_worker(task_args):
    X_k, n_clusters_k, parameter_dict_k, n_init_k = task_args
    kmeans = KMeans(
        n_clusters=n_clusters_k,
        random_state=parameter_dict_k['random_state'],
        n_init=n_init_k
    )
    labels_k = kmeans.fit_predict(X_k)
    current_score_k = -1
    if len(set(labels_k)) > 1:
        current_score_k = silhouette_score(X_k, labels_k)
    return n_init_k, current_score_k


# Helper function for Grid_search_all parallel execution
def _evaluate_param_set_all(param_set_args):
    """Evaluates a single parameter set for Grid_search_all."""
    # Unpack arguments
    param_set, clustering_algorithm, param_keys_local, create_model_func, X_local, data_local, default_parameter_dict = param_set_args

    if clustering_algorithm == 'NeuralGas':
        params = param_set
    else:
        params = dict(zip(param_keys_local, param_set))
    
    model = create_model_func(params)
    
    labels = None
    score = -1
    db_score = float('inf')
    f1 = -1
    acc = -1

    try:
        # print(f"[DEBUG] Worker evaluating: {params} for {clustering_algorithm}") # Optional: worker debug
        if clustering_algorithm in ['CANNwKNN', 'CANN']:
            if data_local is None:
                # This case should ideally be prevented by the caller or handled gracefully.
                print(f"[ERROR] Data is None for {clustering_algorithm} but required.")
                return params, score, db_score, f1, acc # Return default/error scores
            
            cluster_labels = model.fit_predict(X_local, data_local)
            
            if len(set(cluster_labels)) < 1:
                return params, score, db_score, f1, acc

            if 'label' in data_local.columns:
                f1, acc = evaluate_clustering_with_known_benign(
                    data_local, X_local, cluster_labels, len(set(cluster_labels)), data_local['label']
                )
                score = f1 # For CANN, primary score is F1
            else:
                print("[ERROR] 'label' column missing in data_local for CANN/CANNwKNN evaluation.")
                # score remains -1, db_score inf, f1 -1, acc -1

        else: # Other algorithms
            labels = model.fit_predict(X_local)
            if len(set(labels)) < 2:
                return params, score, db_score, f1, acc # score = -1, db_score = inf
            score, db_score = evaluate_clustering(X_local, labels)

    except Exception as e:
        print(f"Error evaluating params {params} for {clustering_algorithm}: {e}")
        # Return default/error scores: score = -1, db_score = inf, f1 = -1, acc = -1

    # print(f"[DEBUG] Worker finished: {params}, score: {score}, db: {db_score}, f1: {f1}, acc: {acc}") # Optional
    return params, score, db_score, f1, acc


# Dynamic import functions (using importlib)
def dynamic_import(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def Grid_search_Kmeans(X, n_clusters, parameter_dict=None):
    # Maintain complete parameter_dict for compatibility
    if parameter_dict is None:
        parameter_dict = {
            'random_state': 42,
            'n_init': 30,
            'max_clusters': 1000,
            'tol': 1e-4,
            'eps': 0.5,
            'count_samples': 5,
            'quantile': 0.2,
            'n_samples': 500,
            'n_start_nodes': 2,
            'max_nodes': 50,
            'step': 0.2,
            'max_edge_age': 50,
            'epochs': 300,
            'batch_size': 256,
            'n_neighbors': 5
        }

    n_init_values = list(range(2, 102, 3))
    best_score = -1
    best_params = None

    # Parallelize this loop
    tasks_kmeans = []
    for n_init in n_init_values:
        tasks_kmeans.append((X, n_clusters, parameter_dict, n_init))

    if tasks_kmeans:
        num_processes_kmeans = min(len(tasks_kmeans), multiprocessing.cpu_count())
        try:
            with multiprocessing.Pool(processes=num_processes_kmeans) as pool:
                # Use the top-level worker function
                results_kmeans = pool.map(_evaluate_kmeans_n_init_worker, tasks_kmeans)
            
            for n_init_res, score_res in results_kmeans:
                if score_res > best_score:
                    best_score = score_res
                    best_params = {'n_init': n_init_res}
        except Exception as e:
            print(f"Error during parallel Kmeans n_init search: {e}. Attempting sequential fallback.")
            # Fallback to original sequential loop
            best_score_seq = -1
            best_params_seq = None
            print("Performing sequential Kmeans n_init search as fallback...")
            for task_args_seq in tasks_kmeans: # Iterate through the prepared tasks
                # Call the worker function directly for sequential execution
                n_init_seq, score_seq = _evaluate_kmeans_n_init_worker(task_args_seq)
                if score_seq > best_score_seq:
                    best_score_seq = score_seq
                    best_params_seq = {'n_init': n_init_seq}
            
            # Use sequential results if they are better or if parallel results were incomplete
            if best_params_seq is not None:
                if best_params is None or best_score_seq > best_score: # If parallel failed or sequential is better
                    best_score = best_score_seq
                    best_params = best_params_seq
                    print("Sequential Kmeans fallback provided results.")
                elif best_params is not None: # Parallel had some result, but ensure it's used if it was better
                    print("Parallel Kmeans results will be used (were better or equal to sequential fallback).")
            else:
                print("Warning: Kmeans sequential fallback also failed or produced no valid results.")
            
            if best_params is None:
                 print("Warning: Kmeans n_init search (parallel and sequential) failed to find best_params. Results might be incomplete.")

    # Merge with complete parameter_dict
    best_param_full = parameter_dict.copy()
    if best_params:
        best_param_full.update(best_params)

    return best_param_full


'''
Grid Search functions for clustering algorithms other than Kmeans
'''

def evaluate_clustering(X, labels):
    """Functions to evaluate clustering performance (Silhouette Score & Davies-Bouldin Score)"""
    if len(set(labels)) < 2:
        return -1, float('inf')  # Returning an invalid score if there is only one cluster
    
    sil_score = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    return sil_score, db_score


# Discriminate Functions for CANNwKNN
def evaluate_clustering_with_known_benign(data, X, clusters, num_clusters, aligned_original_labels):
    # 0: benign, 1: attack (criteria: same as clustering_nomal_identify)
    benign_data = nomal_class_data(data).to_numpy() # Assuming that we only know benign data

    inferred_labels = clustering_nomal_identify(X, aligned_original_labels, clusters, num_clusters)

    # Create ground truth about where my benigns are
    ground_truth = np.ones(len(data))  # The default is attack(1)
    benign_idx = data.index.isin(benign_data.index)
    ground_truth[benign_idx] = 0

    f1 = f1_score(ground_truth, inferred_labels)
    acc = accuracy_score(ground_truth, inferred_labels)
    return f1, acc

# THIS IS THE RESTORED Grid_search_all FUNCTION
def Grid_search_all(X, clustering_algorithm, parameter_dict=None, data=None):
    # Maintain complete parameter_dict for compatibility
    if parameter_dict is None:
        parameter_dict = {
            'random_state': 42,
            'n_init': 30,
            'max_clusters': 1000,
            'tol': 1e-4,
            'eps': 0.5,
            'count_samples': 5,
            'quantile': 0.2,
            'n_samples': 500,
            'n_start_nodes': 2,
            'max_nodes': 50,
            'step': 0.2,
            'max_edge_age': 50,
            'epochs': 300,
            'batch_size': 256,
            'n_neighbors': 5
        }

    best_results = {}  # Dictionary for storing the best result of each algorithm

    print(f"\n{clustering_algorithm} Performing clustering...")

    param_grid = {}
    create_model = None # Renamed from create_model_func for original consistency
    param_combinations = [] # For NeuralGas, this will be a list of dicts

    # START: Algorithm-specific param_grid and create_model setup
    if clustering_algorithm in ['Xmeans', 'xmeans']:
        XMeansWrapper = dynamic_import("Clustering_Method.clustering_Xmeans", "XMeansWrapper")
        param_grid = {'max_clusters': list(range(2, 31, 3))}
        def create_model_local(params_local):
            model_params = { 'random_state': parameter_dict['random_state'], **params_local }
            return XMeansWrapper(**model_params)
        create_model = create_model_local

    elif clustering_algorithm in ['Gmeans', 'gmeans']:
        GMeans = dynamic_import("Clustering_Method.clustering_Gmeans", "GMeans")
        log_range = np.logspace(-6, -1, num=10)
        lin_range = np.linspace(min(log_range), max(log_range), num=10)
        combined_range = np.unique(np.concatenate((log_range, lin_range)))
        param_grid = {'max_clusters': list(range(2, 31, 3)), 'tol': combined_range}
        def create_model_local(params_local):
            model_params = { 'random_state': parameter_dict['random_state'], **params_local }
            return GMeans(**model_params)
        create_model = create_model_local

    elif clustering_algorithm == 'DBSCAN':
        param_grid = {'eps': np.arange(0.1, 1, 0.02), 'min_samples': list(range(3, 20, 2))}
        def create_model_local(params_local):
            return DBSCAN(**params_local)
        create_model = create_model_local
    
    elif clustering_algorithm == 'MShift':
        MeanShiftWithDynamicBandwidth = dynamic_import("Clustering_Method.clustering_MShift", "MeanShiftWithDynamicBandwidth")
        param_grid = {'quantile': np.arange(0.01, 0.31, 0.05), 'n_samples': list(range(50, 210, 30))}
        def create_model_local(params_local):
            return MeanShiftWithDynamicBandwidth(**params_local)
        create_model = create_model_local

    elif clustering_algorithm == 'NeuralGas':
        NeuralGasWithParams = dynamic_import("Clustering_Method.clustering_NeuralGas", "NeuralGasWithParams")
        '''
        param_grid = {'n_start_nodes': [2, 5, 7, 10, 15, 20, 35, 50], 'max_nodes': list(range(50, 501, 50)),
                        'step': np.arange(0.05, 0.51, 0.05), 'max_edge_age': list(range(50, 301, 30))}
        def create_model(params):
            return NeuralGasWithParams(**params)
        '''
        # Automatically calculate reasonable ranges based on data counts
        n = len(X)
        estimated_nodes = int(np.sqrt(n))
        max_nodes_list = [int(0.5 * estimated_nodes), estimated_nodes, int(1.5 * estimated_nodes)]
        max_nodes_list = [m for m in max_nodes_list if m >= 10]
        if not max_nodes_list: max_nodes_list = [10]
        edge_age_list_func = lambda m: [int(0.5 * m), m, int(1.5 * m)]
        
        param_combinations_list_of_dicts = [] 
        for start_nodes in [2, 5, 10]:
            for max_nodes_val in max_nodes_list:
                for edge_age in edge_age_list_func(max_nodes_val):
                    for step_val in [0.1, 0.2, 0.3]:
                        param_combinations_list_of_dicts.append({
                            'n_start_nodes': start_nodes,
                            'max_nodes': max_nodes_val,
                            'step': step_val,
                            'max_edge_age': edge_age
                        })
        param_combinations = param_combinations_list_of_dicts # This is a list of dicts for NeuralGas
        def create_model_local(params_local):
            return NeuralGasWithParams(**params_local)
        create_model = create_model_local

    elif clustering_algorithm in ['CANNwKNN', 'CANN']:
        CANNWithKNN = dynamic_import("Clustering_Method.clustering_CANNwKNN", "CANNWithKNN")
        param_grid = {'epochs': list(range(20, 501, 20)), 'batch_size': list(range(32, 257, 32)), 
                        'n_neighbors': list(range(3, 51, 5))}
        input_shape_val = X.shape[1]
        def create_model_local(params_local):
            model_params = { 'input_shape': input_shape_val, **params_local }
            return CANNWithKNN(**model_params)
        create_model = create_model_local
    else:
        print(f"Unsupported algorithm: {clustering_algorithm}")
        return parameter_dict 
    # END: Algorithm-specific param_grid and create_model setup

    # START: Common worker_tasks preparation
    worker_tasks = []
    param_keys_for_worker = [] # Should be defined before use if not NeuralGas

    if clustering_algorithm != 'NeuralGas':
        if not param_grid: # Should not happen if algorithm is supported
            print(f"Error: param_grid not set for {clustering_algorithm}")
            return parameter_dict
        param_keys_for_worker = list(param_grid.keys())
        param_value_tuples = list(product(*param_grid.values()))
        for param_set_tuple in param_value_tuples:
            worker_tasks.append((param_set_tuple, clustering_algorithm, param_keys_for_worker, create_model, X, data, parameter_dict))
    else: # NeuralGas: param_combinations is already list of dicts
        if not param_combinations: # param_combinations should be a list of dicts from above
            print(f"Error: param_combinations (list of dicts) not set for NeuralGas")
            return parameter_dict
        for param_set_dict in param_combinations: 
            # For NeuralGas, _evaluate_param_set_all uses param_set_dict directly. param_keys_for_worker can be empty.
            worker_tasks.append((param_set_dict, clustering_algorithm, [], create_model, X, data, parameter_dict))
    # END: Common worker_tasks preparation

    # START: Common variables for result tracking
    current_best_score = -1
    current_best_db_score = float('inf')
    current_best_f1 = -1 
    current_best_acc = -1 
    current_best_params = None
    all_param_results = [] # Results from parallel or sequential execution
    # END: Common variables for result tracking 

    # START: Common parallel execution with fallback
    if worker_tasks:
        num_processes_all = min(len(worker_tasks), multiprocessing.cpu_count())
        print(f"[Grid_search_all] Using {num_processes_all} processes for {clustering_algorithm} with {len(worker_tasks)} param combinations.")
        
        try:
            # Attempt parallel execution
            with multiprocessing.Pool(processes=num_processes_all) as pool:
                all_param_results = pool.map(_evaluate_param_set_all, worker_tasks)
        except Exception as e:
            # Parallel execution failed, attempt sequential fallback
            print(f"Error during parallel Grid_search_all for {clustering_algorithm}: {e}. Attempting sequential fallback.")
            all_param_results = [] # Clear any partial results from failed parallel attempt
            for task_args_seq in worker_tasks:
                all_param_results.append(_evaluate_param_set_all(task_args_seq))
            
            if not all_param_results or all(res[1] == -1 and (clustering_algorithm not in ['CANNwKNN', 'CANN'] or res[3] == -1) for res in all_param_results):
                print(f"Warning: Sequential fallback for {clustering_algorithm} also failed or produced no valid results.")
            elif all_param_results: 
                print(f"Sequential fallback for {clustering_algorithm} provided results.")
    else:
        print(f"No parameter combinations to evaluate for {clustering_algorithm}.")
    # END: Common parallel execution with fallback

    # START: Common result processing
    for presult_params, presult_score, presult_db_score, presult_f1, presult_acc in all_param_results:
        if clustering_algorithm in ['CANNwKNN', 'CANN']:
            if presult_f1 > current_best_f1:
                current_best_f1 = presult_f1
                current_best_acc = presult_acc
                current_best_params = presult_params
        else: # Other algorithms use Silhouette score (presult_score)
            if presult_score > current_best_score:
                current_best_score = presult_score
                current_best_db_score = presult_db_score
                current_best_params = presult_params
    # END: Common result processing

    # START: Common result storage and return
    if current_best_params is not None:
        best_results_for_current_algo = {
            'best_params': current_best_params,
            'all_params': parameter_dict, # Keep full original dict for reference
        }
        if clustering_algorithm in ['CANNwKNN', 'CANN']:
            best_results_for_current_algo['f1_score'] = current_best_f1
            best_results_for_current_algo['accuracy'] = current_best_acc
            best_results_for_current_algo['silhouette_score'] = -1 # Not primary
            best_results_for_current_algo['davies_bouldin_score'] = float('inf') # Not primary
        else:
            best_results_for_current_algo['silhouette_score'] = current_best_score
            best_results_for_current_algo['davies_bouldin_score'] = current_best_db_score
        best_results[clustering_algorithm] = best_results_for_current_algo # Store in the main dict
    else:
        print(f"Warning: {clustering_algorithm} search failed to find best_params.")
        # Ensure a default entry in best_results for this algorithm to prevent key errors later
        best_results[clustering_algorithm] = {
            'best_params': {},
            'all_params': parameter_dict,
            'silhouette_score': -1,
            'davies_bouldin_score': float('inf')
        }
        if clustering_algorithm in ['CANNwKNN', 'CANN']:
            best_results[clustering_algorithm]['f1_score'] = -1
            best_results[clustering_algorithm]['accuracy'] = -1

    # Final return logic based on Autotune's expectation
    if best_results.get(clustering_algorithm) and best_results[clustering_algorithm].get('best_params'):
        final_param_dict = parameter_dict.copy()
        final_param_dict.update(best_results[clustering_algorithm]['best_params'])
        
        final_param_dict['silhouette_score_from_grid'] = best_results[clustering_algorithm].get('silhouette_score', -1)
        final_param_dict['davies_bouldin_score_from_grid'] = best_results[clustering_algorithm].get('davies_bouldin_score', float('inf'))
        if clustering_algorithm in ['CANNwKNN', 'CANN']:
             final_param_dict['f1_score_from_grid'] = best_results[clustering_algorithm].get('f1_score', -1)
             final_param_dict['accuracy_from_grid'] = best_results[clustering_algorithm].get('accuracy', -1)
        return final_param_dict
    else:
        print(f"Returning original parameter_dict for {clustering_algorithm} as grid search failed or no valid best_params were found.")
        return parameter_dict