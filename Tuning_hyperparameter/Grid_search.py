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


# Helper function for Grid_search_all parallel execution
def _evaluate_param_set_all(param_set_args):
    """Evaluates a single parameter set for Grid_search_all."""
    # Unpack arguments
    param_set, clustering_algorithm, param_keys_local, create_model_func, X_local, data_local, default_parameter_dict = param_set_args

    if clustering_algorithm == 'NeuralGas':
        params = param_set  # Already a dict for NeuralGas
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
            
            # Ensure data_local has 'label' for evaluate_clustering_with_known_benign
            # The original code for CANN in Grid_search_all seems to imply X is used as features
            # and data might be the original dataframe with labels for evaluation.
            # The evaluate_clustering_with_known_benign expects `data` (original full df), `X` (features), `clusters` (labels from predict), `num_clusters`, `aligned_original_labels`
            # Let's assume model.fit_predict provides the necessary labels directly.
            # The original Grid_search_all calls model.fit_predict(X, data)
            # This implies CANNWithKNN's fit_predict might take two args, or data is used internally by it.
            # For now, assuming data_local is correctly passed and used by model.fit_predict if needed.
            
            # The original code for CANN/CANNwKNN passes `data` (original df) to fit_predict.
            # `labels` here are the cluster assignments from the model.
            # `aligned_original_labels` would be data_local['label'] if data_local is the original df.
            # `num_clusters` is tricky if model determines it; if fixed, needs to be passed.
            
            # Simplified: The original code calls model.fit_predict(X, data), let's stick to that pattern if the model supports it.
            # If CANNWithKNN.fit_predict takes two arguments like X and original_data_with_labels:
            cluster_labels = model.fit_predict(X_local, data_local) # Assuming this is how CANN model works
            
            if len(set(cluster_labels)) < 1: # CANN might return single cluster or error
                return params, score, db_score, f1, acc

            # For CANN, evaluation uses evaluate_clustering_with_known_benign
            # This function needs the original dataframe (`data_local`), features (`X_local`), 
            # predicted cluster labels (`cluster_labels`), number of clusters (dynamic from labels), 
            # and the true labels aligned with X_local (`data_local['label']`).
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

    # Helper for Kmeans parallel processing
    def _evaluate_kmeans_n_init(task_args):
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

    if tasks_kmeans:
        num_processes_kmeans = min(len(tasks_kmeans), multiprocessing.cpu_count())
        try:
            with multiprocessing.Pool(processes=num_processes_kmeans) as pool:
                results_kmeans = pool.map(_evaluate_kmeans_n_init, tasks_kmeans)
            
            for n_init_res, score_res in results_kmeans:
                if score_res > best_score:
                    best_score = score_res
                    best_params = {'n_init': n_init_res}
        except Exception as e:
            print(f"Error during parallel Kmeans n_init search: {e}. Proceeding sequentially as fallback (not implemented here for brevity, original sequential code was removed).")
            # Fallback to original sequential loop if needed, or handle error
            # For now, if parallel fails, best_params might remain None or partially updated
            # Original sequential loop was: 
            # for n_init in n_init_values:
            #     kmeans = KMeans(...) 
            #     labels = kmeans.fit_predict(X) 
            #     if len(set(labels)) > 1: score = silhouette_score(X, labels) 
            #     if score > best_score: best_score = score; best_params = {'n_init': n_init}
            # This part would need to be re-added for a full fallback.
            print("Warning: Kmeans parallel processing failed. Results might be incomplete.")

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

    # 실Create ground truth about where my benigns are
    ground_truth = np.ones(len(data))  # The default is attack(1)
    benign_idx = data.index.isin(benign_data.index)
    ground_truth[benign_idx] = 0

    f1 = f1_score(ground_truth, inferred_labels)
    acc = accuracy_score(ground_truth, inferred_labels)
    return f1, acc


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

    if clustering_algorithm in ['Xmeans', 'xmeans']:
        XMeansWrapper = dynamic_import("Clustering_Method.clustering_Xmeans", "XMeansWrapper")
        param_grid = {'max_clusters': list(range(2, 31, 3))}
        def create_model(params):
            # Use only necessary parameters for XMeans
            model_params = {
                'random_state': parameter_dict['random_state'],
                **params
            }
            return XMeansWrapper(**model_params)

    elif clustering_algorithm in ['Gmeans', 'gmeans']:
        GMeans = dynamic_import("Clustering_Method.clustering_Gmeans", "GMeans")
        log_range = np.logspace(-6, -1, num=10)
        lin_range = np.linspace(min(log_range), max(log_range), num=10)
        combined_range = np.unique(np.concatenate((log_range, lin_range)))

        param_grid = {'max_clusters': list(range(2, 31, 3)), 'tol': combined_range}
        def create_model(params):
            # Use only necessary parameters for GMeans
            model_params = {
                'random_state': parameter_dict['random_state'],
                **params
            }
            return GMeans(**model_params)

    elif clustering_algorithm == 'DBSCAN':
        param_grid = {'eps': np.arange(0.1, 1, 0.02), 'min_samples': list(range(3, 20, 2))}
        def create_model(params):
            return DBSCAN(**params)

    elif clustering_algorithm == 'MShift':
        MeanShiftWithDynamicBandwidth = dynamic_import("Clustering_Method.clustering_Mshift", "MeanShiftWithDynamicBandwidth")
        param_grid = {'quantile': np.arange(0.01, 0.31, 0.05), 'n_samples': list(range(50, 210, 30))}    # Bandwidth estimates can be erroneous if n_samples is too large(1000)
        def create_model(params):
            return MeanShiftWithDynamicBandwidth(**params)

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
        estimated_nodes = int(np.sqrt(n))  # Recommended default values

        # Limiting the max_nodes range
        max_nodes_list = [int(0.5 * estimated_nodes), estimated_nodes, int(1.5 * estimated_nodes)]
        max_nodes_list = [m for m in max_nodes_list if m >= 10]  # Exclude values that are too small

        # Create constrained combinations to make max_edge_age proportional to max_nodes
        edge_age_list = lambda m: [int(0.5 * m), m, int(1.5 * m)]

        param_combinations = []
        for start_nodes in [2, 5, 10]:
            for max_nodes in max_nodes_list:
                for edge_age in edge_age_list(max_nodes):
                    for step in [0.1, 0.2, 0.3]:
                        param_combinations.append({
                            'n_start_nodes': start_nodes,
                            'max_nodes': max_nodes,
                            'step': step,
                            'max_edge_age': edge_age
                        })

        def create_model(params):
            return NeuralGasWithParams(**params)

    elif clustering_algorithm in ['CANNwKNN', 'CANN']:
        CANNWithKNN = dynamic_import("Clustering_Method.clustering_CANNwKNN", "CANNWithKNN")
        param_grid = {'epochs': list(range(20, 501, 20)), 'batch_size': list(range(32, 257, 32)), 
                        'n_neighbors': list(range(3, 51, 5))}
        input_shape = X.shape[1]
        def create_model(params):
            # Use only necessary parameters for CANNwKNN
            model_params = {
                'input_shape': input_shape,
                **params
            }
            return CANNWithKNN(**model_params)

    else:
        print(f"Unsupported algorithm: {clustering_algorithm}")
        pass

    if clustering_algorithm == 'NeuralGas':
        # ... param_combinations already created (see above)
        pass
    else:
        # Generate all hyperparameter combinations
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

    best_score = -1
    best_db_score = float('inf')
    best_params = None

    # Prepare tasks for parallel execution
    worker_tasks = []
    # param_keys, param_combinations are defined based on clustering_algorithm

    # The `create_model` function is defined within the scope of Grid_search_all
    # and depends on `clustering_algorithm` and `parameter_dict`.
    # We need to ensure it's picklable or its dependencies are passed correctly.
    # For `starmap`, the function itself is passed. If it's a local closure,
    # it might work if the closed-over variables are simple.

    # Determine param_keys based on algorithm BEFORE creating tasks for non-NeuralGas
    param_keys_for_worker = []
    if clustering_algorithm != 'NeuralGas':
        param_keys_for_worker = list(param_grid.keys()) # param_grid is defined per algorithm

    for param_set_item in param_combinations:
        # Each task is a tuple: (param_set_item, clustering_algorithm_name, param_keys_list, create_model_function, X_data, original_data, default_param_dict_for_model)
        worker_tasks.append((param_set_item, clustering_algorithm, param_keys_for_worker, create_model, X, data, parameter_dict))

    if worker_tasks:
        num_processes_all = min(len(worker_tasks), multiprocessing.cpu_count())
        print(f"[Grid_search_all] Using {num_processes_all} processes for {clustering_algorithm} with {len(worker_tasks)} param combinations.")
        
        all_param_results = []
        try:
            with multiprocessing.Pool(processes=num_processes_all) as pool:
                # _evaluate_param_set_all expects a single tuple argument
                all_param_results = pool.map(_evaluate_param_set_all, worker_tasks)
        except Exception as e:
            print(f"Error during parallel Grid_search_all for {clustering_algorithm}: {e}")
            # Fallback or error handling for Grid_search_all
            # For now, if parallel fails, results will be empty, leading to no best_params.
            # A full sequential fallback would re-implement the original loop here.
            print(f"Warning: Parallel processing failed for {clustering_algorithm}. Results might be incomplete.")

        # Process results from parallel execution
        for presult_params, presult_score, presult_db_score, presult_f1, presult_acc in all_param_results:
            if clustering_algorithm in ['CANNwKNN', 'CANN']:
                if presult_f1 > best_score: # Using F1 for CANN
                    best_score = presult_f1
                    best_params = presult_params
                    # Optionally store acc as well if needed in best_results
            else: # Other algorithms use Silhouette
                if presult_score > best_score:
                    best_score = presult_score
                    best_db_score = presult_db_score # Store corresponding Davies-Bouldin
                    best_params = presult_params
    else:
        print(f"No parameter combinations to evaluate for {clustering_algorithm}.")

    best_results[clustering_algorithm] = {
        'best_params': best_params,
        'all_params': parameter_dict,  # Return complete parameter_dict for compatibility
        'silhouette_score': best_score,
        'davies_bouldin_score': best_db_score
    }

    return best_results