# A machine to test clustering algorithm for labeling data and determine the performance of each clustering algorithm.

import argparse
import numpy as np
import pandas as pd # Ensure pandas is imported as pd for X.to_numpy()
import time
import math # Added for math.ceil
from sklearn.preprocessing import MinMaxScaler
from Dataset_Choose_Rule.choose_amount_dataset import file_path_line_nonnumber, file_cut
from definition.Anomal_Judgment import anomal_judgment_nonlabel, anomal_judgment_label
from utils.time_transfer import time_scalar_transfer
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
from Modules.PCA import pca_func
from Modules.Clustering_Algorithm_Autotune import choose_clustering_algorithm
from Modules.Clustering_Algorithm_Nonautotune import choose_clustering_algorithm_Non_optimization
from utils.cluster_adjust_mapping import cluster_mapping
from Clustering_Method.clustering_score import evaluate_clustering, evaluate_clustering_wos
from Dataset_Choose_Rule.save_csv import csv_compare_clustering, csv_compare_matrix_clustering
from Dataset_Choose_Rule.time_save import time_save_csv_VL
from utils.minmaxscaler import apply_minmax_scaling_and_save_scalers


def main():
    # argparser
    # Create an instance that can receive argument values
    parser = argparse.ArgumentParser(description='Argparser')

    # Set the argument values to be input (default value can be set)
    parser.add_argument('--file_type', type=str, default="MiraiBotnet")   # data file type
    parser.add_argument('--file_number', type=int, default=1)   # Detach files
    parser.add_argument('--train_test', type=int, default=0)    # train = 0, test = 1
    parser.add_argument('--heterogeneous', type=str, default="Interval_inverse")   # Heterogeneous(Embedding) Methods
    parser.add_argument('--clustering', type=str, default="kmeans")   # Clustering Methods
    parser.add_argument('--eval_clustering_silhouette', type=str, default="n")
    parser.add_argument('--association', type=str, default="apriori")   # Association Rule

    # Save the above in args
    args = parser.parse_args()

    # Output the value of the input arguments
    file_type = args.file_type
    file_number = args.file_number
    train_tset = args.train_test
    heterogeneous_method = args.heterogeneous
    clustering_algorithm = args.clustering
    eval_clustering_silhouette = args.eval_clustering_silhouette
    Association_mathod = args.association

    total_start_time = time.time()  # Start All Time
    timing_info = {}  # For step-by-step time recording


    # 0. Create Global Reference Normal Samples PCA (for CNI function)
    # This is done once based on the full dataset if possible.
    start_global_ref = time.time()
    print("\nStep 0: Creating Global Reference Normal Samples PCA...")
    global_known_normal_samples_pca_for_cni = None
    try:
        # --- MOVED file_path definition here for Step 0 ---
        file_path_for_global_ref, _ = file_path_line_nonnumber(file_type, file_number) # Use a distinct variable if file_number is also used/modified later
        # If file_number from args is only for main processing, using it here might be okay.
        # For clarity, used file_path_for_global_ref.

        # Load FULL data for reference normal selection, regardless of main cut_type
        # This assumes file_path_line_nonnumber gives a path that can be read fully
        # For CICIDS2017, select_csv_file() might need adjustment or a specific full file path
        # For simplicity, we assume file_path is suitable for a full read here.
        # If file_type specific full paths are needed, this logic needs more conditions.
        
        # Temporarily load full data to get global normal distribution
        print("[DEBUG GlobalRef] Loading full data for reference normal selection...")
        # NOTE: file_cut will use its own dtypes. This might be different from the main data load if cut_type varies.
        #       It might be better to have a dedicated full data loader if dtypes or post-processing needs to be identical.
        full_data_for_ref = file_cut(file_type, file_path_for_global_ref, 'all') # Force 'all' to get all data
        full_data_for_ref.columns = full_data_for_ref.columns.str.strip()
        print(f"[DEBUG GlobalRef] Full data for ref loaded. Shape: {full_data_for_ref.shape}")

        # Apply same basic labeling as in Step 2 to this full data
        if file_type in ['MiraiBotnet', 'NSL-KDD']:
            full_data_for_ref['label'], _ = anomal_judgment_nonlabel(file_type, full_data_for_ref)
        elif file_type == 'netML':
            full_data_for_ref['label'] = full_data_for_ref['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        elif file_type == 'DARPA98':
            full_data_for_ref['label'] = full_data_for_ref['Class'].apply(lambda x: 0 if x == '-' else 1)
        elif file_type in ['CICModbus23', 'CICModbus']:
            full_data_for_ref['label'] = full_data_for_ref['Attack'].apply(lambda x: 0 if x.strip() == 'Baseline Replay: In position' else 1)
        elif file_type in ['IoTID20', 'IoTID']:
            full_data_for_ref['label'] = full_data_for_ref['Label'].apply(lambda x: 0 if x.strip() == 'Normal' else 1)
        else:
            full_data_for_ref['label'] = anomal_judgment_label(full_data_for_ref)
        print(f"[DEBUG GlobalRef] Full data labeled. Label counts: {full_data_for_ref['label'].value_counts(dropna=False)}")

        # Apply same embedding and group mapping (Steps 3, simplified for ref normal generation)
        full_data_for_ref_processed = time_scalar_transfer(full_data_for_ref.copy(), file_type) # Use copy
        # Assuming 'N' for regul for consistency in generating reference normals
        ref_embedded_df, _, ref_cat_map, ref_data_list = choose_heterogeneous_method(full_data_for_ref_processed, file_type, heterogeneous_method, 'N')
        ref_group_mapped_df, _ = map_intervals_to_groups(ref_embedded_df, ref_cat_map, ref_data_list, 'N')
        print(f"[DEBUG GlobalRef] Full data group mapped. Shape: {ref_group_mapped_df.shape}")
        
        # Apply MinMax scaling (like Step 3.5)
        # Note: Scalers from this step are not saved globally, only used for this ref normal generation
        ref_scalers_temp = {}
        ref_scaled_features_list = []
        if not ref_group_mapped_df.empty:
            for col_ref in ref_group_mapped_df.columns:
                scaler_ref = MinMaxScaler()
                ref_feature_vals = ref_group_mapped_df[col_ref].values.reshape(-1,1)
                ref_scaled_vals = scaler_ref.fit_transform(ref_feature_vals)
                ref_scaled_features_list.append(pd.Series(ref_scaled_vals.flatten(), name=col_ref, index=ref_group_mapped_df.index))
            ref_X_scaled = pd.concat(ref_scaled_features_list, axis=1)
        else:
            ref_X_scaled = pd.DataFrame(index=ref_group_mapped_df.index)
        print(f"[DEBUG GlobalRef] Full data scaled. Shape: {ref_X_scaled.shape}")

        # Apply PCA (like Step 4)
        # Assuming pca_want is 'Y' for generating global normals for consistency, unless specific file types always avoid PCA
        ref_pca_want = 'N' if file_type in ['DARPA98', 'DARPA', 'CICModbus23', 'CICModbus'] else 'Y'
        if ref_pca_want == 'Y':
            # Ensure pca_func can handle DataFrame input and returns DataFrame/NumPy array
            # If X_reduced is dataframe, it will have columns. If numpy, it won't.
            print("[DEBUG GlobalRef] Applying PCA to scaled full data for reference...")
            ref_X_pca = pca_func(ref_X_scaled) 
        else:
            print("[DEBUG GlobalRef] Skipping PCA for reference normal generation based on file_type.")
            ref_X_pca = ref_X_scaled.to_numpy() if hasattr(ref_X_scaled, 'to_numpy') else ref_X_scaled # Ensure NumPy array
        print(f"[DEBUG GlobalRef] Full data PCA applied. Shape: {ref_X_pca.shape}")

        # Now, from ref_X_pca and full_data_for_ref['label'], create the global reference
        all_normal_samples_pca_ref = ref_X_pca[full_data_for_ref['label'] == 0]
        num_all_normal_ref = all_normal_samples_pca_ref.shape[0]
        print(f"[DEBUG GlobalRef] Total normal samples in full data (PCA space): {num_all_normal_ref}")

        if num_all_normal_ref > 1:
            sample_size_ref = int(num_all_normal_ref * 0.8)
            if sample_size_ref == 0 and num_all_normal_ref > 0: sample_size_ref = 1 # Ensure at least 1 if possible
            # Handle cases where sample_size_ref might be larger than population if num_all_normal_ref is small
            if sample_size_ref > num_all_normal_ref : sample_size_ref = num_all_normal_ref 
            
            if sample_size_ref > 0 : # Proceed only if sample_size_ref is valid
                random_indices_ref = np.random.choice(num_all_normal_ref, size=sample_size_ref, replace=False)
                global_known_normal_samples_pca_for_cni = all_normal_samples_pca_ref[random_indices_ref]
                print(f"[DEBUG GlobalRef] Global reference normal samples (80% of all normals in full data, PCA space) created. Shape: {global_known_normal_samples_pca_for_cni.shape}")
            else:
                print("[WARN GlobalRef] Sample size for global reference normals is 0. No global reference created.")
                global_known_normal_samples_pca_for_cni = np.array([]) # Empty array

        elif num_all_normal_ref == 1:
            global_known_normal_samples_pca_for_cni = all_normal_samples_pca_ref
            print("[DEBUG GlobalRef] Global reference normal samples (1 sample from full data, PCA space) created.")
        else:
            print("[WARN GlobalRef] No normal samples found in the full dataset to create global reference.")
            global_known_normal_samples_pca_for_cni = np.array([]) # Empty array with potentially 0 columns if ref_X_pca was empty
            if ref_X_pca.ndim == 2 and ref_X_pca.shape[1] > 0: # Try to give it correct num_cols if possible
                 global_known_normal_samples_pca_for_cni = np.empty((0, ref_X_pca.shape[1]))

        del full_data_for_ref, full_data_for_ref_processed, ref_embedded_df, ref_group_mapped_df, ref_X_scaled, ref_X_pca, all_normal_samples_pca_ref # Free memory
        print("[DEBUG GlobalRef] Freed memory from temporary full data load.")

    except FileNotFoundError:
        print(f"[WARN GlobalRef] Full data file not found for {file_type} at expected path {file_path_for_global_ref}. Cannot create global normal reference.")
    except KeyError as ke:
        print(f"[WARN GlobalRef] KeyError during global normal reference creation (e.g., 'label' or other column missing): {ke}. Cannot create global normal reference.")
    except ValueError as ve:
        print(f"[WARN GlobalRef] ValueError during global normal reference creation: {ve}. Cannot create global normal reference.")
    except Exception as e:
        print(f"[ERROR GlobalRef] Failed to create global reference normal samples: {e}. Proceeding without it.")
        # Ensure it's None or an empty array so later logic doesn't break
        if global_known_normal_samples_pca_for_cni is not None and not isinstance(global_known_normal_samples_pca_for_cni, np.ndarray):
             global_known_normal_samples_pca_for_cni = None # Fallback
        elif isinstance(global_known_normal_samples_pca_for_cni, np.ndarray) and global_known_normal_samples_pca_for_cni.size == 0 and global_known_normal_samples_pca_for_cni.ndim == 1: # e.g. np.array([])
            # Try to give it 2 dims if it's an empty 1D array from np.array([]) init
            # This depends on whether ref_X_pca was successfully created to get num_cols
            # This part is tricky; ideally, it should be initialized with correct num_cols if possible
            pass # Keep as is, CNI function has robust empty checks

    timing_info['0_global_ref_creation'] = time.time() - start_global_ref
    print(f"Step 0 finished. Time: {timing_info['0_global_ref_creation']:.2f}s. Global ref shape: {global_known_normal_samples_pca_for_cni.shape if global_known_normal_samples_pca_for_cni is not None else 'None'}")


    # 1. Load data from csv
    start = time.time()
    # Define file_path for main data loading using original file_number from args
    file_path, file_number = file_path_line_nonnumber(file_type, args.file_number) 
    # cut_type = str(input("Enter the data cut type: "))
    if file_type in ['DARPA98', 'DARPA', 'NSL-KDD', 'NSL_KDD', 'CICModbus23', 'CICModbus', 'MitM', 'Kitsune', 'ARP']:
        cut_type = 'random'
    else:
        cut_type = 'all'
    data = file_cut(file_type, file_path, cut_type)

    # Clean column names by stripping leading/trailing whitespace
    data.columns = data.columns.str.strip()

    timing_info['1_load_data'] = time.time() - start


    # 2. Check data 'label'
    start = time.time()

    if file_type in ['MiraiBotnet', 'NSL-KDD']:
        data['label'], _ = anomal_judgment_nonlabel(file_type, data)
    elif file_type == 'netML':
        data['label'] = data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    elif file_type == 'DARPA98':
        data['label'] = data['Class'].apply(lambda x: 0 if x == '-' else 1)
    elif file_type in ['CICModbus23', 'CICModbus']:
        data['label'] = data['Attack'].apply(lambda x: 0 if x.strip() == 'Baseline Replay: In position' else 1)
    elif file_type in ['IoTID20', 'IoTID']:
        data['label'] = data['Label'].apply(lambda x: 0 if x.strip() == 'Normal' else 1)
    else:
        data['label'] = anomal_judgment_label(data)

    timing_info['2_label_check'] = time.time() - start


    # 3. Feature-specific embedding and preprocessing
    start = time.time()

    data = time_scalar_transfer(data, file_type)

    # regul = str(input("\nDo you want to Regulation? (Y/n): ")) # Whether to normalize or not
    regul = 'N'

    embedded_dataframe, feature_list, category_mapping, data_list = choose_heterogeneous_method(data, file_type, heterogeneous_method, regul)
    print("embedded_dataframe: ", embedded_dataframe)

    group_mapped_df, mapped_info_df = map_intervals_to_groups(embedded_dataframe, category_mapping, data_list, regul)
    print("mapped group: ", group_mapped_df)
    print("mapped_info: ", mapped_info_df)

    timing_info['3_embedding'] = time.time() - start


    # 3.5 Apply MinMaxScaler using the utility function
    X_scaled_for_pca, saved_scaler_path = apply_minmax_scaling_and_save_scalers(
        group_mapped_df,
        file_type,
        file_number,
        heterogeneous_method
        # base_output_dir can be specified if different from "results"
    )
    # Optionally, you can store saved_scaler_path if needed later, though it's printed by the function.
    

    # 4. Numpy(hstack) processing and PCA
    print("\nStep 4: PCA for main data processing...")
    start_pca_main = time.time() # PCA Timer Start
    X = X_scaled_for_pca # Use scaled data for PCA
    columns_data = list(data.columns)
    columns_X = list(X.columns)
    diff_columns = list(set(columns_data) - set(columns_X))
    print("data-X col: ", diff_columns)


    if file_type in ['DARPA98', 'DARPA', 'CICModbus23', 'CICModbus']:
        pca_want = 'N'
    else:
        pca_want = 'Y'

    # pca_want = str(input("\nDo you want to do PCA? (Y/n): "))
    if pca_want in ['Y', 'y']:
        if clustering_algorithm in ['CANNwKNN', 'CANN']:
            print("CANN is a classification, which means you need to use the full data.")
            X_reduced = X
        else:
            print("[PCA Main] Applying PCA...")
            X_reduced = pca_func(X)
            print("[PCA Main] PCA finished.")
    else:
        print("[PCA Main] Skipping PCA for main data based on user/file_type setting.")
        X_reduced = X

    print(f"\n[DEBUG Data_Labeling.py] X_reduced (data used for clustering) shape: {X_reduced.shape}")
    if hasattr(X_reduced, 'columns'): # X_reduced is a DataFrame
        print(f"[DEBUG Data_Labeling.py] X_reduced columns: {list(X_reduced.columns)}")
    else: # X_reduced is a NumPy array
        print(f"[DEBUG Data_Labeling.py] X_reduced is a NumPy array (no direct column names). First 5 cols of first row: {X_reduced[0, :5] if X_reduced.shape[0] > 0 else 'empty'}")
    
    # Note: Information about X (group_mapped_df, before PCA) is also good to output
    if hasattr(X, 'columns'):
        print(f"[DEBUG Data_Labeling.py] X (pre PCA, group_mapped_df) shape: {X.shape}")
        print(f"[DEBUG Data_Labeling.py] X (pre PCA, group_mapped_df) columns: {list(X.columns)}")

    end_pca_main = time.time() # PCA Timer End
    pca_duration_main = end_pca_main - start_pca_main
    print(f"[PCA Main] PCA processing (or skipping) for main data took: {pca_duration_main:.2f}s")

    # Create original labels aligned with X_reduced
    # Assumption: Rows in 'data' DataFrame correspond to rows in 'X' (group_mapped_df),
    # and subsequently to rows in 'X_reduced' if X is a DataFrame OR if pca_func preserves row order from X (if X is NumPy).
    # The logs show data.shape[0], X.shape[0], and X_reduced.shape[0] are all the same (2504267).
    
    if 'label' not in data.columns:
        raise ValueError("'label' column is missing from 'data' DataFrame. Ensure labeling step (Step 2) is correct.")
    
    # Check if X (group_mapped_df) has an index that can be used to align with 'data'
    # If X was derived from 'data' and row order is preserved, direct use of data['label'] is fine.
    # If X involved row reordering or filtering inconsistent with data, a more robust alignment (e.g., using original indices) would be needed.
    # For now, we assume direct correspondence in row order and length.
    if len(data) != X_reduced.shape[0]:
        # This case should ideally not happen if data processing keeps row counts consistent
        raise ValueError(f"Row count mismatch: 'data' ({len(data)}) vs 'X_reduced' ({X_reduced.shape[0]}). Cannot reliably align labels.")
    
    original_labels_for_X_reduced = data['label'].to_numpy()
    print(f"[DEBUG Data_Labeling.py] 'original_labels_for_X_reduced' created - Shape: {original_labels_for_X_reduced.shape}, Unique values: {np.unique(original_labels_for_X_reduced, return_counts=True)}")

    timing_info['4_pca_time'] = time.time() - start_pca_main # PCA Timer End
    print(f"Step 4 finished. PCA Time: {timing_info['4_pca_time']:.2f}s. X_pca shape: {X_reduced.shape if pca_want == 'Y' else 'PCA_SKIPPED'}")

    # Prepare data for clustering (either PCA output or scaled data if PCA was skipped)
    if pca_want == 'Y':
        data_for_clustering = X_reduced
    else:
        # Ensure X (scaled data) is numpy before chunking, if it's a DataFrame
        data_for_clustering = X.to_numpy() if hasattr(X, 'to_numpy') else X 
    
    original_labels_for_chunking = data['label'].to_numpy() # Ensure original labels are numpy array

    # 5. Clustering Algorithm Application (with Chunking)
    print("\nStep 5: Clustering Algorithm Application...")
    start = time.time()

    chunk_size = 2000
    num_samples = data_for_clustering.shape[0]
    num_chunks = math.ceil(num_samples / chunk_size)
    print(f"Total samples: {num_samples}, Chunk size: {chunk_size}, Number of chunks: {num_chunks}")

    all_chunk_cluster_labels = []
    all_chunk_best_params = [] # Store best_params from each chunk if needed, though may not be consistent
    all_chunk_gmm_types = [] # Added to collect GMM_type from autotune path

    for i in range(num_chunks):
        start_chunk_time = time.time()
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_samples)
        
        current_chunk_data = data_for_clustering[start_idx:end_idx]
        current_chunk_original_labels = original_labels_for_chunking[start_idx:end_idx]
        # The 'data' df passed to clustering functions might need to be chunked too if used internally beyond just labels.
        # For now, focusing on essential data for CNI.
        # current_data_df_chunk = data.iloc[start_idx:end_idx] 

        print(f"Processing chunk {i+1}/{num_chunks}, samples {start_idx}-{end_idx-1}. Chunk shape: {current_chunk_data.shape}")

        if eval_clustering_silhouette == 'y': # Autotune
            # Pass the current chunk data, its original labels, and the global PCA normals
            clustering_result_dict, gmm_type_chunk = choose_clustering_algorithm(
                data, # Original full data (might be needed by some hyperparam tuning, though ideally only chunked data used)
                current_chunk_data, 
                current_chunk_original_labels, 
                clustering_algorithm, 
                global_known_normal_samples_pca=global_known_normal_samples_pca_for_cni
            )
            all_chunk_gmm_types.append(gmm_type_chunk) # Collect GMM_type
        else: # Non-Autotune
            clustering_result_dict, _ = choose_clustering_algorithm_Non_optimization(
                data, # Original full data (see note above)
                current_chunk_data, 
                current_chunk_original_labels, 
                clustering_algorithm, 
                global_known_normal_samples_pca=global_known_normal_samples_pca_for_cni
            )
        
        all_chunk_cluster_labels.append(clustering_result_dict['Cluster_labeling'])
        all_chunk_best_params.append(clustering_result_dict.get('Best_parameter_dict')) # Store params
        
        end_chunk_time = time.time()
        print(f"Chunk {i+1} processed in {end_chunk_time - start_chunk_time:.2f}s. Labels predicted: {len(clustering_result_dict['Cluster_labeling'])}")

    # Concatenate results from all chunks
    if all_chunk_cluster_labels:
        final_predict_results = np.concatenate(all_chunk_cluster_labels)
    else:
        final_predict_results = np.array([]) # Handle cases with no data or no chunks
    
    print(f"All chunks processed. Total predicted labels: {len(final_predict_results)}. Expected: {num_samples}")
    if len(final_predict_results) != num_samples and num_samples > 0:
        print(f"[WARNING] Length of concatenated labels ({len(final_predict_results)}) does not match number of samples ({num_samples}).")
        # Fallback or error handling might be needed here.
        # For now, if lengths mismatch significantly, it might be better to use an empty array or raise error.
        # If it's a minor off-by-one due to slicing, it might be less critical but still needs investigation.
        # For simplicity in this step, we proceed but this is a critical check.

    # Assign concatenated results back to the original dataframe for evaluation
    # Ensure the index aligns if data was not a simple range index before chunking
    if len(final_predict_results) == len(data):
        data['cluster'] = final_predict_results
        # --- ADDED: Call cluster_mapping after final_predict_results are assigned ---
        cluster_mapping(data) # This will create data['adjusted_cluster']
        print(f"[INFO] cluster_mapping applied. 'adjusted_cluster' column created.")
        # --- END ADDED ---
    else:
        # If lengths don't match, avoid assigning to prevent errors. Store separately or handle.
        print(f"[ERROR] Final predicted labels length {len(final_predict_results)} does not match original data length {len(data)}. Cannot assign to data['cluster'].")
        # data['cluster'] will remain unassigned or have its old value.
        # This will likely cause issues in evaluation steps. For robust solution, this needs careful handling.
        # As a temporary measure for evaluation, we might need to use original_labels_for_chunking for y_true
        # and final_predict_results (if length matches original_labels_for_chunking) for y_pred.

    timing_info['5_clustering_time'] = time.time() - start
    print(f"Step 5 finished. Clustering Algorithm: {clustering_algorithm}. Time: {timing_info['5_clustering_time']:.2f}s")


    # 6. Evaluation and Comparison (using concatenated results)
    start = time.time()

    y_true = data['label'].to_numpy() # Ground truth labels from the full dataset
    
    # Prepare X_data for silhouette score calculation - use X_reduced (PCA or scaled data)
    # Ensure it's a numpy array for evaluation functions
    if isinstance(X_reduced, pd.DataFrame):
        X_data_for_eval = X_reduced.to_numpy()
    else:
        X_data_for_eval = X_reduced # Assuming it's already a NumPy array

    metrics_original = {}
    metrics_adjusted = {}

    if 'cluster' in data.columns and len(data['cluster']) == len(y_true):
        y_pred_original = data['cluster'].to_numpy()
        if eval_clustering_silhouette == 'y':
            metrics_original = evaluate_clustering(y_true, y_pred_original, X_data_for_eval)
        else:
            metrics_original = evaluate_clustering_wos(y_true, y_pred_original)
        print("Clustering Scores (Original - using 'cluster' column): ", metrics_original)
    else:
        print("[WARN Eval] 'cluster' column not available or length mismatch. Skipping original metrics calculation.")

    if 'adjusted_cluster' in data.columns and len(data['adjusted_cluster']) == len(y_true):
        y_pred_adjusted = data['adjusted_cluster'].to_numpy()
        if eval_clustering_silhouette == 'y':
            metrics_adjusted = evaluate_clustering(y_true, y_pred_adjusted, X_data_for_eval)
        else:
            metrics_adjusted = evaluate_clustering_wos(y_true, y_pred_adjusted)
        print("Clustering Scores (Adjusted - using 'adjusted_cluster' column): ", metrics_adjusted)
    else:
        print("[WARN Eval] 'adjusted_cluster' column not available or length mismatch. Skipping adjusted metrics calculation.")

    # Fallback for y_pred if 'cluster' column was not populated, for functions expecting a single y_pred
    # This part was from the chunking logic, ensure it's still relevant or integrated with metrics_original
    y_pred_for_general_use = final_predict_results
    if len(y_pred_for_general_use) != len(y_true) and len(y_true) > 0:
        print(f"[WARN Eval] Length mismatch between y_true ({len(y_true)}) and y_pred_for_general_use ({len(y_pred_for_general_use)}). Evaluation might be incorrect or fail.")
        if len(y_pred_for_general_use) == 0 and len(y_true) > 0:
            print("[WARN Eval] y_pred_for_general_use is empty. Using dummy predictions (all zeros) to avoid crash.")
            y_pred_for_general_use = np.zeros_like(y_true)
    
    # Ensure y_pred_for_general_use is not empty before general evaluation (if any part still uses it directly)
    # The primary evaluation is now through metrics_original and metrics_adjusted.

    # --- This section with diff_columns might be redundant if metrics_original/adjusted cover all cases ---
    # if len(y_pred_for_general_use) > 0:
    #     # if 'label' is an original column, use evaluate_clustering_wos, otherwise use evaluate_clustering.
    #     if 'label' in diff_columns: # diff_columns was defined much earlier, check its scope and relevance
    #         # This logic might be superseded by the specific metrics_original/metrics_adjusted calls above
    #         pass # clustering_scores = evaluate_clustering_wos(y_true, y_pred_for_general_use)
    #     else:
    #         pass # clustering_scores = evaluate_clustering(y_true.flatten(), y_pred_for_general_use.flatten(), X_data_for_eval)
    #     # print("Clustering Scores (based on concatenated results / y_pred_for_general_use): ", clustering_scores)
    # else:
    #     print("[ERROR] No predicted labels (y_pred_for_general_use is empty). Skipping some evaluation parts.")
    # --- End potentially redundant section ---

    timing_info['6_evaluation_time'] = time.time() - start
    print(f"Step 6 finished. Evaluation Time: {timing_info['6_evaluation_time']:.2f}s")

    # 7. Save results
    start = time.time()

    # Determine GMM_type_for_save from collected chunk GMM_types
    # For simplicity, if GMM was used and any chunk returned a GMM_type, use the first one found.
    # This might need more sophisticated handling if GMM_types can vary meaningfully across chunks for the same run.
    GMM_type_for_save = None
    if clustering_algorithm.upper() == "GMM" and eval_clustering_silhouette == 'y':
        for gmm_t in all_chunk_gmm_types:
            if gmm_t is not None:
                GMM_type_for_save = gmm_t
                break
        if GMM_type_for_save:
            print(f"[INFO Save] Using GMM_type: {GMM_type_for_save} for saving.")
        else:
            print("[WARN Save] GMM algorithm was used (Autotune), but no GMM_type collected from chunks. GMM_type specific filenames might be affected.")
    elif clustering_algorithm.upper() == "GMM": # Non-autotune GMM might still need a default GMM_type if save funcs expect it
        # In Non_optimization, choose_clustering_algorithm_Non_optimization doesn't return GMM_type explicitly.
        # It's usually embedded in the clustering_algorithm string itself (e.g., "GMM_normal").
        # We might need to parse it from clustering_algorithm or assume a default if critical for filename.
        # For now, if not autotune, GMM_type_for_save will remain None unless explicitly handled.
        # save_csv.py handles GMM_type=None for filenames, so this might be acceptable.
        parts = clustering_algorithm.split('_')
        if len(parts) > 1 and parts[0].upper() == 'GMM':
             GMM_type_for_save = parts[1] # e.g., "normal" from "GMM_normal"
             print(f"[INFO Save] Parsed GMM_type: {GMM_type_for_save} from clustering_algorithm for saving.")


    # Call csv_compare_clustering with correct arguments
    # Signature: csv_compare_clustering(file_type, clusterint_method, file_number, data, GMM_type=None)
    if 'cluster' in data.columns and 'adjusted_cluster' in data.columns: # Ensure necessary columns exist
        csv_compare_clustering(file_type=file_type, 
                               clusterint_method=clustering_algorithm, 
                               file_number=file_number, 
                               data=data, 
                               GMM_type=GMM_type_for_save)
        print(f"[INFO Save] csv_compare_clustering called.")
    else:
        print("[WARN Save] 'cluster' or 'adjusted_cluster' not in data. Skipping csv_compare_clustering.")

    # Call csv_compare_matrix_clustering with correct arguments
    # Signature: csv_compare_matrix_clustering(file_type, file_number, clusterint_method, metrics_original, metrics_adjusted, GMM_type)
    if metrics_original and metrics_adjusted: # Ensure metrics were calculated
        csv_compare_matrix_clustering(file_type=file_type, 
                                      file_number=file_number, 
                                      clusterint_method=clustering_algorithm, 
                                      metrics_original=metrics_original, 
                                      metrics_adjusted=metrics_adjusted, 
                                      GMM_type=GMM_type_for_save)
        print(f"[INFO Save] csv_compare_matrix_clustering called.")
    elif not metrics_original:
        print("[WARN Save] metrics_original is empty. Skipping csv_compare_matrix_clustering.")
    elif not metrics_adjusted:
        print("[WARN Save] metrics_adjusted is empty. Skipping csv_compare_matrix_clustering.")


    timing_info['7_save_results_time'] = time.time() - start
    print(f"Step 7 finished. Save Time: {timing_info['7_save_results_time']:.2f}s")


    # Calculate total time
    total_end_time = time.time()
    timing_info['0_total_time'] = total_end_time - total_start_time

    # Save time information as a CSV
    time_save_csv_VL(file_type, file_number, clustering_algorithm, timing_info)


    return


if __name__ == '__main__':
    main()