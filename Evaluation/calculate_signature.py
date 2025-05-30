# Evaluate TP, TN, FP, FN by comparing each signature (list dictionary) to a real dataset
# Return: [{'Signature_dict': signature_name, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}, {}, ...]

import pandas as pd
import numpy as np
import multiprocessing # Added for parallel processing
import traceback # Added for detailed error logging

# Helper function for parallel signature calculation
def _calculate_single_signature_metrics(args):
    data_subset, signature = args
    # Ensure all keys in signature exist in data_subset columns to avoid KeyError during .eq()
    # This might be overly cautious if data is guaranteed to have all keys, 
    # but helps if signatures can have keys not in data (though that implies a problem upstream)
    valid_signature_keys = [k for k in signature.keys() if k in data_subset.columns]
    if not valid_signature_keys: # or if set(signature.keys()) != set(valid_signature_keys):
        # Handle cases where signature keys are not in data, or are empty.
        # This indicates an issue, perhaps log it. For now, return zero metrics.
        # print(f"Warning: Signature {signature} has keys not in data or is effectively empty against data. Skipping.")
        return {
            'Signature_dict': signature,
            'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0
        }

    # Proceed with calculation only using valid keys present in the data
    # Create a filtered signature for the .eq() comparison
    filtered_signature_for_comparison = {k: signature[k] for k in valid_signature_keys}

    # It is crucial that data_subset passed to this worker function already has the 'label' column.
    # Ensure data[valid_signature_keys] doesn't create an empty DataFrame if valid_signature_keys is empty.
    # The check for `if not valid_signature_keys:` above should handle this.
    matches = data_subset[valid_signature_keys].eq(pd.Series(filtered_signature_for_comparison)).all(axis=1)

    # Calculate TP, FN, FP, TN
    TP = ((matches) & (data_subset['label'] == 1)).sum()
    FN = ((~matches) & (data_subset['label'] == 1)).sum()
    FP = ((matches) & (data_subset['label'] == 0)).sum()
    TN = ((~matches) & (data_subset['label'] == 0)).sum()

    return {
        'Signature_dict': signature,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }

def calculate_signature(data, signatures):
    if not signatures: # Handle empty signatures list
        return []

    # Prepare data for workers: select all unique keys from all signatures plus 'label'
    # This avoids sending the whole dataframe `data` to each worker if it's very wide.
    all_signature_keys = set()
    for sig in signatures:
        all_signature_keys.update(sig.keys())
    
    columns_to_select = list(all_signature_keys)
    if 'label' not in columns_to_select:
        columns_to_select.append('label')
    
    # Ensure all selected columns actually exist in the input dataframe `data`
    existing_columns_in_data = [col for col in columns_to_select if col in data.columns]
    if not existing_columns_in_data or 'label' not in existing_columns_in_data:
        print(f"Warning: 'label' column ({'label' in existing_columns_in_data}) or critical signature columns (all_signature_keys present: {all(k in existing_columns_in_data for k in all_signature_keys)}) missing from data in calculate_signature. Returning empty results.") # More detailed warning
        # It might be better to raise an error or log extensively here
        return [{'Signature_dict': sig, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for sig in signatures]

    data_subset_for_processing = data[existing_columns_in_data].copy() # Use .copy() to avoid SettingWithCopyWarning on slices if data is a slice

    tasks = [(data_subset_for_processing, sig) for sig in signatures]
    results = []

    if tasks:
        # Restore dynamic process count based on available cores and number of tasks
        num_processes = min(len(tasks), multiprocessing.cpu_count())
        if num_processes == 0 and len(tasks) > 0: # Should not happen if cpu_count() >=1
             num_processes = 1
        
        print(f"[CalcSig] Using {num_processes} processes for {len(tasks)} signatures.")
        try:
            if num_processes > 0:
                 with multiprocessing.Pool(processes=num_processes) as pool:
                    results = pool.map(_calculate_single_signature_metrics, tasks)
            else:
                results = []

        except Exception as e:
            print(f"Error during parallel signature calculation: {e}")
            traceback.print_exc()
            print("Falling back to sequential processing for calculate_signature...")
            results = []
            for task_arg in tasks:
                results.append(_calculate_single_signature_metrics(task_arg))
    
    return results


# Helper function for parallel recall calculation
def _calculate_recall_for_row_chunk(args):
    data_chunk, signatures_list = args
    # Ensure signatures_list is not empty and data_chunk is not empty
    if not signatures_list or data_chunk.empty:
        return 0, 0, 0, 0 # TP, FN, FP, TN

    TP_chunk = 0
    FN_chunk = 0
    FP_chunk = 0
    TN_chunk = 0

    for _, row in data_chunk.iterrows():
        row_satisfied = False
        # Check if the row satisfies any of the signatures
        for sig_item in signatures_list:
            actual_signature = sig_item['signature_name']['Signature_dict']
            # Check if all conditions in the current signature are met by the row
            # Also ensure all keys in actual_signature exist in the row to prevent KeyError
            if all(k in row and row[k] == v for k, v in actual_signature.items()):
                row_satisfied = True
                break # Row matches at least one signature
        
        if row['label'] == 1:
            if row_satisfied:
                TP_chunk += 1
            else:
                FN_chunk += 1
        else: # label == 0
            if row_satisfied:
                FP_chunk += 1
            else:
                TN_chunk += 1
    return TP_chunk, FN_chunk, FP_chunk, TN_chunk

# Tools for evaluating recall in an aggregated signature collection
def calculate_signatures(data, signatures): # Renamed from original calculate_signatures to avoid conflict if any old calls remain by mistake
    if not signatures: # if signatures list is empty or None
        return 0.0

    # Extract only the actual signature conditions and needed columns for the data subset
    needed_columns = set()
    processed_signatures_for_workers = [] # Store the part of signature needed by workers

    for signature_eval_metric in signatures: # `signatures` here is the list of dicts from signature_evaluate/under_limit
        if 'signature_name' in signature_eval_metric and isinstance(signature_eval_metric['signature_name'], dict) and \
           'Signature_dict' in signature_eval_metric['signature_name']:
            actual_signature_dict = signature_eval_metric['signature_name']['Signature_dict']
            if isinstance(actual_signature_dict, dict):
                needed_columns.update(actual_signature_dict.keys())
                # Pass a simplified structure to workers if possible, or the necessary part
                processed_signatures_for_workers.append({'signature_name': {'Signature_dict': actual_signature_dict}})
            else:
                # Log or handle malformed signature_eval_metric['signature_name']['Signature_dict']
                # print(f"Skipping malformed actual_signature_dict: {actual_signature_dict}")
                pass # Or continue, or raise error
        else:
            # Log or handle malformed signature_eval_metric structure
            # print(f"Skipping malformed signature_eval_metric: {signature_eval_metric}")
            pass # Or continue, or raise error

    if not processed_signatures_for_workers:
        # print("No valid signatures found to process for recall calculation.")
        return 0.0 # No signatures to check against

    needed_columns.add('label')
    
    # Ensure all needed columns exist in the input dataframe `data`
    existing_needed_columns = [col for col in list(needed_columns) if col in data.columns]
    if 'label' not in existing_needed_columns:
        # print("Warning: 'label' column or critical signature columns missing from data in calculate_signatures (recall). Returning 0 recall.")
        return 0.0
    
    # Create the data_subset with only the necessary columns
    # Use .copy() to prevent SettingWithCopyWarning if `data` is a slice itself
    data_subset = data[existing_needed_columns].copy() 

    if data_subset.empty:
        # print("Data subset is empty after selecting needed columns for recall calculation.")
        return 0.0

    num_processes = multiprocessing.cpu_count()
    # Split data into chunks for parallel processing
    # Adjust chunk_size dynamically or set to a fixed reasonable number
    # For very large datasets, too many small chunks can also be inefficient.
    num_rows = len(data_subset)
    if num_rows == 0: return 0.0

    # Determine chunk size. If num_rows is small, it might be better to run sequentially or with fewer processes.
    # Let's aim for roughly num_processes * 2 to num_processes * 4 chunks if data is large enough.
    # Or a simpler approach: ensure chunk_size is at least 1.
    # And avoid creating more chunks than rows.
    ideal_chunks_count = num_processes * 2 # Aim for a few chunks per process
    chunk_size = max(1, (num_rows + ideal_chunks_count - 1) // ideal_chunks_count) # Ensure it's at least 1
    if chunk_size * num_processes > num_rows * 2 and num_rows > num_processes: # Heuristic to prevent tiny chunks if num_rows isn't massive
        chunk_size = (num_rows + num_processes -1) // num_processes

    data_chunks = [data_subset.iloc[i:i + chunk_size] for i in range(0, num_rows, chunk_size)]
    
    tasks = [(chunk, processed_signatures_for_workers) for chunk in data_chunks if not chunk.empty]

    total_TP = 0
    total_FN = 0
    # total_FP = 0 # Not strictly needed for recall, but good to have if we want full metrics
    # total_TN = 0

    if not tasks:
        # print("No tasks created for parallel recall calculation, possibly due to empty data_chunks.")
        return 0.0

    # print(f"[CalcRecall] Using {num_processes} processes for {len(data_chunks)} data chunks (chunk_size ~{chunk_size}). Total rows: {num_rows}")
    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            results_from_pool = pool.map(_calculate_recall_for_row_chunk, tasks)
        
        for tp_c, fn_c, fp_c, tn_c in results_from_pool:
            total_TP += tp_c
            total_FN += fn_c
            # total_FP += fp_c
            # total_TN += tn_c

    except Exception as e:
        print(f"Error during parallel recall calculation: {e}")
        traceback.print_exc()
        # Fallback to sequential processing for debugging or robustness
        print("Falling back to sequential processing for recall calculation...")
        total_TP, total_FN = 0, 0 # Reset for sequential run
        for chunk in data_chunks:
            if not chunk.empty:
                tp_c, fn_c, _, _ = _calculate_recall_for_row_chunk((chunk, processed_signatures_for_workers))
                total_TP += tp_c
                total_FN += fn_c

    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    # print(f"[CalcRecall] Calculated Recall: {recall} (TP: {total_TP}, FN: {total_FN})")
    return recall
