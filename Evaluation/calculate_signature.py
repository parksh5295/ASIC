# Evaluate TP, TN, FP, FN by comparing each signature (list dictionary) to a real dataset
# Return: [{'Signature_dict': signature_name, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}, {}, ...]

import pandas as pd
import numpy as np
import multiprocessing # Added for parallel processing

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
        # print("Warning: 'label' column or critical signature columns missing from data in calculate_signature. Returning empty results.")
        return [{'Signature_dict': sig, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for sig in signatures] # Or handle error appropriately

    data_subset_for_processing = data[existing_columns_in_data].copy() # Use .copy() to avoid SettingWithCopyWarning on slices if data is a slice

    tasks = [(data_subset_for_processing, sig) for sig in signatures]
    results = []

    if tasks:
        num_processes = min(len(tasks), multiprocessing.cpu_count())
        # print(f"[CalcSig] Using {num_processes} processes for {len(tasks)} signatures.")
        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(_calculate_single_signature_metrics, tasks)
        except Exception as e:
            print(f"Error during parallel signature calculation: {e}. Falling back to sequential.")
            # Fallback to sequential processing
            results = []
            for sig in signatures:
                # In sequential fallback, pass the same data_subset_for_processing
                results.append(_calculate_single_signature_metrics((data_subset_for_processing, sig)))
    
    return results


# Tools for evaluating recall in an aggregated signature collection
def calculate_signatures(data, signatures):
    # Extract only the actual signature conditions
    needed_columns = set()
    for signature in signatures:
        # extract the actual feature from the Signature_dict inside the signature_name key
        if 'signature_name' in signature and 'Signature_dict' in signature['signature_name']:
            actual_signature = signature['signature_name']['Signature_dict']
            needed_columns.update(actual_signature.keys())
    
    needed_columns.add('label')
    data_subset = data[list(needed_columns)]
    
    TP = FN = FP = TN = 0
    for _, row in data_subset.iterrows():
        # Compare with Signature_dict for each signature
        row_satisfied = any(
            all(row[k] == sig['signature_name']['Signature_dict'][k] 
                for k in sig['signature_name']['Signature_dict'].keys())
            for sig in signatures
        )
        
        if row['label'] == 1:
            if row_satisfied:
                TP += 1
            else:
                FN += 1
        else:
            if row_satisfied:
                FP += 1
            else:
                TN += 1

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return recall
