import multiprocessing # Added for parallel processing


# Helper function for NSL-KDD column validation (for parallel processing)
def _is_nsl_kdd_column_valid(args):
    col_name, series_data, threshold, min_distinct_frequent_values = args
    value_counts = series_data.value_counts()
    count_distinct_frequent = sum(1 for count in value_counts if count >= threshold)
    return col_name, count_distinct_frequent >= min_distinct_frequent_values

# Functions to proactively remove rare items
def remove_rare_columns(df, min_support_ratio, file_type=None, min_distinct_frequent_values=2):
    '''
    if file_type in ['NSL-KDD', 'NSL_KDD']:
        # NSL-KDD is applied with a lower threshold (only 20% of the original min_support_ratio)
        threshold = int(len(df) * min_support_ratio * 0.2)
    else:
    '''
    threshold = int(len(df) * min_support_ratio)
    
    if file_type in ['NSL-KDD', 'NSL_KDD']:
        # NSL-KDD modified logic
        original_cols = df.columns.tolist() # Ensure it's a list for multiprocessing tasks
        valid_cols = []
        threshold = int(len(df) * min_support_ratio * 0.2) # Calculate threshold once

        tasks = [(col, df[col], threshold, min_distinct_frequent_values) for col in original_cols]

        if tasks:
            num_processes = min(len(tasks), multiprocessing.cpu_count())
            # print(f"[RemoveRare] Using {num_processes} processes for {len(tasks)} columns (NSL-KDD).")
            try:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    results = pool.map(_is_nsl_kdd_column_valid, tasks)
                valid_cols = [col_name for col_name, is_valid in results if is_valid]
            except Exception as e:
                print(f"Error during parallel column validation (NSL-KDD): {e}. Falling back to sequential.")
                # Fallback to sequential processing
                valid_cols = []
                for col in original_cols:
                    value_counts = df[col].value_counts()
                    count_distinct_frequent = sum(1 for count in value_counts if count >= threshold)
                    if count_distinct_frequent >= min_distinct_frequent_values:
                        valid_cols.append(col)
        else: # No columns to process
            valid_cols = original_cols # or empty list depending on desired behavior for empty df

        print(f"Original columns: {len(original_cols)}")
        print(f"Threshold value (for individual value frequency): {threshold}")
        print(f"Required distinct frequent values: {min_distinct_frequent_values}")
        print(f"Remaining columns after filtering: {len(valid_cols)}")

        if len(valid_cols) == 0:
            print("Warning: All columns were filtered out! Using original columns instead.")
            return df

        # Safeguards: If too many columns are removed (e.g., fewer than 5), consider keeping the originals
        if len(valid_cols) < 5 and len(original_cols) >= 5 :
             print(f"Warning: Too few columns remaining ({len(valid_cols)}). Falling back to original columns to avoid issues.")
             return df

        return df[valid_cols]
    else:
        # Keep the original logic
        threshold = int(len(df) * min_support_ratio)
        valid_cols = df.columns[df.sum() > threshold]
        return df[valid_cols]

