# Input embedded_data (after separate_bin)

import pandas as pd
from Heterogeneous_Method.build_interval_mapping import build_interval_mapping_dataframe
import multiprocessing # Added for parallel processing

# Helper function for parallel interval mapping
def _map_single_interval_column(args):
    col_name, col_data, col_mapping_series = args
    interval_to_group = {}
    for s in col_mapping_series.dropna(): # process actual mapping strings
        try:
            interval_str, group_num = s.split('=')
            interval_to_group[interval_str.strip()] = int(group_num.strip())
        except ValueError:
            # print(f"Invalid format in mapping for column {col_name}: {s}")
            pass # Or handle more robustly
    
    mapped_series = col_data.astype(str).map(interval_to_group)
    return col_name, mapped_series, interval_to_group

def map_intervals_to_groups(df, category_mapping, data_list, regul='N'):
    # mapped_df = pd.DataFrame() # Initialize later from results
    mapping_info = {}   # Save per-feature mapping information

    # Ensure 'interval' key exists and is a DataFrame
    if 'interval' not in category_mapping or not isinstance(category_mapping['interval'], pd.DataFrame):
        # If no interval mapping, or mapping is not in expected format, handle appropriately
        # For now, assume interval_columns would be empty, or raise error
        print("Warning/Error: 'interval' mapping is missing or not a DataFrame in category_mapping.")
        interval_columns = []
        interval_df = pd.DataFrame(index=df.index) # Empty DataFrame with original index
    else:
        interval_columns = [col for col in category_mapping['interval'].columns if col in df.columns]
        if not interval_columns:
            print("No common columns found between df and interval_mapping for interval processing.")
            interval_df = pd.DataFrame(index=df.index) # Empty DataFrame with original index
        else:
            interval_df = df[interval_columns]  # Organize only the conditions that want to map

    interval_mapping_df = category_mapping.get('interval', pd.DataFrame()) # interval information is taken from here

    tasks = []
    for col in interval_df.columns: # Iterate over columns present in both df and interval_mapping_df
        if col not in interval_mapping_df.columns:
            # This case should be less likely now due to pre-filtering interval_columns
            # print(f"Warning: Interval mapping for column `{col}` is missing in interval_mapping_df. Skipping.")
            # mapped_df[col] = interval_df[col] # Keep original if no mapping? Or NaN?
            continue 
        tasks.append((col, interval_df[col], interval_mapping_df[col]))

    processed_columns = {}
    if tasks:
        num_processes = min(len(tasks), multiprocessing.cpu_count())
        # print(f"[MapIntervals] Using {num_processes} processes for {len(tasks)} interval columns.")
        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(_map_single_interval_column, tasks)
        except Exception as e:
            print(f"Error during parallel interval mapping: {e}. Falling back to sequential.")
            results = [_map_single_interval_column(task) for task in tasks]
        
        for col_name, mapped_series, i_to_g in results:
            processed_columns[col_name] = mapped_series
            mapping_info[col_name] = i_to_g
    
    if processed_columns:
        mapped_df_from_intervals = pd.DataFrame(processed_columns, index=interval_df.index) # Ensure correct index
    else:
        # Ensure mapped_df is initialized correctly if no interval columns were processed
        mapped_df_from_intervals = pd.DataFrame(index=df.index) 

    # Concatenation logic (ensure data_list[0] and data_list[-1] have compatible indices)
    # It is assumed that data_list[0] (categorical) and data_list[-1] (binary) are already processed 
    # and have an index compatible with df and thus with mapped_df_from_intervals.
    # If data_list elements are empty DataFrames, they should also have the correct index.
    
    # Ensure all parts have an index, default to df.index if they are empty and don't have one
    part1 = data_list[0] if not data_list[0].empty else pd.DataFrame(index=df.index)
    part3 = data_list[len(data_list)-1] if not data_list[len(data_list)-1].empty else pd.DataFrame(index=df.index)

    # Ensure mapped_df_from_intervals also has the correct index if it's empty
    if mapped_df_from_intervals.empty and not processed_columns:
        mapped_df_from_intervals = pd.DataFrame(index=df.index)

    # Filter out empty DataFrames before concat to avoid issues if one part is truly empty of columns but has an index.
    # Concat will handle empty DFs gracefully if they have an index. The main concern is if they are None or lack an index.
    dfs_to_concat = []
    if not part1.empty or part1.shape[1] > 0: dfs_to_concat.append(part1)
    if not mapped_df_from_intervals.empty or mapped_df_from_intervals.shape[1] > 0: dfs_to_concat.append(mapped_df_from_intervals)
    if not part3.empty or part3.shape[1] > 0: dfs_to_concat.append(part3)
    
    if dfs_to_concat:
        mapped_df = pd.concat(dfs_to_concat, axis=1)
    else: # All parts were empty
        mapped_df = pd.DataFrame(index=df.index)

    mapped_info_df = build_interval_mapping_dataframe(mapping_info)

    if regul in ["N", "n"]:
        mapped_info_df = pd.concat([
            category_mapping.get('categorical', pd.DataFrame()),
            mapped_info_df,
            category_mapping.get('binary', pd.DataFrame())
        ], axis=1, ignore_index=False)

    return mapped_df, mapped_info_df
