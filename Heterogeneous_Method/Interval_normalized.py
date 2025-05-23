# Encoding and Normalization
# Section length; keep the number of sections the same for each section
# Output file: A dataframe separated by groups, with group names substituted for feature values
# The output is a dataframe and feature list divided into groups.

import pandas as pd
from Heterogeneous_Method.Feature_Encoding import Heterogeneous_Feature_named_featrues, Heterogeneous_Feature_named_combine, Heterogeneous_Feature_named_combine_standard
from utils.separate_bin import interval_length_Inverse_Count
from Heterogeneous_Method.build_interval_mapping import build_interval_mapping_dataframe
import multiprocessing # Added for parallel processing

# Helper function for parallel execution of interval_length_Inverse_Count
def _process_feature_group(args):
    data_semilist_local, feature_semilist_local = args
    if not feature_semilist_local: # Check if feature list for the group is empty
        return pd.DataFrame(), {} # Return empty df and mapping if no features
    return interval_length_Inverse_Count(data_semilist_local, feature_semilist_local)

def Heterogeneous_Interval_Inverse(data, file_type, regul):
    feature_name = Heterogeneous_Feature_named_featrues(file_type)

    categorical_features = feature_name['categorical_features']
    time_features = feature_name['time_features']
    packet_length_features = feature_name['packet_length_features']
    count_features = feature_name['count_features']
    binary_features = feature_name['binary_features']

    feature_list = [categorical_features, time_features, packet_length_features, count_features, binary_features]
    print("hey: ", feature_list)

    df = pd.DataFrame() # A dataframe to store the entire condition

    if regul in ['Y', 'y']:
        data_list = Heterogeneous_Feature_named_combine_standard(categorical_features, time_features, packet_length_features, count_features, binary_features, data)
        category_mapping = False

        full_group_mapping_info = {}  # a dict to hold the mapping information for all features
        
        # Prepare tasks for parallel execution (excluding categorical_features at index 0)
        tasks_std = []
        for i in range(1, len(data_list)):
            if data_list[i].empty or not feature_list[i]: # Skip empty dataframes or feature lists
                continue
            tasks_std.append((data_list[i], feature_list[i]))

        processed_results_std = []
        if tasks_std:
            num_processes = min(len(tasks_std), multiprocessing.cpu_count())
            print(f"[Hetero_Interval_Inverse STD] Using {num_processes} processes for {len(tasks_std)} feature groups.")
            try:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    processed_results_std = pool.map(_process_feature_group, tasks_std)
            except Exception as e:
                print(f"Error during parallel processing (STD): {e}. Falling back to sequential.")
                processed_results_std = [_process_feature_group(task) for task in tasks_std]

        # Concatenate results
        dfs_to_concat_std = []
        for small_df, group_mapping_info_small in processed_results_std:
            if not small_df.empty:
                dfs_to_concat_std.append(small_df)
            if group_mapping_info_small:
                full_group_mapping_info.update(group_mapping_info_small)
        
        if dfs_to_concat_std:
            df = pd.concat(dfs_to_concat_std, axis=1, ignore_index=False)
        else:
            df = pd.DataFrame() # Ensure df is an empty DataFrame if no results

        # The original category_mapping for regul='Y' was False. If interval_mapping is still needed:
        # This part might need clarification based on how category_mapping is used later for regul='Y'
        # If full_group_mapping_info is populated, we might still want to build interval_mapping_df
        if full_group_mapping_info: # Check if any mapping info was generated
             interval_mapping_df_std = build_interval_mapping_dataframe(full_group_mapping_info)
             # How to integrate this with category_mapping = False needs to be defined.
             # For now, let's assume category_mapping remains False, and this df is for other uses or logging.
             # print("Interval mapping for standardized data:", interval_mapping_df_std)
        

    elif regul in ['N', 'n']:
        data_list, category_mapping = Heterogeneous_Feature_named_combine(categorical_features, time_features, packet_length_features, count_features, binary_features, data)

        full_group_mapping_info = {}  # a dict to hold the mapping information for all features
        
        # Prepare tasks for parallel execution (excluding categorical_features at index 0 and binary_features at the end)
        tasks_non_std = []
        # Loop from index 1 up to second to last (len(data_list)-2 inclusive for upper bound)
        for i in range(1, len(data_list) -1): 
            if data_list[i].empty or not feature_list[i]: # Skip empty dataframes or feature lists
                continue
            tasks_non_std.append((data_list[i], feature_list[i]))
        
        processed_results_non_std = []
        if tasks_non_std:
            num_processes = min(len(tasks_non_std), multiprocessing.cpu_count())
            print(f"[Hetero_Interval_Inverse Non-STD] Using {num_processes} processes for {len(tasks_non_std)} feature groups.")
            try:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    processed_results_non_std = pool.map(_process_feature_group, tasks_non_std)
            except Exception as e:
                print(f"Error during parallel processing (Non-STD): {e}. Falling back to sequential.")
                processed_results_non_std = [_process_feature_group(task) for task in tasks_non_std]

        # Concatenate results from parallel processing
        dfs_to_concat_non_std = []
        for small_df, group_mapping_info_small in processed_results_non_std:
            if not small_df.empty:
                dfs_to_concat_non_std.append(small_df)
            if group_mapping_info_small:
                full_group_mapping_info.update(group_mapping_info_small)
        
        if dfs_to_concat_non_std:
            df = pd.concat(dfs_to_concat_non_std, axis=1, ignore_index=False)
        else:
            df = pd.DataFrame() # Ensure df is an empty DataFrame if no results

        # Adding Binary Features (data_list[len(data_list)-1]) to DF Later
        # Ensure it is a DataFrame before concat
        binary_features_df = data_list[len(data_list)-1]
        if not isinstance(binary_features_df, pd.DataFrame):
             # This case should not happen if Heterogeneous_Feature_named_combine is consistent
             print("Warning: Binary features part is not a DataFrame. Converting.")
             if isinstance(binary_features_df, pd.Series):
                 binary_features_df = binary_features_df.to_frame()
             else: # Or handle other types / raise error
                 binary_features_df = pd.DataFrame(binary_features_df) 

        if not binary_features_df.empty:
            if df.empty:
                df = binary_features_df
            else:
                df = pd.concat([df, binary_features_df], axis=1, ignore_index=False)

        interval_mapping_df = build_interval_mapping_dataframe(full_group_mapping_info)
        category_mapping['interval'] = interval_mapping_df  # This, in turn, calls the mapping_info
        print("category_mapping: ", category_mapping)

    df = pd.concat([data_list[0], df], axis=1, ignore_index=False)   # Adding Categorical Features to DF Later

    print("embedded data: ", df)

    return df, feature_list, category_mapping, data_list   # df = embedded data
    # category_mapping: dict; categorical, interval, binary (key) -> Complete mapping_info
    # data_list; has categorical, binary feature mapping information in it