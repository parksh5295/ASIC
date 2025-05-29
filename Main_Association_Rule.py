# A program that automatically generates signatures using association rules.

import argparse
import numpy as np
import time
import multiprocessing # Added for parallel processing
import functools # Added for functools.partial if needed, though starmap is used here
import os
from Dataset_Choose_Rule.association_data_choose import file_path_line_association
from Dataset_Choose_Rule.choose_amount_dataset import file_cut
from definition.Anomal_Judgment import anomal_judgment_label, anomal_judgment_nonlabel
from utils.time_transfer import time_scalar_transfer
from utils.class_row import anomal_class_data, without_labelmaking_out, nomal_class_data, without_label
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
from utils.remove_rare_columns import remove_rare_columns
from Modules.Association_module import association_module
from Modules.Signature_evaluation_module import signature_evaluate
from Modules.Signature_underlimit import under_limit
from Evaluation.calculate_signature import calculate_signatures
from Modules.Difference_sets import dict_list_difference
from Dataset_Choose_Rule.save_csv import csv_association
from Dataset_Choose_Rule.time_save import time_save_csv_CS

# Helper function for parallel processing
def process_confidence_iteration(min_confidence, anomal_grouped_data, nomal_grouped_data, Association_mathod, min_support, association_metric, group_mapped_df, signature_ea, precision_underlimit):
    """Processes a single iteration of the confidence loop."""
    print(f"Processing for min_confidence: {min_confidence}")
    association_list_anomal = association_module(anomal_grouped_data, Association_mathod, min_support, min_confidence, association_metric)
    association_list_nomal = association_module(nomal_grouped_data, Association_mathod, min_support, min_confidence, association_metric)
    signatures = dict_list_difference(association_list_anomal, association_list_nomal)
    signature_result = signature_evaluate(group_mapped_df, signatures)
    signature_sets = under_limit(signature_result, signature_ea, precision_underlimit)
    current_recall = calculate_signatures(group_mapped_df, signature_sets)

    # Debug prints for this iteration (optional, can be removed for cleaner output)
    print(f"  min_confidence: {min_confidence}")
    print(f"    Anomal association rules: {len(association_list_anomal)}")
    print(f"    Normal association rules: {len(association_list_nomal)}")
    print(f"    Pure anomal signatures: {len(signatures)}")
    print(f"    Evaluated signatures: {len(signature_result)}")
    print(f"    Filtered signatures: {len(signature_sets) if signature_sets else 0}")
    print(f"    Current recall: {current_recall}")

    return min_confidence, current_recall, signature_sets

def main():
    # argparser
    # Create an instance that can receive argument values
    parser = argparse.ArgumentParser(description='Argparser')

    # Set the argument values to be input (default value can be set)
    parser.add_argument('--file_type', type=str, default="MiraiBotnet")   # data file type
    parser.add_argument('--file_number', type=int, default=1)   # Detach files
    parser.add_argument('--train_test', type=int, default=0)    # train = 0, test = 1
    parser.add_argument('--heterogeneous', type=str, default="Normalized")   # Heterogeneous(Embedding) Methods
    parser.add_argument('--clustering', type=str, default="kmeans")   # Clustering Methods
    parser.add_argument('--eval_clustering_silhouette', type=str, default="n")
    parser.add_argument('--association', type=str, default="apriori")   # Association Rule
    parser.add_argument('--precision_underlimit', type=float, default=0.6)
    parser.add_argument('--signature_ea', type=int, default=15)
    parser.add_argument('--association_metric', type=str, default='confidence')

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
    precision_underlimit = args.precision_underlimit
    signature_ea = args.signature_ea
    association_metric = args.association_metric

    total_start_time = time.time()  # Start All Time
    timing_info = {}  # For step-by-step time recording


    # 1. Data loading
    start = time.time()

    file_path, file_number = file_path_line_association(file_type, file_number)
    # cut_type = str(input("Enter the data cut type: "))
    cut_type = 'all'
    data = file_cut(file_type, file_path, cut_type)

    timing_info['1_load_data'] = time.time() - start


    # 2. Handling judgments of Anomal or Nomal
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

    timing_info['2_anomal_judgment'] = time.time() - start


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

    # Save mapped_info_df for Validate_Signature.py
    # Ensure the directory exists
    mapped_info_save_path_dir = f"../Dataset_Paral/signature/{file_type}/"
    if not os.path.exists(mapped_info_save_path_dir):
        os.makedirs(mapped_info_save_path_dir)
    mapped_info_save_path = f"{mapped_info_save_path_dir}{file_type}_{file_number}_mapped_info.csv"
    mapped_info_df.to_csv(mapped_info_save_path, index=False)
    print(f"Saved mapped_info_df to: {mapped_info_save_path}")

    group_mapped_df['label'] = data['label']

    # ===== Convert NSL-KDD string labels to numeric =====
    if file_type in ['NSL-KDD', 'NSL_KDD']:
        print("Converting NSL-KDD labels ('normal'->0, 'attack'->1)...")
        # Make sure the mapping matches the actual string values
        # It seems the label might be 'attack' based on user output, not 'anomal'
        label_map = {'normal': 0, 'attack': 1} 
        group_mapped_df['label'] = group_mapped_df['label'].map(label_map)
        # Verify conversion
        print("Label distribution after NSL-KDD conversion:")
        print(group_mapped_df['label'].value_counts())
    # =====================================================

    # ===== Check group_mapped_df before splitting =====
    print(f"Shape of group_mapped_df: {group_mapped_df.shape}")
    if 'label' in group_mapped_df.columns:
        print("Label distribution in group_mapped_df:")
        print(group_mapped_df['label'].value_counts())
    else:
        print("Warning: 'label' column not found in group_mapped_df before splitting.")
    # ====================================================


    # Information about how to set up association rule groups
    anomal_grouped_data = anomal_class_data(group_mapped_df)
    anomal_grouped_data = without_label(anomal_grouped_data)
    print("anomal_grouped_data: ", anomal_grouped_data)
    # anomal_grouped_data is DataFrame
    # fl: feature list; Same contents but not used because it's not inside a DF.

    # Make nomal row
    nomal_grouped_data = nomal_class_data(group_mapped_df)
    nomal_grouped_data = without_label(nomal_grouped_data)
    print("nomal_grouped_data: ", nomal_grouped_data)
    # nomal_grouped_data is DataFrame
    # flo: feature list; Same contents but not used because it's not inside a DF.

    timing_info['3_embedding'] = time.time() - start


    # 4. Set association statements (confidence ratios, etc.)
    start = time.time()


    min_support = 0.1

    # Use a lower min_support value for NSL-KDD
    if file_type in ['NSL-KDD', 'NSL_KDD']:
        # Restore to previously successful settings
        min_support_ratio_for_rare = 0.07
        min_distinct = 2
        print(f"NSL-KDD settings for remove_rare_columns: min_support_ratio={min_support_ratio_for_rare}, min_distinct={min_distinct}") # 값 확인용
    else:
        min_support_ratio_for_rare = 0.1   # Other dataset defaults
        min_distinct = 2 # Other dataset defaults

    best_confidence = 0.8    # Initialize the variables to change
    # Considering anomalies and nomals simultaneously

    confidence_values = np.arange(0.1, 1.0, 0.05)
    best_recall = 0

    print("min_support: ", min_support)
    print("Applying remove_rare_columns...")
    # Assuming you call utils.remove_rare_columns
    anomal_grouped_data = remove_rare_columns(anomal_grouped_data, min_support_ratio_for_rare, file_type, min_distinct_frequent_values=min_distinct)
    nomal_grouped_data = remove_rare_columns(nomal_grouped_data, min_support_ratio_for_rare, file_type, min_distinct_frequent_values=min_distinct)
    print("Finished remove_rare_columns.")
    print("Anomal data shape after pruning:", anomal_grouped_data.shape) # Result check
    print("Normal data shape after pruning:", nomal_grouped_data.shape) # Result check

    timing_info['4_association_setting'] = time.time() - start


    # Identify the signatures with the highest recall in user's situation
    # 5. Excute Association Rule, Manage related groups
    start = time.time()

    last_signature_sets = None

    print("Starting parallel processing for confidence values...")

    # Prepare arguments for the worker function
    # These are constant across all iterations for min_confidence
    static_args = (
        anomal_grouped_data,
        nomal_grouped_data,
        Association_mathod,
        min_support,
        association_metric,
        group_mapped_df,
        signature_ea,
        precision_underlimit
    )

    # Create a list of arguments for starmap: (min_confidence_value, *static_args)
    # confidence_values is defined earlier as np.arange(0.1, 1.0, 0.05)
    tasks = [(conf_val,) + static_args for conf_val in confidence_values]

    # Determine number of processes (e.g., number of CPU cores)
    # You might want to adjust this based on your system and other running processes
    num_processes = multiprocessing.cpu_count() 
    print(f"Using {num_processes} processes for parallel execution.")

    results = []
    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            # starmap blocks until all results are ready
            # Each item in 'results' will be (min_confidence, current_recall, signature_sets)
            results = pool.starmap(process_confidence_iteration, tasks)
    except Exception as e:
        print(f"An error occurred during parallel processing: {e}")
        # Fallback or error handling, e.g., run sequentially or re-raise
        # For now, just print and proceed, which might mean empty results
        # depending on where the error occurred.

    print("Parallel processing finished. Aggregating results...")
    
    # Process results to find the best one
    if results: # Check if results were successfully populated
        for res_min_confidence, res_current_recall, res_signature_sets in results:
            if res_current_recall > best_recall:
                best_recall = res_current_recall
                best_confidence = res_min_confidence
                last_signature_sets = res_signature_sets
    else:
        print("No results from parallel processing. Check for errors.")


    association_result = {
        'Verified_Signatures': last_signature_sets,
        'Recall': best_recall,
        'Best_confidence': best_confidence
    }
    print(association_result)

    save = csv_association(file_type, file_number, Association_mathod, association_result, association_metric, signature_ea)

    timing_info['5_excute_association'] = time.time() - start


    # Full time history
    total_end_time = time.time()
    timing_info['0_total_time'] = total_end_time - total_start_time

    # Save time information as a CSV
    time_save_csv_CS(file_type, file_number, Association_mathod, timing_info, best_confidence, min_support) # Added best_confidence and min_support for context in timing


    return association_result


if __name__ == '__main__':
    main()