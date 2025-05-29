# for save Clustering, Association row to csv

import os
import pandas as pd


# Functions for creating folders if they don't exist
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def csv_compare_clustering(file_type, clusterint_method, file_number, data, GMM_type=None, optimal_cni_threshold=None):
    row_compare_df = data[['cluster', 'adjusted_cluster', 'label']].copy() # Use .copy() to avoid SettingWithCopyWarning
    if optimal_cni_threshold is not None:
        row_compare_df['Optimal_CNI_Threshold'] = optimal_cni_threshold
    
    save_path = f"../Dataset_Paral/save_dataset/{file_type}/"
    ensure_directory_exists(save_path)  # Verify and create the folder
    
    if clusterint_method == "GMM":
        file_path = f"{save_path}{file_type}_{clusterint_method}_{file_number}_clustering_Compare_gmm{GMM_type}.csv"
    else:
        file_path = f"{save_path}{file_type}_{clusterint_method}_{file_number}_clustering_Compare.csv"
    row_compare_df.to_csv(file_path, index=False)
    print(f"[INFO SaveCSV] Clustering comparison data saved to: {os.path.abspath(file_path)}")
    
    return row_compare_df

def csv_compare_matrix_clustering(file_type, file_number, clusterint_method, metrics_original, metrics_adjusted, GMM_type, optimal_cni_threshold=None):
    # Ensure metrics_original and metrics_adjusted are dictionaries before creating DataFrame
    # Create copies to modify them before DataFrame creation if needed
    metrics_original_with_thresh = metrics_original.copy() if metrics_original else {}
    metrics_adjusted_with_thresh = metrics_adjusted.copy() if metrics_adjusted else {}

    if optimal_cni_threshold is not None:
        metrics_original_with_thresh['Optimal_CNI_Threshold'] = optimal_cni_threshold
        metrics_adjusted_with_thresh['Optimal_CNI_Threshold'] = optimal_cni_threshold

    metrics_df = pd.DataFrame([metrics_original_with_thresh, metrics_adjusted_with_thresh], index=["Original", "Adjusted"])
    
    save_path = f"../Dataset_Paral/save_dataset/{file_type}/"
    ensure_directory_exists(save_path)  # Verify and create the folder
    
    if clusterint_method == "GMM":
        file_path = f"{save_path}{file_type}_{clusterint_method}_{file_number}_clustering_Compare_Metrics_gmm{GMM_type}.csv"
    else:
        file_path = f"{save_path}{file_type}_{clusterint_method}_{file_number}_clustering_Compare_Metrics.csv"
    metrics_df.to_csv(file_path, index=True)
    print(f"[INFO SaveCSV] Clustering metrics data saved to: {os.path.abspath(file_path)}")
    
    return metrics_df


def csv_association(file_type, file_number, association_rule, association_result, association_metric, signature_ea):
    df = pd.DataFrame([association_result])

    save_path = f"../Dataset_Paral/signature/{file_type}/"
    ensure_directory_exists(save_path)  # Verify and create the folder

    file_path = f"{save_path}{file_type}_{association_rule}_{file_number}_{association_metric}_signature_train_ea{signature_ea}.csv"

    df.to_csv(file_path, index=False)
    