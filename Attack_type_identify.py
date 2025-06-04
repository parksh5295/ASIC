import argparse
import csv
import ast
import os
import pandas as pd
from utils.class_row import anomal_class_data, nomal_class_data, without_label
from utils.remove_rare_columns import remove_rare_columns
from Dataset_Choose_Rule.association_data_choose import file_path_line_association
from definition.Anomal_Judgment import anomal_judgment_label, anomal_judgment_nonlabel
import multiprocessing
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups


def evaluate_signature_task(signature, data, attack_type_column):
    attack_type = signature['signature_name']['Signature_dict'].get(attack_type_column, 'Unknown')
    return {
        'Signature': signature['signature_name']['Signature_dict'],
        'Identified Attack': attack_type
    }

def evaluate_signatures(signature_data, data, attack_type_column):
    # Use multiprocessing to evaluate signatures in parallel
    with multiprocessing.Pool() as pool:
        detailed_results = pool.starmap(evaluate_signature_task, [(signature, data, attack_type_column) for signature in signature_data])
    return detailed_results

def write_results_to_csv(results, output_file_path):
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Attack Type', 'Total Precision', 'Total Recall', 'Average Precision', 'Average Recall'])
        for attack_type, metrics in results.items():
            total_precision = metrics['precision']
            total_recall = metrics['recall']
            count = metrics['count']
            avg_precision = total_precision / count if count else 0
            avg_recall = total_recall / count if count else 0
            writer.writerow([attack_type, total_precision, total_recall, avg_precision, avg_recall])

def write_detailed_results_to_csv(results, output_file_path):
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Signature', 'Identified Attack'])
        for result in results:
            writer.writerow([
                result['Signature'],
                result['Identified Attack']
            ])


def main():
    parser = argparse.ArgumentParser(description='Evaluate attack identification using signatures.')
    parser.add_argument('--file_type', type=str, required=True, help='Type of the dataset file')
    parser.add_argument('--file_number', type=int, required=True, help='File number for dataset')
    parser.add_argument('--association', type=str, required=True, help='Association method to use')
    args = parser.parse_args()

    file_type = args.file_type
    file_number = args.file_number
    association_method = args.association

    # Use the file_path_line_association function to get the data file path
    data_csv_path, _ = file_path_line_association(file_type, file_number)
    # Use association_method to dynamically set the signature_csv_path
    signature_csv_path = f'Dataset_Paral/signature/{file_type}/{file_type}_{association_method}_{file_number}_confidence_signature_train_ea15.csv'
    output_csv_path = f'Dataset_Paral/signature/{file_type}/{file_type}_attack_identification_results.csv'
    detailed_output_csv_path = f'Dataset_Paral/signature/{file_type}/{file_type}_detailed_signature_results.csv'

    # Load and preprocess data
    with open(data_csv_path, mode='r', newline='') as data_file:
        data_reader = csv.DictReader(data_file)
        data = [row for row in data_reader]

    # Convert data to a DataFrame
    data_df = pd.DataFrame(data)

    # Assign labels based on file_type
    if file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
        data_df['label'], _ = anomal_judgment_nonlabel(file_type, data_df)
    elif file_type == 'netML':
        data_df['label'] = data_df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
    elif file_type == 'DARPA98':
        data_df['label'] = data_df['Class'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
    elif file_type in ['CICIDS2017', 'CICIDS']:
        if 'Label' in data_df.columns:
            data_df['label'] = data_df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
        else:
            data_df['label'] = 0
    elif file_type in ['CICModbus23', 'CICModbus']:
        data_df['label'] = data_df['Attack'].apply(lambda x: 0 if str(x).strip() == 'Baseline Replay: In position' else 1)
    elif file_type in ['IoTID20', 'IoTID']:
        data_df['label'] = data_df['Label'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
    else:
        data_df['label'] = anomal_judgment_label(data_df)

    # Determine the correct attack type column based on file type
    if file_type in ['DARPA98', 'DARPA']:
        attack_type_column = 'Class'
    elif file_type in ['CICModbus23', 'CICModbus']:
        attack_type_column = 'Attack'
    else:
        attack_type_column = 'AttackType'

    # Load mapping information
    mapping_file_path = f"{file_type}_{file_number}_mapped_info.csv"
    if not os.path.exists(mapping_file_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file_path}")
    mapping_info_df = pd.read_csv(mapping_file_path)

    # Extract mapping information
    category_mapping = {
        'interval': {},
        'categorical': pd.DataFrame(),
        'binary': pd.DataFrame()
    }

    for column in mapping_info_df.columns:
        column_mappings = []
        for value in mapping_info_df[column].dropna():
            if isinstance(value, str) and '=' in value:
                column_mappings.append(value)
        if column_mappings:
            category_mapping['interval'][column] = pd.Series(column_mappings)

    category_mapping['interval'] = pd.DataFrame(category_mapping['interval'])

    # Create data_list
    data_list = [pd.DataFrame(), pd.DataFrame()]

    # Perform mapping only for columns present in mapping_info_df
    columns_to_map = [col for col in data.columns if col in mapping_info_df.columns]
    data_to_map = data[columns_to_map]

    # Perform mapping
    group_mapped_df, _ = map_intervals_to_groups(data_to_map, category_mapping, data_list, regul='N')

    # Add unmapped columns back to the DataFrame
    unmapped_columns = [col for col in data.columns if col not in columns_to_map]
    group_mapped_df = pd.concat([group_mapped_df, data[unmapped_columns]], axis=1)

    # Add the label from the source data to group_mapped_df
    group_mapped_df['label'] = data['label']

    print("Mapping completed. Group mapped data:", group_mapped_df.head())

    # Remove rare columns
    min_support_ratio_for_rare = 0.01 if file_type in ['NSL-KDD', 'NSL_KDD'] else 0.1
    min_distinct = 1 if file_type in ['NSL-KDD', 'NSL_KDD'] else 2
    group_mapped_df = remove_rare_columns(group_mapped_df, min_support_ratio_for_rare, file_type, min_distinct_frequent_values=min_distinct)

    with open(signature_csv_path, mode='r', newline='') as sig_file:
        sig_reader = csv.DictReader(sig_file)
        for sig_row in sig_reader:
            signature_data = ast.literal_eval(sig_row['Verified_Signatures'])
            results = evaluate_signatures(signature_data, group_mapped_df, attack_type_column)
            write_results_to_csv(results, output_csv_path)
            write_detailed_results_to_csv(results, detailed_output_csv_path)

if __name__ == '__main__':
    main()


