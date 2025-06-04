import argparse
import csv
import ast
import os
from utils.class_row import anomal_class_data, nomal_class_data, without_label
from utils.remove_rare_columns import remove_rare_columns
from Dataset_Choose_Rule.association_data_choose import file_path_line_association


def evaluate_signatures(signature_data, data, attack_type_column):
    results = {}
    detailed_results = []
    for signature in signature_data:
        attack_type = signature['signature_name']['Signature_dict'].get(attack_type_column, 'Unknown')
        if attack_type not in results:
            results[attack_type] = {'true_positive': 0, 'false_positive': 0, 'false_negative': 0}
        
        # Calculate precision and recall by comparing attack types in the dataset to those in the signature
        for row in data:
            actual_attack_type = row[attack_type_column]  # Use the attack type column name in the dataset
            if attack_type == actual_attack_type:
                results[attack_type]['true_positive'] += 1
            else:
                results[attack_type]['false_positive'] += 1
                results[attack_type]['false_negative'] += 1

        # Add detailed results for each signature
        detailed_results.append({
            'Signature': signature['signature_name']['Signature_dict'],
            'Identified Attack': attack_type,
            'True Positives': results[attack_type]['true_positive'],
            'False Positives': results[attack_type]['false_positive'],
            'False Negatives': results[attack_type]['false_negative']
        })

    # Calculate precision and recall
    for attack_type, metrics in results.items():
        tp = metrics['true_positive']
        fp = metrics['false_positive']
        fn = metrics['false_negative']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        results[attack_type]['precision'] = precision
        results[attack_type]['recall'] = recall

    return results, detailed_results

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
        writer.writerow(['Signature', 'Identified Attack', 'True Positives', 'False Positives', 'False Negatives'])
        for result in results:
            writer.writerow([result['Signature'], result['Identified Attack'], result['True Positives'], result['False Positives'], result['False Negatives']])


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

    # Determine the correct attack type column based on file type
    if file_type in ['DARPA98', 'DARPA']:
        attack_type_column = 'Class'
    elif file_type in ['CICModbus23', 'CICModbus']:
        attack_type_column = 'Attack'
    else:
        attack_type_column = 'AttackType'

    # Group mapping
    group_mapped_df = anomal_class_data(data)
    group_mapped_df = without_label(group_mapped_df)

    # Remove rare columns
    min_support_ratio_for_rare = 0.01 if file_type in ['NSL-KDD', 'NSL_KDD'] else 0.1
    min_distinct = 1 if file_type in ['NSL-KDD', 'NSL_KDD'] else 2
    group_mapped_df = remove_rare_columns(group_mapped_df, min_support_ratio_for_rare, file_type, min_distinct_frequent_values=min_distinct)

    with open(signature_csv_path, mode='r', newline='') as sig_file:
        sig_reader = csv.DictReader(sig_file)
        for sig_row in sig_reader:
            signature_data = ast.literal_eval(sig_row['Verified_Signatures'])
            results, detailed_results = evaluate_signatures(signature_data, group_mapped_df, attack_type_column)
            write_results_to_csv(results, output_csv_path)
            write_detailed_results_to_csv(detailed_results, detailed_output_csv_path)

if __name__ == '__main__':
    main()


