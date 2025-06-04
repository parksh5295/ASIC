import argparse
import csv
import ast
import os
from utils.class_row import anomal_class_data, nomal_class_data, without_label
from utils.remove_rare_columns import remove_rare_columns


def evaluate_signatures(signature_data, data):
    results = {}
    for signature in signature_data:
        attack_type = signature['signature_name']['Signature_dict'].get('AttackType', 'Unknown')
        if attack_type not in results:
            results[attack_type] = {'precision': 0, 'recall': 0, 'count': 0}
        
        # Calculate precision and recall by comparing attack types in the dataset to those in the signature
        for row in data:
            if row['AttackType'] == attack_type:  # Use the attack type column name in the dataset
                results[attack_type]['precision'] += 1  # Need to replace with real logic
                results[attack_type]['recall'] += 1  # Need to replace with real logic
            results[attack_type]['count'] += 1
    return results

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

def main():
    parser = argparse.ArgumentParser(description='Evaluate attack identification using signatures.')
    parser.add_argument('--file_type', type=str, required=True, help='Type of the dataset file')
    parser.add_argument('--file_number', type=int, required=True, help='File number for dataset')
    parser.add_argument('--association', type=str, required=True, help='Association method to use')
    args = parser.parse_args()

    file_type = args.file_type
    file_number = args.file_number
    association_method = args.association

    signature_csv_path = f'Dataset_Paral/signature/{file_type}/{file_type}_RARM_{file_number}_confidence_signature_train_ea15.csv'
    data_csv_path = f'Dataset_Paral/data/{file_type}/{file_type}_{file_number}.csv'
    output_csv_path = f'Dataset_Paral/signature/{file_type}/{file_type}_attack_identification_results.csv'

    # 데이터 로딩 및 전처리
    with open(data_csv_path, mode='r', newline='') as data_file:
        data_reader = csv.DictReader(data_file)
        data = [row for row in data_reader]

    # 그룹 매핑
    group_mapped_df = anomal_class_data(data)
    group_mapped_df = without_label(group_mapped_df)

    # 희귀 컬럼 제거
    min_support_ratio_for_rare = 0.01 if file_type in ['NSL-KDD', 'NSL_KDD'] else 0.1
    min_distinct = 1 if file_type in ['NSL-KDD', 'NSL_KDD'] else 2
    group_mapped_df = remove_rare_columns(group_mapped_df, min_support_ratio_for_rare, file_type, min_distinct_frequent_values=min_distinct)

    with open(signature_csv_path, mode='r', newline='') as sig_file:
        sig_reader = csv.DictReader(sig_file)
        for sig_row in sig_reader:
            signature_data = ast.literal_eval(sig_row['Verified_Signatures'])
            results = evaluate_signatures(signature_data, group_mapped_df)
            write_results_to_csv(results, output_csv_path)

if __name__ == '__main__':
    main()

# 예제 사용법
# main('path_to_signature_csv.csv', 'path_to_data_csv.csv', 'output_results.csv')
