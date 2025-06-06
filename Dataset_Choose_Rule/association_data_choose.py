# Association training, functions for selecting test datasets
# from Dataset_Choose_Rule.CICIDS2017_csv_selector import select_csv_file # Commented out
import pandas as pd # Added for potential future use if we directly read CSV here


# For train
def file_path_line_association(file_type, file_number=1): # file_number is not used, but insert to prevent errors from occurring
    if file_type == 'MiraiBotnet':
        file_path = "../Dataset/load_dataset/MiraiBotnet/output-dataset_ESSlab.csv"
    elif file_type in ['ARP', 'MitM', 'Kitsune']:
        file_path = "../Dataset/load_dataset/ARP_MitM_Kitsune/ARP_MitM_dataset.csv/ARP_MitM_dataset_final.csv"
    elif file_type in ['CICIDS2017', 'CICIDS']:
        # file_path, file_number =  select_csv_file() # Original line
        file_path = "~/asic/Dataset/load_dataset/CICIDS2017/CICIDS_all.csv" # Use unified CSV
        file_number = 1 # Default file_number, as select_csv_file used to return it
    elif file_type == 'netML' :
        file_path = "../Dataset/load_dataset/netML/netML_dataset.csv"
    elif file_type in ['NSL-KDD', 'NSL_KDD']:
        file_path = "../Dataset/load_dataset/NSL-KDD/train/train_payload.csv"
    elif file_type in ['DARPA', 'DARPA98']:
        file_path = "../Dataset/load_dataset/DARPA98/train/DARPA98.csv"
    elif file_type in ['CICModbus23', 'CICModbus']:
        file_path = "../Dataset/load_dataset/CICModbus23/CICModbus23_total.csv"
    elif file_type in ['IoTID20', 'IoTID']:
        file_path = "../Dataset/load_dataset/IoTID20/IoTID20.csv"
    else:
        print("No file information yet, please double-check the file type or provide new data!")
        file_path_line_association(file_type)
    return file_path, file_number


# For Test
def file_path_line_signatures(file_type, file_number=1): # file_number is not used, but insert to prevent errors from occurring
    if file_type == 'MiraiBotnet':
        file_path = "../Dataset/load_dataset/MiraiBotnet/output-dataset_ESSlab.csv"
    elif file_type in ['ARP', 'MitM', 'Kitsune']:
        file_path = "../Dataset/load_dataset/ARP_MitM_Kitsune/ARP_MitM_dataset.csv/ARP_MitM_dataset_final.csv"
    elif file_type in ['CICIDS2017', 'CICIDS']:
        # file_path, file_number =  select_csv_file() # Original line
        file_path = "~/asic/Dataset/load_dataset/CICIDS2017/CICIDS_all.csv" # Use unified CSV
        file_number = 1 # Default file_number
    elif file_type == 'netML' :
        file_path = "../Dataset/load_dataset/netML/netML_dataset.csv"
    elif file_type in ['NSL-KDD', 'NSL_KDD']:
        file_path = "../Dataset/load_dataset/NSL-KDD/test/test_payload.csv"
    elif file_type in ['DARPA', 'DARPA98']:
        file_path = "../Dataset/load_dataset/DARPA98/test/DARPA98.csv"
    elif file_type in ['CICModbus23', 'CICModbus']:
        file_path = "../Dataset/load_dataset/CICModbus23/CICModbus23_total.csv"
    elif file_type in ['IoTID20', 'IoTID']:
        file_path = "../Dataset/load_dataset/IoTID20/IoTID20.csv"
    else:
        print("No file information yet, please double-check the file type or provide new data!")
        file_path_line_association(file_type)
    return file_path, file_number