import csv
import os
from datetime import datetime

def time_save_csv_VL(file_type, file_number, clustering_algorithm, timing_info):
    """
    Save timing information for each step and the total execution time to a CSV file.

    Parameters:
    - file_type (str): Dataset type (e.g., MiraiBotnet)
    - file_number (int): File index or part number
    - clustering_algorithm (str): Name of clustering algorithm used
    - timing_info (dict): Dictionary containing time taken per step
    - save_dir (str): Directory to save timing CSVs
    """

    save_dir = f"../Dataset_Paral/time_log/virtual_labeling/{file_type}"

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Filename format: [filetype]_[filenumber]_[clustering]_[timestamp].csv
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{file_type}_{file_number}_{clustering_algorithm}_{timestamp}.csv"
    filepath = os.path.join(save_dir, filename)

    # Write CSV file
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Step', 'Time_Seconds'])
        for step, duration in timing_info.items():
            writer.writerow([step, round(duration, 4)])

    print(f"\n Timing log saved to: {filepath}")

    return


def time_save_csv_CS(file_type, file_number, Association_mathod, timing_info, best_confidence=None, min_support=None):
    """
    Save timing information for each step, total execution time, best_confidence, and min_support to a CSV file.
    """

    save_dir = f"../Dataset_Paral/time_log/condition_assocation/{file_type}"

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Filename format: [filetype]_[filenumber]_[clustering]_[timestamp].csv
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{file_type}_{file_number}_{Association_mathod}_{timestamp}.csv"
    filepath = os.path.join(save_dir, filename)

    # Write CSV file
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Step', 'Time_Seconds'])
        for step, duration in timing_info.items():
            writer.writerow([step, round(duration, 4)])
        
        # Add best_confidence and min_support if available
        if best_confidence is not None:
            writer.writerow(['Best_Confidence', best_confidence])
        if min_support is not None:
            writer.writerow(['Min_Support', min_support])

    print(f"\n Timing log saved to: {filepath}")

    return


# Save Validation signature time
def time_save_csv_VS(file_type, file_number, Association_mathod, timing_info):
    save_dir = f"../Dataset_Paral/time_log/validation_signature/{file_type}"

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Filename format: [filetype]_[filenumber]_[clustering]_[timestamp].csv
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{file_type}_{file_number}_{Association_mathod}_{timestamp}.csv"
    filepath = os.path.join(save_dir, filename)

    # Write CSV file
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Step', 'Time_Seconds'])
        for step, duration in timing_info.items():
            writer.writerow([step, round(duration, 4)])

    print(f"\n Timing log saved to: {filepath}")

    return

