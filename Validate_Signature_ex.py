import pandas as pd
import numpy as np
import json
import argparse
import logging
import os
import sys
import ast
from datetime import datetime

# Set the project root path based on the path of the current script (adjust as needed)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add the project root to the front of sys.path (avoid duplicate additions if it already exists)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Import project-specific modules --- 
# Now all local modules are imported relative to the project root.
from utils.time_transfer import time_scalar_transfer, convert_cic_time_to_numeric_scalars
from utils.save_data_io import save_to_json, load_from_json

from Rebuild_Method.FalsePositive_Check import (
    apply_signatures_to_dataset,
    calculate_fp_scores,
    evaluate_false_positives,
    summarize_fp_results
)

from Validation.generation_fp import generate_fake_fp_signatures
from Validation.Validation_util import map_data_using_category_mapping
    
from Dataset_Choose_Rule.choose_amount_dataset import file_cut_GEN
from Dataset_Choose_Rule.Raw_Dataset_infos import Dataset_infos
from Dataset_Choose_Rule.association_data_choose import file_path_line_association


# Logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set default path (may need to be adjusted for your environment)
# PROJECT_ROOT is ~/asic/ASIC_0605/ (example)
# We need to go one level up from PROJECT_ROOT to get to ~/asic/
GRANDPARENT_DIR = os.path.dirname(PROJECT_ROOT) # This should give ~/asic/
BASE_SIGNATURE_PATH = os.path.join(GRANDPARENT_DIR, "Dataset_Paral", "signature")
BASE_MAPPING_PATH = os.path.join(GRANDPARENT_DIR, "Dataset_Paral", "signature") # Corrected path
BASE_DATA_PATH = os.path.join(GRANDPARENT_DIR, "Dataset", "load_dataset") # Corrected path for datasets


def load_signatures(file_type, config_name_prefix):
    sig_file_name_json = f"{file_type}_{config_name_prefix}.json"
    sig_file_path_json = os.path.join(BASE_SIGNATURE_PATH, file_type, sig_file_name_json)
    
    signatures = None
    
    logger.info(f"Attempting to load signatures from JSON: {sig_file_path_json}")
    if os.path.exists(sig_file_path_json):
        signatures = load_from_json(sig_file_path_json)
        if signatures is not None:
            logger.info(f"Successfully loaded signatures from JSON: {sig_file_path_json}")
        else:
            logger.warning(f"Failed to load signatures from existing JSON file: {sig_file_path_json}")
    else:
        logger.info(f"JSON signature file not found: {sig_file_path_json}")

    if signatures is None:
        sig_file_name_csv = f"{file_type}_{config_name_prefix}.csv"
        sig_file_path_csv = os.path.join(BASE_SIGNATURE_PATH, file_type, sig_file_name_csv)
        logger.info(f"Attempting to load signatures from CSV: {sig_file_path_csv}")
        if os.path.exists(sig_file_path_csv):
            try:
                df = pd.read_csv(sig_file_path_csv)
                if not df.empty and 'Verified_Signatures' in df.columns:
                    # Assuming the relevant data is in the first row
                    verified_signatures_str = df['Verified_Signatures'].iloc[0]
                    if pd.isna(verified_signatures_str):
                        logger.error(f"'Verified_Signatures' content is NaN in CSV: {sig_file_path_csv}")
                    else:
                        try:
                            parsed_list = ast.literal_eval(verified_signatures_str)
                            if isinstance(parsed_list, list):
                                extracted_signatures = []
                                for idx, item in enumerate(parsed_list):
                                    if isinstance(item, dict) and \
                                       'signature_name' in item and \
                                       isinstance(item['signature_name'], dict) and \
                                       'Signature_dict' in item['signature_name']:
                                        
                                        rule_dict = item['signature_name']['Signature_dict']
                                        # Use a generated ID for now, or look for a specific field if available
                                        # For example, if item['signature_name'] has an 'id' or 'name' field
                                        sig_id = item['signature_name'].get('name', f"csv_sig_{idx}") 
                                        extracted_signatures.append({'id': sig_id, 'rule_dict': rule_dict})
                                    else:
                                        logger.warning(f"Skipping item from CSV due to unexpected structure: {item}")
                                signatures = extracted_signatures
                                logger.info(f"Successfully extracted {len(signatures)} signatures from CSV: {sig_file_path_csv}")
                            else:
                                logger.error(f"Parsed 'Verified_Signatures' from CSV is not a list: {sig_file_path_csv}")
                        except (ValueError, SyntaxError) as e:
                            logger.error(f"Error parsing 'Verified_Signatures' string from CSV {sig_file_path_csv}: {e}")
                else:
                    logger.error(f"CSV file {sig_file_path_csv} is empty or 'Verified_Signatures' column is missing.")
            except Exception as e:
                logger.error(f"Error reading CSV file {sig_file_path_csv}: {e}")
        else:
            logger.info(f"CSV signature file not found: {sig_file_path_csv}")

    if signatures is None:
        logger.error(f"Failed to load signatures from both JSON and CSV for prefix: {config_name_prefix}")
        return None

    # Common processing for signatures loaded from either JSON or CSV
    if not isinstance(signatures, list):
        if isinstance(signatures, dict) and all(isinstance(s_val, dict) for s_val in signatures.values()):
            logger.info("Attempting to convert signatures from dict of dicts to list of dicts...")
            signatures_list = []
            for s_id, s_content in signatures.items():
                s_content['id'] = s_id # Assign or overwrite 'id'
                # Ensure 'rule_dict' exists, if s_content is supposed to be the rule itself
                if 'rule_dict' not in s_content: 
                     # This case implies the dict itself is the rule_dict
                     # This might happen if the JSON format was { "id1": {"rule_key": "val"}, ...}
                     # And we expect {"id": "id1", "rule_dict": {"rule_key": "val"}}
                     # However, the original code appends s_content directly, assuming it's already structured
                     # correctly or that rule_dict is a primary key within it.
                     # For safety, if 'rule_dict' is not in s_content, we might need to decide if s_content *is* the rule_dict
                     # This part of logic is from the original code for JSON dict-of-dicts.
                     # If CSV always produces list of {'id': X, 'rule_dict': Y}, this dict-of-dicts branch is only for JSON.
                     pass # Assuming s_content structure is what's expected or 'rule_dict' is already a key within.
                signatures_list.append(s_content)
            signatures = signatures_list
            logger.info(f"Successfully converted signatures to list format. Count: {len(signatures)}")
        else:
            logger.error("Signatures are not in the expected list format and could not be converted.")
            return None
            
    # Final validation
    if not signatures: # Check if signatures list is empty after all processing
        logger.error("No signatures were successfully loaded or extracted.")
        return None

    if not all(isinstance(s, dict) and 'id' in s and 'rule_dict' in s for s in signatures):
        logger.error("Signatures are not in the expected final format (list of dicts with 'id' and 'rule_dict').")
        logger.debug(f"Problematic signatures data: {signatures[:5]}") # Log first few items for debugging
        return None
        
    logger.info(f"Final loaded signature count: {len(signatures)}")
    return signatures


def load_category_mapping(file_type, file_number):
    map_file_name = f"{file_type}_{file_number}_mapped_info.csv"
    map_file_path = os.path.join(BASE_MAPPING_PATH, file_type, map_file_name)
    
    logger.info(f"Attempting to load category mapping from CSV: {map_file_path}")

    if not os.path.exists(map_file_path):
        logger.error(f"CSV Mapping file not found: {map_file_path}")
        return None

    '''
    mapping_data = load_from_json(map_file_path)
    
    if mapping_data and 'interval' in mapping_data and isinstance(mapping_data['interval'], dict):
        try:
            # Ensure keys of inner dicts are preserved if they are numeric (e.g. for Date_scalar original string keys)
            # pd.DataFrame will try to infer dtype for columns. If original rule strings are keys, that's fine.
            mapping_data['interval'] = pd.DataFrame(mapping_data['interval'])
            logger.info("Converted 'interval' mapping from dict to DataFrame.")
        except Exception as e:
            logger.error(f"Failed to convert 'interval' mapping to DataFrame: {e}")
    return mapping_data
    '''

    try:
        df_mapping = pd.read_csv(map_file_path)
        if df_mapping.empty:
            logger.error(f"Mapping CSV file is empty: {map_file_path}")
            return None
        
        category_mapping = {
            'interval': {},
            # Add other types like 'categorical', 'binary' if they are also expected from CSV
            # For now, focusing on 'interval' based on the provided CSV structure
        }
        
        for column in df_mapping.columns:
            # Collect non-NaN mapping strings for the current column
            # The Series constructor will automatically handle NaN values by not including them if not told otherwise
            # and will keep the original index if needed, though here we just want the values.
            column_mappings = df_mapping[column].dropna().astype(str).tolist()
            if column_mappings: # Only add if there are actual mapping entries
                category_mapping['interval'][column] = pd.Series(column_mappings)
            else:
                logger.warning(f"No valid mapping entries found for column '{column}' in {map_file_path}")

        if not category_mapping['interval']:
            logger.error(f"No interval mappings were extracted from {map_file_path}")
            return None # Or return an empty structure based on downstream requirements

        # Convert the dictionary of Series to a DataFrame for 'interval'
        # This matches the structure expected by the rest of the original code if it was loading from JSON dict
        category_mapping['interval'] = pd.DataFrame(category_mapping['interval'])
        logger.info(f"Successfully loaded and processed category mapping from {map_file_path}")
        return category_mapping
        
    except pd.errors.EmptyDataError:
        logger.error(f"Mapping CSV file is empty or unreadable: {map_file_path}")
        return None
    except Exception as e:
        logger.error(f"Error processing mapping CSV file {map_file_path}: {e}")
        return None


def load_dataset(file_type):
    # Get the base dataset path from the imported function
    # file_path_line_association returns (file_path, file_number), we only need file_path
    base_dataset_path, _ = file_path_line_association(file_type)

    # The path returned by file_path_line_association is relative to its own location.
    # We need to make it absolute or relative to the current PROJECT_ROOT or GRANDPARENT_DIR.
    # Assuming paths in association_data_choose.py like "../Dataset/load_dataset/..." 
    # are intended to be relative from a script in a subfolder of GRANDPARENT_DIR (e.g. PROJECT_ROOT)
    # So, os.path.join(GRANDPARENT_DIR, path_from_assoc_choose.lstrip('../')) might work.
    # Let's check how GRANDPARENT_DIR and BASE_DATA_PATH are defined again.
    # GRANDPARENT_DIR = os.path.dirname(PROJECT_ROOT) # e.g., /home/work/asic
    # BASE_DATA_PATH = os.path.join(GRANDPARENT_DIR, "Dataset", "load_dataset") # e.g., /home/work/asic/Dataset/load_dataset
    
    # file_path_line_association returns paths like "../Dataset/load_dataset/CICModbus23/CICModbus23_total.csv"
    # If Validate_Signature_ex.py is at /home/work/asic/ASIC_0605/Validate_Signature_ex.py
    # Then os.path.abspath(os.path.join(os.path.dirname(__file__), base_dataset_path)) should resolve correctly.
    # Or, more simply, ensure base_dataset_path is correctly relative to GRANDPARENT_DIR if it starts with "../"
    
    if base_dataset_path.startswith("../"):
        # Resolve path relative to the directory of Validate_Signature_ex.py (PROJECT_ROOT)
        # then go up one level as implied by ".." in the path string from association_data_choose
        # and then follow the rest of the path. GRANDPARENT_DIR is effectively PROJECT_ROOT/..
        dataset_path = os.path.abspath(os.path.join(PROJECT_ROOT, base_dataset_path))
    elif base_dataset_path.startswith("~"):
        dataset_path = os.path.expanduser(base_dataset_path)
    else: 
        # If it's already an absolute path or a path type not starting with ../ or ~
        # This case might need more specific handling if other path types occur
        dataset_path = base_dataset_path

    # Original filename logic for logging, though the actual file loaded is dataset_path
    # This might be confusing if dataset_name_suffix was important for distinguishing versions/types.
    # For now, we use the direct path. The concept of dataset_name_suffix might need re-evaluation
    # if different versions of the *same base file* are needed.
    dataset_filename_for_log = os.path.basename(dataset_path) 

    logger.info(f"Loading dataset from: {dataset_path} (determined by association_data_choose.py)")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        return None
    try:
        dataset_info = Dataset_infos.get(file_type, {})
        header_row = 0 if dataset_info.get('has_header', True) else None
        df = file_cut_GEN(file_type, dataset_path, 'all', header=header_row)
    except Exception as e:
        logger.error(f"Error loading data using file_cut_GEN for {dataset_path}: {e}")
        try:
            # Fallback to simple pd.read_csv if file_cut_GEN fails
            df = pd.read_csv(dataset_path, header=header_row) # Pass header_row here too
            logger.info(f"Successfully loaded {dataset_path} using pd.read_csv fallback.")
        except Exception as e_pd:
            logger.error(f"Error loading data using pd.read_csv for {dataset_path}: {e_pd}")
            return None
    logger.info(f"Loaded dataset {dataset_filename_for_log}, shape: {df.shape if df is not None else 'Error'}")
    return df

def main(args):
    logger.info(f"Starting validation process for {args.file_type}, number {args.file_number}, config: {args.config_name_prefix}")

    signatures = load_signatures(args.file_type, args.config_name_prefix)
    if signatures is None: return

    category_mapping = load_category_mapping(args.file_type, args.file_number)
    if category_mapping is None: return

    # Pass only file_type to load_dataset as it now determines path internally
    raw_attack_free_df = load_dataset(args.file_type) 
    if raw_attack_free_df is None: return
    raw_test_df = load_dataset(args.file_type) # Assuming test data also comes from the same base file
    if raw_test_df is None: return

    logger.info("Applying time_scalar_transfer to datasets...")
    processed_attack_free_df = time_scalar_transfer(raw_attack_free_df.copy(), args.file_type)
    processed_test_df = time_scalar_transfer(raw_test_df.copy(), args.file_type)

    # Apply CICModbus specific numeric time scalar conversion if applicable
    if args.file_type in ['CICModbus23', 'CICModbus']:
        logger.info(f"Applying CICModbus specific numeric time scalar conversion for {args.file_type}...")
        if processed_attack_free_df is not None:
            processed_attack_free_df = convert_cic_time_to_numeric_scalars(processed_attack_free_df)
        if processed_test_df is not None:
            processed_test_df = convert_cic_time_to_numeric_scalars(processed_test_df)
    
    # --- DEBUG LOGGING START ---
    logger.info("--- Debugging: Category Mapping for Date/Time Scalars ---")
    if category_mapping and 'interval' in category_mapping and isinstance(category_mapping['interval'], pd.DataFrame):
        interval_df = category_mapping['interval']
        if 'Date_scalar' in interval_df.columns:
            logger.info(f"Mapping rules for Date_scalar:\n{interval_df['Date_scalar'].dropna().unique().tolist()}")
        else:
            logger.info("No Date_scalar mapping rules found in category_mapping['interval'].")
        if 'StartTime_scalar' in interval_df.columns:
            logger.info(f"Mapping rules for StartTime_scalar:\n{interval_df['StartTime_scalar'].dropna().unique().tolist()}")
        else:
            logger.info("No StartTime_scalar mapping rules found in category_mapping['interval'].")
    else:
        logger.info("category_mapping['interval'] is not a DataFrame or not found.")

    logger.info("--- Debugging: Processed DataFrame Date/Time Scalar Values (sample) ---")
    if not processed_attack_free_df.empty:
        if 'Date_scalar' in processed_attack_free_df.columns:
            logger.info(f"Sample Date_scalar values from processed_attack_free_df (first 5 unique non-NaN):\n{processed_attack_free_df['Date_scalar'].dropna().unique()[:5]}")
        else:
            logger.info("Date_scalar column not found in processed_attack_free_df.")
        if 'StartTime_scalar' in processed_attack_free_df.columns:
            logger.info(f"Sample StartTime_scalar values from processed_attack_free_df (first 5 unique non-NaN):\n{processed_attack_free_df['StartTime_scalar'].dropna().unique()[:5]}")
        else:
            logger.info("StartTime_scalar column not found in processed_attack_free_df.")
    else:
        logger.info("processed_attack_free_df is empty before mapping.")
    # --- DEBUG LOGGING END ---

    logger.info("Mapping datasets using category_mapping...")
    mapped_attack_free_df = map_data_using_category_mapping(processed_attack_free_df, category_mapping, file_type=args.file_type)
    mapped_test_df = map_data_using_category_mapping(processed_test_df, category_mapping, file_type=args.file_type)

    if mapped_attack_free_df.empty or mapped_test_df.empty:
        logger.error("One or both datasets are empty after mapping. Exiting.")
        return

    logger.info("\n--- Evaluating False Positives for Loaded Signatures ---")
    alerts_df = apply_signatures_to_dataset(mapped_test_df, signatures)

    if alerts_df.empty:
        logger.info("No alerts generated from the test data with loaded signatures.")
    else:
        logger.info(f"Generated {len(alerts_df)} alerts from test data.")
        fp_results_df = calculate_fp_scores(alerts_df, mapped_attack_free_df, file_type=args.file_type,
                                            t0_nra=args.t0_nra, n0_nra=args.n0_nra,
                                            lambda_haf=args.lambda_haf, lambda_ufp=args.lambda_ufp)
        signatures_map = {sig['id']: sig for sig in signatures}
        evaluated_fp_df = evaluate_false_positives(
            fp_results_df, signatures_map, known_fp_sig_dicts=[], 
            attack_free_df=mapped_attack_free_df, file_type=args.file_type,
            combine_method=args.combine_method, belief_threshold=args.belief_threshold)
        summary = summarize_fp_results(evaluated_fp_df)
        logger.info(f"Summary of FP evaluation for loaded signatures:\n{summary}")

    logger.info("\n--- Generating and Evaluating Fake FP Signatures ---")
    # Unpack the tuple returned by generate_fake_fp_signatures
    # We are interested in the first element (list of signature dicts) 
    # and potentially the count of actually generated signatures (last element).
    returned_values = generate_fake_fp_signatures(
        args.file_type, args.file_number, category_mapping, [], 
        association_method=args.association, association_metric=args.association_metric,
        num_fake_signatures=args.num_fake_signatures, min_support=args.fake_min_support)
    
    fake_fp_rules_list = returned_values[0] # The list of signature dicts
    # actual_num_generated_by_func = returned_values[5] # If needed for logging

    if not fake_fp_rules_list: # Check the list itself
        logger.info("No fake FP signatures were generated (list is empty).")
    else:
        logger.info(f"Generated {len(fake_fp_rules_list)} fake FP signatures.")
        for i, rule in enumerate(fake_fp_rules_list):
            # Now, 'rule' should be a dictionary from the list
            if isinstance(rule, dict):
                if 'id' not in rule: rule['id'] = f"fake_fp_sig_{i+1}"
                if 'name' not in rule: rule['name'] = f"Fake FP Signature {i+1}"
            else:
                logger.warning(f"Skipping item in fake_fp_rules_list as it is not a dict: {rule}")
                continue # Skip to the next item

        # Use fake_fp_rules_list for apply_signatures_to_dataset
        fake_alerts_df = apply_signatures_to_dataset(mapped_attack_free_df, fake_fp_rules_list)

        if fake_alerts_df.empty:
            logger.info("No alerts generated from attack-free data with fake FP signatures.")
        else:
            logger.info(f"Generated {len(fake_alerts_df)} alerts from attack-free data using fake FP signatures.")
            fake_fp_scores_df = calculate_fp_scores(
                fake_alerts_df, mapped_attack_free_df, file_type=args.file_type,
                t0_nra=args.t0_nra, n0_nra=args.n0_nra,
                lambda_haf=args.lambda_haf, lambda_ufp=args.lambda_ufp)
            fake_signatures_map = {sig['id']: sig for sig in fake_fp_rules_list}
            evaluated_fake_fp_df = evaluate_false_positives(
                fake_fp_scores_df, fake_signatures_map, known_fp_sig_dicts=[],
                attack_free_df=mapped_attack_free_df, file_type=args.file_type,
                combine_method=args.combine_method, belief_threshold=args.belief_threshold)
            fake_summary = summarize_fp_results(evaluated_fake_fp_df)
            logger.info(f"Summary of FP evaluation for FAKE signatures:\n{fake_summary}")

    logger.info("Validation process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signature Validation and FP Evaluation Script")
    parser.add_argument("--file_type", type=str, default='CICModbus23', required=True, help="Type of the dataset (e.g., CICModbus23, DARPA98)")
    parser.add_argument("--file_number", type=str, default='1', required=True, help="Identifier for the dataset instance/config (e.g., '1', 'RARM_1')")
    parser.add_argument("--config_name_prefix", type=str, default='RARM_1_confidence_signature_train_ea15', required=True, help="Prefix for signature and mapping config files (e.g., RARM_1_confidence_signature_train_ea15)")
    parser.add_argument("--attack_free_suffix", type=str, default="Normal_train_ea15", help="Suffix for the attack-free (normal) dataset filename")
    parser.add_argument("--test_data_suffix", type=str, default="Test_All_data", help="Suffix for the test dataset filename")
    parser.add_argument("--t0_nra", type=int, default=60, help="Time window for NRA calculation")
    parser.add_argument("--n0_nra", type=int, default=20, help="Normalization factor for NRA")
    parser.add_argument("--lambda_haf", type=float, default=100.0, help="Lambda for HAF calculation")
    parser.add_argument("--lambda_ufp", type=float, default=10.0, help="Lambda for UFP calculation")
    parser.add_argument("--combine_method", type=str, default='max', choices=['max', 'avg', 'weighted'], help="Method to combine FP scores")
    parser.add_argument("--belief_threshold", type=float, default=0.5, help="Belief threshold for classifying as FP")
    parser.add_argument("--num_fake_signatures", type=int, default=5, help="Number of fake FP signatures to generate")
    parser.add_argument("--association", type=str, default='apriori', help="Association rule method for fake sigs (e.g. apriori, fpgrowth)")
    parser.add_argument("--association_metric", type=str, default='confidence', help="Association rule metric for fake sigs")
    parser.add_argument("--fake_min_support", type=float, default=0.2, help="Minimum support for generating fake FP signatures")
    cli_args = parser.parse_args()
    main(cli_args) 