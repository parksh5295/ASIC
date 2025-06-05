import pandas as pd
import numpy as np
import json
import argparse
import logging
import os
import sys
from datetime import datetime

# Set the project root path based on the path of the current script (adjust as needed)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add the project root to the front of sys.path (avoid duplicate additions if it already exists)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Import project-specific modules --- 
# Now all local modules are imported relative to the project root.
from utils.time_transfer import time_scalar_transfer
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


# Logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set default path (may need to be adjusted for your environment)
BASE_SIGNATURE_PATH = os.path.join(PROJECT_ROOT, "Dataset_Paral", "signature")
BASE_MAPPING_PATH = os.path.join(PROJECT_ROOT, "Dataset_Paral", "mapped_info")
BASE_DATA_PATH = os.path.join(PROJECT_ROOT, "Dataset_Paral", "train_test_data")


def load_signatures(file_type, config_name_prefix):
    sig_file_name = f"{file_type}_{config_name_prefix}.json"
    sig_file_path = os.path.join(BASE_SIGNATURE_PATH, file_type, sig_file_name)
    logger.info(f"Loading signatures from: {sig_file_path}")
    if not os.path.exists(sig_file_path):
        logger.error(f"Signature file not found: {sig_file_path}")
        return None
    signatures = load_from_json(sig_file_path)
    if not isinstance(signatures, list):
        if isinstance(signatures, dict) and all(isinstance(s_val, dict) for s_val in signatures.values()):
            logger.info("Attempting to convert signatures from dict of dicts to list of dicts...")
            signatures_list = []
            for s_id, s_content in signatures.items():
                s_content['id'] = s_id
                signatures_list.append(s_content)
            signatures = signatures_list
            logger.info(f"Successfully converted signatures to list format. Count: {len(signatures)}")
        else:
            logger.error("Signatures are not in the expected list format and could not be converted.")
            return None
    if not all(isinstance(s, dict) and 'id' in s and 'rule_dict' in s for s in signatures):
        logger.error("Signatures are not in the expected format (list of dicts with 'id' and 'rule_dict').")
        return None
    return signatures


def load_category_mapping(file_type, config_name_prefix):
    if "signature" in config_name_prefix:
        mapping_config_prefix = config_name_prefix.replace("_signature_", "_category_mapping_")
    else:
        mapping_config_prefix = config_name_prefix.replace("signature", "category_mapping") if "signature" in config_name_prefix else f"{config_name_prefix}_category_mapping"
    map_file_name = f"{file_type}_{mapping_config_prefix}.json"
    map_file_path = os.path.join(BASE_MAPPING_PATH, file_type, map_file_name)
    
    logger.info(f"Attempting to load category mapping from: {map_file_path}")

    if not os.path.exists(map_file_path):
        logger.error(f"JSON Mapping file not found: {map_file_path}")
        return None

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


def load_dataset(file_type, dataset_name_suffix):
    dataset_filename = f"{file_type}_{dataset_name_suffix}.csv"
    dataset_path = os.path.join(BASE_DATA_PATH, file_type, dataset_filename)
    logger.info(f"Loading dataset from: {dataset_path}")
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
            df = pd.read_csv(dataset_path)
            logger.info(f"Successfully loaded {dataset_path} using pd.read_csv fallback.")
        except Exception as e_pd:
            logger.error(f"Error loading data using pd.read_csv for {dataset_path}: {e_pd}")
            return None
    logger.info(f"Loaded dataset {dataset_filename}, shape: {df.shape}")
    return df

def main(args):
    logger.info(f"Starting validation process for {args.file_type}, number {args.file_number}, config: {args.config_name_prefix}")

    signatures = load_signatures(args.file_type, args.config_name_prefix)
    if signatures is None: return

    category_mapping = load_category_mapping(args.file_type, args.config_name_prefix)
    if category_mapping is None: return

    raw_attack_free_df = load_dataset(args.file_type, args.attack_free_suffix)
    if raw_attack_free_df is None: return
    raw_test_df = load_dataset(args.file_type, args.test_data_suffix)
    if raw_test_df is None: return

    logger.info("Applying time_scalar_transfer to datasets...")
    processed_attack_free_df = time_scalar_transfer(raw_attack_free_df.copy(), args.file_type)
    processed_test_df = time_scalar_transfer(raw_test_df.copy(), args.file_type)
    
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
    fake_fp_rules = generate_fake_fp_signatures(
        args.file_type, args.file_number, category_mapping, [], 
        association_method=args.assoc_method, association_metric=args.assoc_metric,
        num_fake_signatures=args.num_fake_signatures, min_support=args.fake_min_support)

    if not fake_fp_rules:
        logger.info("No fake FP signatures were generated.")
    else:
        logger.info(f"Generated {len(fake_fp_rules)} fake FP signatures.")
        for i, rule in enumerate(fake_fp_rules):
            if 'id' not in rule: rule['id'] = f"fake_fp_sig_{i+1}"
            if 'name' not in rule: rule['name'] = f"Fake FP Signature {i+1}"

        fake_alerts_df = apply_signatures_to_dataset(mapped_attack_free_df, fake_fp_rules)

        if fake_alerts_df.empty:
            logger.info("No alerts generated from attack-free data with fake FP signatures.")
        else:
            logger.info(f"Generated {len(fake_alerts_df)} alerts from attack-free data using fake FP signatures.")
            fake_fp_scores_df = calculate_fp_scores(
                fake_alerts_df, mapped_attack_free_df, file_type=args.file_type,
                t0_nra=args.t0_nra, n0_nra=args.n0_nra,
                lambda_haf=args.lambda_haf, lambda_ufp=args.lambda_ufp)
            fake_signatures_map = {sig['id']: sig for sig in fake_fp_rules}
            evaluated_fake_fp_df = evaluate_false_positives(
                fake_fp_scores_df, fake_signatures_map, known_fp_sig_dicts=[],
                attack_free_df=mapped_attack_free_df, file_type=args.file_type,
                combine_method=args.combine_method, belief_threshold=args.belief_threshold)
            fake_summary = summarize_fp_results(evaluated_fake_fp_df)
            logger.info(f"Summary of FP evaluation for FAKE signatures:\n{fake_summary}")

    logger.info("Validation process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signature Validation and FP Evaluation Script")
    parser.add_argument("--file_type", type=str, required=True, help="Type of the dataset (e.g., CICModbus23, DARPA98)")
    parser.add_argument("--file_number", type=str, required=True, help="Identifier for the dataset instance/config (e.g., '1', 'RARM_1')")
    parser.add_argument("--config_name_prefix", type=str, required=True, help="Prefix for signature and mapping config files (e.g., RARM_1_confidence_signature_train_ea15)")
    parser.add_argument("--attack_free_suffix", type=str, default="Normal_train_ea15", help="Suffix for the attack-free (normal) dataset filename")
    parser.add_argument("--test_data_suffix", type=str, default="Test_All_data", help="Suffix for the test dataset filename")
    parser.add_argument("--t0_nra", type=int, default=60, help="Time window for NRA calculation")
    parser.add_argument("--n0_nra", type=int, default=20, help="Normalization factor for NRA")
    parser.add_argument("--lambda_haf", type=float, default=100.0, help="Lambda for HAF calculation")
    parser.add_argument("--lambda_ufp", type=float, default=10.0, help="Lambda for UFP calculation")
    parser.add_argument("--combine_method", type=str, default='max', choices=['max', 'avg', 'weighted'], help="Method to combine FP scores")
    parser.add_argument("--belief_threshold", type=float, default=0.5, help="Belief threshold for classifying as FP")
    parser.add_argument("--num_fake_signatures", type=int, default=5, help="Number of fake FP signatures to generate")
    parser.add_argument("--assoc_method", type=str, default='apriori', help="Association rule method for fake sigs (e.g. apriori, fpgrowth)")
    parser.add_argument("--assoc_metric", type=str, default='confidence', help="Association rule metric for fake sigs")
    parser.add_argument("--fake_min_support", type=float, default=0.2, help="Minimum support for generating fake FP signatures")
    cli_args = parser.parse_args()
    main(cli_args) 