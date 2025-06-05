import pandas as pd
import numpy as np
import logging
from collections import defaultdict

from Dataset_Choose_Rule.association_data_choose import file_path_line_association
from Dataset_Choose_Rule.choose_amount_dataset import file_cut
from definition.Anomal_Judgment import anomal_judgment_label, anomal_judgment_nonlabel
from utils.time_transfer import time_scalar_transfer, convert_cic_time_to_numeric_scalars
# map_intervals_to_groups is removed as we will use _apply_numeric_interval_mapping_for_fake_sigs for all interval cols
# from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups 
from Modules.Association_module import association_module
# Import the helper from Validation_util.py
from Validation.Validation_util import (
    _apply_numeric_interval_mapping_for_fake_sigs, 
    # _parse_interval_rule_string_for_fake_sigs, # Not used directly in generation_fp.py
    _apply_categorical_mapping_for_fake_sigs # Importing Newly Added Functions
)

logger = logging.getLogger(__name__)

def generate_fake_fp_signatures(file_type, file_number, category_mapping, data_list, association_method, association_metric, num_fake_signatures=3, min_support=0.3, min_confidence=0.8):
    """
    Args:
        file_type (str): Type of the dataset (e.g., 'DARPA98').
        file_number (int): Number of the dataset file.
        category_mapping (dict): Mapping information loaded from mapped_info.csv.
        data_list (list): NO LONGER USED by this function for mapping, but kept for signature compatibility if other parts rely on it.
        association_method (str): Association rule algorithm (e.g., 'apriori').
        association_metric (str): Metric to use for association rule mining (e.g., 'confidence').
        num_fake_signatures (int): Number of fake signatures to generate.
        min_support (float): Minimum support threshold for association mining on ANOMALOUS data.
        min_confidence (float): Original minimum confidence threshold from function signature (this function
                              will internally override and use 0.7 for the association_module call).

    Returns:
        list: A list of dictionaries, where each dictionary represents a fake signature rule.
              Returns empty list if generation fails.
    """
    print(f"\n--- Generating {num_fake_signatures} Fake FP Signatures from ANOMALOUS Data (using min_confidence=0.7) ---")
    fake_signatures = []
    try:
        # 1. Load data
        print("Loading data for fake signature generation...")
        file_path, _ = file_path_line_association(file_type, file_number)
        full_data = file_cut(file_type, file_path, 'all') # Load all data

        print("Applying time scalar transfer...")
        full_data = time_scalar_transfer(full_data, file_type)
        
        # Apply CICModbus specific numeric time scalar conversion if applicable
        if file_type in ['CICModbus23', 'CICModbus']:
            logger.info(f"Applying CICModbus specific numeric time scalar conversion for {file_type} within fake signature generation...")
            if full_data is not None and not full_data.empty: # Check if full_data is a DataFrame and not None
                full_data = convert_cic_time_to_numeric_scalars(full_data)
                # --- DEBUG LOGGING: After convert_cic_time_to_numeric_scalars ---
                if 'Date_scalar' in full_data.columns:
                    logger.info(f"DEBUG_FAKE_SIGS: Date_scalar after conversion - dtype: {full_data['Date_scalar'].dtype}, NaNs: {full_data['Date_scalar'].isnull().sum()}, sample: {full_data['Date_scalar'].dropna().unique()[:5]}")
                if 'StartTime_scalar' in full_data.columns:
                    logger.info(f"DEBUG_FAKE_SIGS: StartTime_scalar after conversion - dtype: {full_data['StartTime_scalar'].dtype}, NaNs: {full_data['StartTime_scalar'].isnull().sum()}, sample: {full_data['StartTime_scalar'].dropna().unique()[:5]}")
            # --- END DEBUG LOGGING ---

        # Date_scalar is now numeric (Unix timestamp)

        # Debugging after time_scalar_transfer (can be enabled if needed)
        # print("DEBUG: full_data.head() after time_scalar_transfer:")
        # print(full_data.head().to_string())
        # if 'Date_scalar' in full_data.columns:
        #     print(f"DEBUG: Date_scalar dtype: {full_data['Date_scalar'].dtype}, NaNs: {full_data['Date_scalar'].isnull().sum()}, sample: {full_data['Date_scalar'].head().to_list() if not full_data.empty else 'empty'}")

        # 2. Assign labels (same as before)
        print("Assigning labels...")
        if file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
            full_data['label'], _ = anomal_judgment_nonlabel(file_type, full_data)
        elif file_type == 'netML':
            full_data['label'] = full_data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
        elif file_type == 'DARPA98':
            full_data['label'] = full_data['Class'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
        elif file_type in ['CICIDS2017', 'CICIDS']:
            if 'Label' in full_data.columns:
                full_data['label'] = full_data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
            else: logger.error(f"ERROR: 'Label' column not found for {file_type}."); full_data['label'] = 0
        elif file_type in ['CICModbus23', 'CICModbus']:
            full_data['label'] = full_data['Attack'].apply(lambda x: 0 if str(x).strip() == 'Baseline Replay: In position' else 1)
        elif file_type in ['IoTID20', 'IoTID']:
            full_data['label'] = full_data['Label'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
        else:
            logger.warning(f"WARNING: Using generic anomal_judgment_label for {file_type}.")
            full_data['label'] = anomal_judgment_label(full_data)

        # 3. Filter for ANOMALOUS data.
        anomalous_data_df = full_data[full_data['label'] == 1].copy()
        if anomalous_data_df.empty:
            print("Warning: No ANOMALOUS data found. Cannot generate fake signatures.")
            return []
        print(f"Filtered for ANOMALOUS data. Rows: {anomalous_data_df.shape[0]}")

        # 4. Map the ANOMALOUS data using category_mapping and _apply_numeric_interval_mapping_for_fake_sigs
        data_to_map_for_rules = anomalous_data_df.drop(columns=['label'], errors='ignore')
        
        all_mapped_series = {} # Store mapped series here

        # --- Interval Feature Mapping --- 
        interval_rules_df = category_mapping.get('interval', pd.DataFrame())
        if not interval_rules_df.empty:
            logger.info(f"Applying feature mapping for columns: {interval_rules_df.columns.tolist()}")
            for col_name in interval_rules_df.columns:
                if col_name in data_to_map_for_rules.columns:
                    data_series = data_to_map_for_rules[col_name]
                    current_rule_series_for_col = interval_rules_df[col_name]

                    logger.info(f"  Processing mapping for column: {col_name}")
                    # --- DEBUG LOGGING: Before mapping ---
                    logger.info(f"    DEBUG_FAKE_SIGS: For {col_name} - Input data_series (sample): {data_series.dropna().unique()[:5]}, dtype: {data_series.dtype}, NaNs: {data_series.isnull().sum()}")
                    unique_rules_sample = current_rule_series_for_col.dropna().unique()[:5]
                    logger.info(f"    DEBUG_FAKE_SIGS: For {col_name} - Input rule_series (unique sample): {unique_rules_sample.tolist()}")
                    # --- END DEBUG LOGGING ---

                    mapped_series = pd.Series([pd.NA] * len(data_series), index=data_series.index, dtype='Int64') # Initialize with NA
                    actionable_rules = current_rule_series_for_col.dropna()

                    if not actionable_rules.empty:
                        first_rule = str(actionable_rules.iloc[0]) # Peek at the first valid rule to determine type
                        
                        # Heuristic for interval rule: contains '(', ',', and ')' or ']'
                        is_interval_rule = ('(' in first_rule and ',' in first_rule and 
                                            (')' in first_rule or ']' in first_rule))
                        # Heuristic for categorical: contains '='
                        is_categorical_rule_candidate = '=' in first_rule

                        if is_interval_rule:
                            logger.info(f"    Treating {col_name} as numeric interval mapping.")
                            mapped_series = _apply_numeric_interval_mapping_for_fake_sigs(data_series, current_rule_series_for_col, feature_name=col_name)
                        elif is_categorical_rule_candidate: # Check if it's categorical and not an interval that happens to have '='.
                            logger.info(f"    Treating {col_name} as categorical mapping.")
                            mapped_series = _apply_categorical_mapping_for_fake_sigs(data_series, current_rule_series_for_col, feature_name=col_name)
                        else:
                            logger.warning(f"    Could not determine rule type for {col_name} from rule: '{first_rule}'. All values for this column will be NA.")
                    else:
                        logger.warning(f"    No valid rules found for {col_name}. All values for this column will be NA.")
                    
                    all_mapped_series[col_name] = mapped_series
                    # --- DEBUG LOGGING: After mapping ---
                    logger.info(f"    DEBUG_FAKE_SIGS: Mapped {col_name} NaNs: {mapped_series.isnull().sum()}, Mapped unique values (sample): {mapped_series.dropna().unique()[:5]}")
                    # --- END DEBUG LOGGING ---
                else:
                    logger.warning(f"  Warning: Rule column '{col_name}' not in data_to_map_for_rules.")
        else:
            logger.warning("Warning: No interval/categorical rules found in category_mapping ('interval' key).")

        # --- (Optional) Categorical and Binary Feature Mapping --- 
        # If fake signatures should also be generated based on categorical/binary features from category_mapping:
        # This part would require similar logic: iterate through category_mapping['categorical'] and category_mapping['binary'],
        # parse their rules (which are simpler, usually direct value-to-group), and apply to data_to_map_for_rules.
        # For now, assuming fake signatures are primarily based on interval features as per original structure focus.
        # If categorical_rules_df = category_mapping.get('categorical', pd.DataFrame()): ... etc.

        if not all_mapped_series:
            print("Warning: No features were mapped. Cannot generate association rules.")
            return []

        mapped_df = pd.DataFrame(all_mapped_series, index=data_to_map_for_rules.index)
        # Fill remaining columns in data_to_map_for_rules that were not mapped
        for col in data_to_map_for_rules.columns:
            if col not in mapped_df.columns:
                mapped_df[col] = data_to_map_for_rules[col]

        # Debugging after all mapping and before dropna
        # print(f"DEBUG_FAKE_SIGS: normal_mapped_df head AFTER all mapping (before dropna):\n{normal_mapped_df.head().to_string()}")
        # print(f"DEBUG_FAKE_SIGS: normal_mapped_df NaNs AFTER all mapping (before dropna):\n{normal_mapped_df.isnull().sum().to_string()}")
        # print(f"DEBUG_FAKE_SIGS: normal_mapped_df dtypes:\n{normal_mapped_df.dtypes}")

        # --- Handle NaN values from the mapped data ---
        rows_before_dropna = mapped_df.shape[0]
        mapped_df = mapped_df.dropna()
        rows_after_dropna = mapped_df.shape[0]
        if rows_before_dropna > rows_after_dropna:
            print(f"Dropped {rows_before_dropna - rows_after_dropna} rows containing NaN values from mapped ANOMALOUS data.")
        
        if mapped_df.empty:
            print("Warning: No data left after dropping NaN rows from mapped ANOMALOUS data. Cannot generate fake signatures.")
            return []
        print(f"Shape of final mapped ANOMALOUS data for association rules: {mapped_df.shape}")

        # 5. Run association rule mining on the (now anomalous) mapped data.
        min_support = 0.2 # As per user request for fake signature generation
        _internal_fixed_confidence = 0.7
        print(f"Running {association_method} on ANOMALOUS data (min_support={min_support}, using fixed min_confidence={_internal_fixed_confidence})...")
        
        # Ensure all columns in mapped_df are of a type that association_module can handle (e.g., int, category)
        # If _apply_numeric_interval_mapping_for_fake_sigs returns float for mapped categories, convert to int.
        for col in mapped_df.columns:
            if pd.api.types.is_float_dtype(mapped_df[col]):
                # Attempt to convert to integer; if it fails due to non-integer floats (e.g. NaN still present if dropna failed for some reason)
                # this might need more robust handling or ensuring _apply_numeric... returns int-compatible types.
                try:
                    mapped_df[col] = mapped_df[col].astype(pd.Int64Dtype()) # Allows <NA>
                except Exception as e_astype:
                    print(f"Warning: Could not convert column {col} to Int64Dtype: {e_astype}. It might contain non-integer floats or unhandled NaNs.")
        
        rules_df = association_module(
            mapped_df, 
            association_method,
            association_metric=association_metric,
            min_support=min_support,
            min_confidence=_internal_fixed_confidence
        )

        # 6. Filter rules based on the association_metric (e.g., confidence)
        # rules_df is what's returned by association_module. It can be a DataFrame or a list of dicts.
        filtered_rules = [] # This will store a list of rule dictionaries
        
        # Check if rules_df actually contains rules before trying to process
        if rules_df is not None: 
            if isinstance(rules_df, pd.DataFrame):
                if not rules_df.empty and association_metric in rules_df.columns:
                    logger.info(f"Filtering {len(rules_df)} rules (DataFrame) by {association_metric} >= {min_confidence}")
                    # Apply filter
                    filtered_rules_df = rules_df[rules_df[association_metric] >= min_confidence]
                    # Convert to list of dicts for unified processing later
                    filtered_rules = filtered_rules_df.to_dict(orient='records')
                else:
                    logger.warning(f"'rules_df' is a DataFrame but it's empty or missing the '{association_metric}' column. No rules to filter.")
            elif isinstance(rules_df, list):
                if rules_df: # Check if list is not empty
                    logger.info(f"Filtering {len(rules_df)} rules (list of dicts) by '{association_metric}' >= {min_confidence}")
                    for rule_dict in rules_df:
                        if isinstance(rule_dict, dict):
                            if association_metric in rule_dict and rule_dict[association_metric] >= min_confidence:
                                filtered_rules.append(rule_dict)
                            elif association_metric not in rule_dict:
                                logger.warning(f"Skipping rule dict as it is missing '{association_metric}' key: {rule_dict}")
                        else:
                            logger.warning(f"Skipping item in rules list as it is not a dictionary: {rule_dict}")
                else:
                    logger.info("'rules_df' is an empty list. No rules to filter.")
            else:
                logger.warning(f"'rules_df' is of an unexpected type: {type(rules_df)}. Expected DataFrame or list. No rules to filter.")
        else:
            logger.info("No rules generated by association_module (rules_df is None). Nothing to filter.")

        # Sort rules by confidence (or other metric) and select top N
        top_rules = [] # This will store the final list of rule dictionaries to be converted to signatures
        if filtered_rules:
            try:
                # Sort the list of dictionaries
                filtered_rules.sort(key=lambda x: x.get(association_metric, float('-inf')), reverse=True)
                top_rules = filtered_rules[:num_fake_signatures]
                logger.info(f"Selected top {len(top_rules)} rules after filtering and sorting.")
            except KeyError as e: # Should be caught by get if metric is missing, but good for safety
                logger.error(f"Error sorting rules: A rule dictionary was missing the key '{association_metric}'. Error: {e}")
            except TypeError as e:
                 logger.error(f"Error sorting rules: Could not sort rules, possibly due to incompatible types for metric '{association_metric}'. Error: {e}. Rule sample: {filtered_rules[0] if filtered_rules else 'N/A'}")
        else:
            logger.info("No rules met the filtering criteria or no rules were generated initially.")

        # 7. Convert the top rules (which are now a list of dicts) into the desired signature format.
        fake_signatures_df = pd.DataFrame() # Default to empty DataFrame
        actual_num_generated = 0

        if top_rules: # If we have any rules to convert
            # Convert the list of rule dictionaries (top_rules) to a DataFrame, 
            # as _convert_rules_to_signatures likely expects a DataFrame.
            rules_for_conversion_df = pd.DataFrame(top_rules)
            
            # Ensure the created DataFrame is not empty and has essential columns before proceeding
            # (e.g., 'antecedents', 'consequents' are common in association rule dicts)
            if not rules_for_conversion_df.empty and \ 
               all(col in rules_for_conversion_df.columns for col in ['antecedents', 'consequents']): 
                try:
                    logger.info(f"Converting {len(rules_for_conversion_df)} rules to signatures.")
                    # The item_mapping_df should be available from earlier in the function
                    # where transactions were prepared for the association_module.
                    fake_signatures_df, actual_num_generated = _convert_rules_to_signatures(
                        rules_for_conversion_df, 
                        item_mapping_df, # This should be defined from _generate_transactions_and_mapping
                        file_type,
                        is_fake_positive=True, # These are FP signatures
                        category_mapping=category_mapping # Pass along for potential use
                    )
                    if actual_num_generated > 0:
                        logger.info(f"Successfully generated {actual_num_generated} fake FP signatures.")
                    else:
                        logger.info("No fake FP signatures were generated by _convert_rules_to_signatures (it returned 0).")
                except Exception as e_convert:
                    logger.error(f"Error during _convert_rules_to_signatures: {e_convert}", exc_info=True)
            elif rules_for_conversion_df.empty:
                logger.warning("DataFrame created from top_rules is empty. Cannot convert to signatures.")
            else:
                missing_cols = [col for col in ['antecedents', 'consequents'] if col not in rules_for_conversion_df.columns]
                logger.warning(f"DataFrame created from top_rules is missing essential columns ({missing_cols}) for signature conversion.")
        else:
            logger.info("No top rules available to convert into signatures.")

        # Ensure frequent_itemsets_df is defined, even if empty, if it's part of the function's return signature.
        if 'frequent_itemsets_df' not in locals():
            frequent_itemsets_df = pd.DataFrame() # Initialize if not created elsewhere
        
        # The function should return the rules that were actually used/selected.
        # top_rules is a list of dicts; convert to DataFrame for consistency if needed by caller.
        final_rules_df_to_return = pd.DataFrame(top_rules) if top_rules else pd.DataFrame()

        logger.info(f"--- Fake FP Signature Generation (from ANOMALOUS data with {min_confidence} confidence) Complete ---")
        if actual_num_generated == 0:
            logger.info("No fake FP signatures were generated overall in this run.")

        # Adjust return statement to match what the calling code expects.
        # Common return: fake_signatures_df, item_mapping_df, frequent_itemsets_df, rules_df, signature_count, actual_num_generated
        return fake_signatures_df, item_mapping_df, frequent_itemsets_df, final_rules_df_to_return, len(fake_signatures_df), actual_num_generated

    except Exception as e:
        print(f"Error during fake signature generation (intended from ANOMALOUS data): {e}")
        import traceback
        traceback.print_exc()

    print("--- Fake FP Signature Generation (from ANOMALOUS data with 0.7 confidence) Complete ---")
    return fake_signatures