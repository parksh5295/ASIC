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

from Validation.generation_asso import temp_rarm_for_fake_fp

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
    logger.info(f"--- Generating {num_fake_signatures} Fake FP Signatures from ANOMALOUS Data (using {association_metric}={min_confidence}) ---")
    
    # Initialize return variables and item_mapping_df to prevent NameError
    fake_signatures_df = pd.DataFrame()
    item_mapping_df = pd.DataFrame() # Initialize item_mapping_df
    frequent_itemsets_df = pd.DataFrame() # Initialize frequent_itemsets_df
    final_rules_df_to_return = pd.DataFrame()
    actual_num_generated = 0
    signature_count = 0

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
            return fake_signatures_df, item_mapping_df, frequent_itemsets_df, final_rules_df_to_return, signature_count, actual_num_generated
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
        categorical_rules = category_mapping.get('categorical')
        if isinstance(categorical_rules, dict) and categorical_rules:
            logger.info("Applying categorical mapping for fake signature generation...")
            for col_name, mapping_dict in categorical_rules.items():
                if col_name in data_to_map_for_rules.columns and isinstance(mapping_dict, dict):
                    # Robust mapping: convert both data and mapping keys to string
                    type_unified_mapping_dict = {str(k): v for k, v in mapping_dict.items()}
                    mapped_series = data_to_map_for_rules[col_name].astype(str).map(type_unified_mapping_dict)
                    all_mapped_series[col_name] = mapped_series

        binary_rules = category_mapping.get('binary')
        if isinstance(binary_rules, dict) and binary_rules:
            logger.info("Applying binary mapping for fake signature generation...")
            for col_name in binary_rules.keys():
                if col_name in data_to_map_for_rules.columns and data_to_map_for_rules[col_name].dtype == bool:
                     all_mapped_series[col_name] = data_to_map_for_rules[col_name].astype(int)

        if not all_mapped_series:
            print("Warning: No features were mapped. Cannot generate association rules.")
            return fake_signatures_df, item_mapping_df, frequent_itemsets_df, final_rules_df_to_return, signature_count, actual_num_generated

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
            return fake_signatures_df, item_mapping_df, frequent_itemsets_df, final_rules_df_to_return, signature_count, actual_num_generated
        print(f"Shape of final mapped ANOMALOUS data for association rules: {mapped_df.shape}")

        # 5. Run association rule mining on the (now anomalous) mapped data.
        # The _internal_fixed_confidence was originally a value passed to the RARM module, but temp_rarm_for_fake_fp uses its own fixed_confidence.
        # min_support is passed to temp_rarm_for_fake_fp as min_support_threshold.
        # association_method is not directly used when calling temp_rarm_for_fake_fp, but can be kept for logging, etc.
        logger.info(f"Using temporary RARM logic for fake FP signature generation from Validation.generation_asso.py")
        logger.info(f"Targeting {num_fake_signatures} fake FPs with min_support={min_support} for itemset generation.")

        # temp_rarm_for_fake_fp compares values directly without string conversion

        rules_df = temp_rarm_for_fake_fp(
            df=mapped_df, 
            min_support_threshold=min_support, 
            num_rules_to_generate=num_fake_signatures, # Try to generate the target number of rules
            itemset_size=2,  # Example: Generate 2-itemset rules (adjustable)
            fixed_confidence=0.95 # Give high fixed confidence to generated rules (for min_confidence filter)
        )
        # temp_rarm_for_fake_fp returns a list of dictionaries in the form of {'col':'val', 'confidence': X, 'support': Y}

        # 6. Filter rules based on the association_metric (e.g., confidence)
        # rules_df is now a list of dictionaries containing 'confidence' and 'support'
        filtered_rules = [] 
        
        if rules_df is not None and isinstance(rules_df, list):
            if rules_df: # If the list is not empty
                logger.info(f"Filtering {len(rules_df)} rules (list of dicts from temp_rarm) by '{association_metric}' >= {min_confidence}")
                for rule_dict in rules_df:
                    if isinstance(rule_dict, dict):
                        # temp_rarm_for_fake_fp returns the association_metric (usually 'confidence')
                        if association_metric in rule_dict and rule_dict[association_metric] >= min_confidence:
                            filtered_rules.append(rule_dict)
                        elif association_metric not in rule_dict:
                            logger.warning(f"Skipping rule dict from temp_rarm as it is missing '{association_metric}' key: {rule_dict}")
                        # else: rule_dict[association_metric] < min_confidence (filtered)
                    else:
                        logger.warning(f"Skipping item in rules list from temp_rarm as it is not a dictionary: {rule_dict}")
            else:
                logger.info("Temporary RARM returned an empty list. No rules to filter.")
        else:
            logger.warning(f"Temporary RARM returned an unexpected type: {type(rules_df)} or None. Expected a list.")

        # Sort rules by the association_metric (e.g., confidence) and select top N
        top_rules = [] 
        if filtered_rules:
            try:
                # temp_rarm_for_fake_fp uses fixed_confidence, so all rules' confidence may be the same.
                # Currently, sorting is done by association_metric (confidence).
                filtered_rules.sort(key=lambda x: x.get(association_metric, float('-inf')), reverse=True)
                top_rules = filtered_rules[:num_fake_signatures] # Select the top num_fake_signatures rules
                logger.info(f"Selected top {len(top_rules)} rules after filtering and sorting.")
            except KeyError as e: 
                logger.error(f"Error sorting rules from temp_rarm: A rule dictionary was missing the key '{association_metric}'. Error: {e}")
            except TypeError as e:
                 logger.error(f"Error sorting rules from temp_rarm: Could not sort rules, possibly due to incompatible types for metric '{association_metric}'. Error: {e}. Rule sample: {filtered_rules[0] if filtered_rules else 'N/A'}")
        else:
            logger.info("No rules met the filtering criteria from temp_rarm output, or no rules were generated initially.")

        # 7. Convert the top rules into the desired signature format.
        fake_signatures_list, actual_num_generated = _convert_rules_to_signatures(
            top_rules_list_of_dicts=top_rules,
            item_mapping_df=item_mapping_df, # Pass the initial empty one, it will be populated if needed
            file_type=file_type,
            is_fake_positive=True,
            category_mapping=category_mapping
        )
        
        if fake_signatures_list:
            temp_df_for_logging = pd.DataFrame(fake_signatures_list)
            if not temp_df_for_logging.empty and 'rule_dict' not in temp_df_for_logging.columns:
                logger.error("CRITICAL: DataFrame created for logging from _convert_rules_to_signatures output is missing 'rule_dict' column.")
        # else: fake_signatures_list is empty, so an empty list is returned

        if actual_num_generated > 0:
            logger.info(f"Successfully prepared {actual_num_generated} fake FP signature dicts via _convert_rules_to_signatures using temp_rarm.")
        else:
            logger.info("No fake FP signature dicts were prepared by _convert_rules_to_signatures (it returned 0 or an empty list) using temp_rarm.")

        # frequent_itemsets_df is not used here, but returns a placeholder to preserve return formatting
        frequent_itemsets_df_placeholder = pd.DataFrame()
        
        # Validate_Signature_ex.py expects a list of signature dictionaries, so return fake_signatures_list
        return fake_signatures_list, frequent_itemsets_df_placeholder

    except Exception as e:
        logger.error(f"An unexpected error occurred during fake signature generation: {e}", exc_info=True)
        # Return empty structures in case of error
        return [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0, 0
    
    # Final step: Convert the generated rules (list of dicts) into the required signature format
    final_signatures_list, item_mapping_df_final, frequent_itemsets_df_final, final_rules_df_final = _convert_rules_to_signatures(
        top_rules_list_of_dicts=top_rules,
        item_mapping_df=item_mapping_df, # Pass the initial empty one, it will be populated if needed
        file_type=file_type,
        is_fake_positive=True,
        category_mapping=category_mapping
    )

    actual_num_generated = len(final_signatures_list)
    logger.info(f"Successfully prepared {actual_num_generated} fake FP signature dicts via _convert_rules_to_signatures using temp_rarm.")

    # Ensure to return all 6 expected values
    return final_signatures_list, item_mapping_df_final, frequent_itemsets_df_final, final_rules_df_final, actual_num_generated, actual_num_generated

# ---- Helper function to convert RARM rules to signature format ----
def _convert_rules_to_signatures(top_rules_list_of_dicts, item_mapping_df, file_type, is_fake_positive, category_mapping):
    """
    Converts a list of rule dictionaries (from RARM) into the desired signature format.
    Each rule dictionary from RARM is expected to be the condition set itself.

    Args:
        top_rules_list_of_dicts (list): List of dictionaries, where each dict represents a rule's conditions
                                        (e.g., [{'Feature1': 'Value1', 'Feature2': 'Value2'}, ...]).
        item_mapping_df: Currently unused, placeholder for compatibility.
        file_type: Currently unused, placeholder for compatibility.
        is_fake_positive: Currently unused, placeholder for compatibility.
        category_mapping: Currently unused, placeholder for compatibility.

    Returns:
        tuple: (list_of_signature_dicts, count)
               list_of_signature_dicts: Each dict is like {'rule_dict': conditions_dict}
               count: Number of signatures generated.
    """
    signatures_output_list = []
    if not top_rules_list_of_dicts:
        logger.info("_convert_rules_to_signatures: Received empty top_rules_list_of_dicts.")
        return [], 0

    logger.info(f"_convert_rules_to_signatures: Converting {len(top_rules_list_of_dicts)} rules.")

    for rule_conditions_dict in top_rules_list_of_dicts:
        if not isinstance(rule_conditions_dict, dict) or not rule_conditions_dict:
            logger.warning(f"_convert_rules_to_signatures: Skipping invalid or empty rule condition dict: {rule_conditions_dict}")
            continue
        
        # The rule_conditions_dict from RARM is directly used as the 'rule_dict'
        sig_dict = {
            'rule_dict': rule_conditions_dict
            # 'id' and 'name' will be added later in Validate_Signature_ex.py
            # Other metrics like confidence/support could be added here if present in rule_conditions_dict
            # and if needed by the signature structure.
        }
        signatures_output_list.append(sig_dict)
    
    logger.info(f"_convert_rules_to_signatures: Successfully converted {len(signatures_output_list)} rules to signatures.")
    return signatures_output_list, len(signatures_output_list)
# ---- End of _convert_rules_to_signatures ----