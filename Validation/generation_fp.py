


def generate_fake_fp_signatures(file_type, file_number, category_mapping, data_list, association_method, association_metric, num_fake_signatures=3, min_support=0.3, min_confidence=0.8):
    """
    Args:
        file_type (str): Type of the dataset (e.g., 'DARPA98').
        file_number (int): Number of the dataset file.
        category_mapping (dict): Mapping information loaded from mapped_info.csv.
        data_list (list): List used by map_intervals_to_groups.
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

        # --- Add time scalar transfer step --- 
        print("Applying time scalar transfer...")
        full_data = time_scalar_transfer(full_data, file_type)

        # === START DEBUG: Check full_data after time_scalar_transfer ===
        print("DEBUG: full_data.head() after time_scalar_transfer:")
        print(full_data.head().to_string())
        if 'Date_scalar' in full_data.columns and 'StartTime_scalar' in full_data.columns:
            print("DEBUG: full_data[['Date_scalar', 'StartTime_scalar']].isnull().sum() after time_scalar_transfer:")
            print(full_data[['Date_scalar', 'StartTime_scalar']].isnull().sum().to_string())
        elif 'Date_scalar' in full_data.columns:
            print("DEBUG: full_data['Date_scalar'].isnull().sum() after time_scalar_transfer:")
            print(full_data['Date_scalar'].isnull().sum().to_string())
        elif 'StartTime_scalar' in full_data.columns:
            print("DEBUG: full_data['StartTime_scalar'].isnull().sum() after time_scalar_transfer:")
            print(full_data['StartTime_scalar'].isnull().sum().to_string())
        else:
            print("DEBUG: Neither Date_scalar nor StartTime_scalar found after time_scalar_transfer.")
        # === END DEBUG ===

        # -------------------------------------

        # 2. Assign labels
        print("Assigning labels...")
        if file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
            full_data['label'], _ = anomal_judgment_nonlabel(file_type, full_data)
        elif file_type == 'netML':
            # print(f"[DEBUG netML MAR] Columns in 'data' DataFrame for netML before processing: {data.columns.tolist()}")
            full_data['label'] = full_data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
        elif file_type == 'DARPA98':
            full_data['label'] = full_data['Class'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
        elif file_type in ['CICIDS2017', 'CICIDS']:
            print(f"INFO: Processing labels for {file_type}. Mapping BENIGN to 0, others to 1.")
            # Ensure 'Label' column exists
            if 'Label' in full_data.columns:
                full_data['label'] = full_data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
                logger.info(f"Applied BENIGN/Attack mapping for {file_type}.")
            else:
                logger.error(f"ERROR: 'Label' column not found in data for {file_type}. Cannot apply labeling.")
                # Potentially raise an error or exit if label column is critical and missing
                # For now, it will proceed and might fail later if 'label' is expected
                data['label'] = 0 # Default to 0 or some other placeholder if Label is missing
        elif file_type in ['CICModbus23', 'CICModbus']:
            full_data['label'] = full_data['Attack'].apply(lambda x: 0 if str(x).strip() == 'Baseline Replay: In position' else 1)
        elif file_type in ['IoTID20', 'IoTID']:
            full_data['label'] = full_data['Label'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
        else:
            # This is a fallback, ensure your file_type is covered above for specific handling
            logger.warning(f"WARNING: Using generic anomal_judgment_label for {file_type}.")
            full_data['label'] = anomal_judgment_label(full_data)

        # 3. Filter for ANOMALOUS data.
        #    The variable name `normal_data_df` is INTENTIONALLY PRESERVED from the original code
        #    to minimize diffs, but it will now hold anomalous data.
        normal_data_df = full_data[full_data['label'] == 1].copy() # << CORE LOGIC CHANGE: Filter for label == 1 (anomalous)
        if normal_data_df.empty:
            print("Warning: No ANOMALOUS data found after filtering. Cannot generate fake signatures.")
            return []
        print(f"Filtered for ANOMALOUS data. Rows obtained: {normal_data_df.shape[0]}")

        # 4. Map the ANOMALOUS data (using existing mapping info).
        normal_data_to_map = normal_data_df.drop(columns=['label'], errors='ignore')

        # --- START: Special handling for Date_scalar and StartTime_scalar ---
        mapped_date_scalar = None
        mapped_starttime_scalar = None
        cols_to_process_separately = ['Date_scalar', 'StartTime_scalar']
        remaining_cols_for_map_intervals = list(normal_data_to_map.columns)
        temp_category_mapping_interval = category_mapping['interval'].copy() # Work on a copy

        for col_name in cols_to_process_separately:
            if col_name in normal_data_to_map.columns and col_name in temp_category_mapping_interval.columns:
                print(f"INFO: Separately mapping '{col_name}' for fake signature generation.")
                # Ensure data is numeric before passing to our helper
                data_series = pd.to_numeric(normal_data_to_map[col_name], errors='coerce')
                rule_series = temp_category_mapping_interval[col_name]
                mapped_series = _apply_numeric_interval_mapping_for_fake_sigs(data_series, rule_series)
                
                if col_name == 'Date_scalar':
                    mapped_date_scalar = mapped_series.rename('Date_scalar_mapped') # Rename to avoid clash if needed, though we drop original
                elif col_name == 'StartTime_scalar':
                    mapped_starttime_scalar = mapped_series.rename('StartTime_scalar_mapped')
                
                # === START DEBUG 1: Check individually mapped scalar columns ===
                if col_name == 'Date_scalar' and mapped_date_scalar is not None:
                    print(f"DEBUG_FAKE_SIGS: mapped_date_scalar head:\n{mapped_date_scalar.head().to_string()}")
                    print(f"DEBUG_FAKE_SIGS: mapped_date_scalar NaNs: {mapped_date_scalar.isnull().sum()}")
                if col_name == 'StartTime_scalar' and mapped_starttime_scalar is not None:
                    print(f"DEBUG_FAKE_SIGS: mapped_starttime_scalar head:\n{mapped_starttime_scalar.head().to_string()}")
                    print(f"DEBUG_FAKE_SIGS: mapped_starttime_scalar NaNs: {mapped_starttime_scalar.isnull().sum()}")
                # === END DEBUG 1 ===

                # Remove from data and category_mapping before passing to map_intervals_to_groups
                if col_name in remaining_cols_for_map_intervals: # Should always be true here
                    remaining_cols_for_map_intervals.remove(col_name)
                if col_name in temp_category_mapping_interval.columns: # Should always be true here
                    temp_category_mapping_interval = temp_category_mapping_interval.drop(columns=[col_name])
            else:
                print(f"INFO: Column '{col_name}' not found in data or category_mapping for separate processing.")

        # Prepare data and category_mapping for the original map_intervals_to_groups
        data_for_map_intervals = normal_data_to_map[remaining_cols_for_map_intervals]
        # Create a new category_mapping dict for map_intervals_to_groups to use, with the modified interval part
        category_mapping_for_map_intervals = {
            'interval': temp_category_mapping_interval,
            'categorical': category_mapping.get('categorical', pd.DataFrame()),
            'binary': category_mapping.get('binary', pd.DataFrame())
        }
        # --- END: Special handling ---

        # Call original map_intervals_to_groups for remaining columns
        print(f"Mapping remaining interval columns using original map_intervals_to_groups: {temp_category_mapping_interval.columns.tolist()}")
        if not data_for_map_intervals.empty and not temp_category_mapping_interval.empty:
            other_mapped_df, _ = map_intervals_to_groups(data_for_map_intervals, category_mapping_for_map_intervals, data_list, regul='N')
            # === START DEBUG 2: Check other_mapped_df (from map_intervals_to_groups) ===
            print(f"DEBUG_FAKE_SIGS: other_mapped_df head after map_intervals_to_groups:\n{other_mapped_df.head().to_string()}")
            print(f"DEBUG_FAKE_SIGS: other_mapped_df NaNs after map_intervals_to_groups:\n{other_mapped_df.isnull().sum().to_string()}")
            # === END DEBUG 2 ===
        else:
            print("INFO: No remaining columns or interval rules for map_intervals_to_groups. Creating empty DataFrame for other_mapped_df.")
            other_mapped_df = pd.DataFrame(index=normal_data_to_map.index) # Ensure index compatibility

        # Combine manually mapped scalar time columns with other_mapped_df
        final_mapped_parts = []
        if other_mapped_df.shape[1] > 0:
             final_mapped_parts.append(other_mapped_df)
        if mapped_date_scalar is not None:
            # Use original name for consistency if no clash, or new name if preferred
            final_mapped_parts.append(mapped_date_scalar.rename('Date_scalar')) 
        if mapped_starttime_scalar is not None:
            final_mapped_parts.append(mapped_starttime_scalar.rename('StartTime_scalar'))
        
        if final_mapped_parts:
            # === START DEBUG 3a: Check parts before concat ===
            print("DEBUG_FAKE_SIGS: Checking parts before pd.concat:")
            for i, part_df in enumerate(final_mapped_parts):
                if part_df is not None:
                    print(f"  Part {i} ({part_df.name if hasattr(part_df, 'name') else 'DataFrame'}): shape={part_df.shape}, NaNs={part_df.isnull().sum().sum() if isinstance(part_df, pd.Series) else part_df.isnull().sum().sum()}")
                    print(f"    Head:\n{part_df.head().to_string()}")
                else:
                    print(f"  Part {i} is None")
            # === END DEBUG 3a ===
            normal_mapped_df = pd.concat(final_mapped_parts, axis=1)
            # === START DEBUG 3b: Check normal_mapped_df after concat (this is the state just before dropna) ===
            print(f"DEBUG_FAKE_SIGS: normal_mapped_df head AFTER concat (before dropna):\n{normal_mapped_df.head().to_string()}")
            print(f"DEBUG_FAKE_SIGS: normal_mapped_df NaNs AFTER concat (before dropna):\n{normal_mapped_df.isnull().sum().to_string()}")
            # === END DEBUG 3b ===
        else: # Should not happen if there was any data to map
            print("Warning: All parts for final mapped df are empty.")
            normal_mapped_df = pd.DataFrame(index=normal_data_to_map.index)

        print(f"Shape of combined mapped ANOMALOUS data: {normal_mapped_df.shape}")
        # DEBUG: Check normal_mapped_df before dropna (this was a previous debug point, can be re-enabled if needed)
        # print("DEBUG: normal_mapped_df.head() before dropna:")
        # print(normal_mapped_df.head().to_string())
        # print("DEBUG: normal_mapped_df.isnull().sum() before dropna:")
        # print(normal_mapped_df.isnull().sum().to_string())

        # --- Handle NaN values from the (now anomalous) mapped data --- 
        rows_before_dropna = normal_mapped_df.shape[0]
        normal_mapped_df = normal_mapped_df.dropna()
        rows_after_dropna = normal_mapped_df.shape[0]
        if rows_before_dropna > rows_after_dropna:
            print(f"Dropped {rows_before_dropna - rows_after_dropna} rows containing NaN values from mapped ANOMALOUS data.")
        if normal_mapped_df.empty:
            print("Warning: No data left after dropping NaN rows from mapped ANOMALOUS data. Cannot generate fake signatures.")
            return []
        # -------------------------------------------------

        # 5. Run association rule mining on the (now anomalous) mapped data.
        #    A fixed min_confidence of 0.7 will be used for this specific generation process.

        # === USER REQUESTED CHANGE: Force min_support to 0.2 for fake signature generation ===
        min_support = 0.2
        print(f"INFO: Overriding min_support to {min_support} for fake signature generation process (set before association).")
        # === END USER REQUESTED CHANGE ===

        _internal_fixed_confidence = 0.7 # Temporary internal variable for clarity
        print(f"Running {association_method} on ANOMALOUS data (min_support={min_support}, using fixed min_confidence={_internal_fixed_confidence})...")
        
        rules_df = association_module(
            normal_mapped_df, # This DataFrame, despite its name, now contains ANOMALOUS data
            association_method,
            association_metric=association_metric,
            min_support=min_support,
            min_confidence=_internal_fixed_confidence # << CORE LOGIC CHANGE: Using the fixed 0.7 confidence
        )

        # 6. Extract top rules as fake signatures
        if rules_df is not None and not rules_df.empty and 'rule' in rules_df.columns:
            potential_rules = rules_df['rule'].tolist()
            valid_rules = [rule for rule in potential_rules if isinstance(rule, dict)]

            fake_signatures = valid_rules[:num_fake_signatures]
            print(f"Generated {len(fake_signatures)} fake signature rules from ANOMALOUS data.")
        else:
            print("Warning: Association rule mining on ANOMALOUS data did not produce usable rules.")

    except Exception as e:
        print(f"Error during fake signature generation (intended from ANOMALOUS data): {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback

    print("--- Fake FP Signature Generation (from ANOMALOUS data with 0.7 confidence) Complete ---")
    return fake_signatures