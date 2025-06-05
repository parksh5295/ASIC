

# Helper function to parse interval rule strings specifically for fake signature generation needs
# This avoids modifying the global separate_group_mapping.py
def _parse_interval_rule_string_for_fake_sigs(rule_str):
    """
    Parses an interval rule string like "(L, U]=G" or "[L, U)=G".
    Returns (lower_bound, upper_bound, lower_inclusive, upper_inclusive, group_index).
    Handles '-inf' as lower bound.
    """
    rule_str = str(rule_str).strip() # Ensure it's a string
    match = re.match(r'([(\[])\s*(-inf|[-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*([)\]])\s*=\s*(\d+)', rule_str)
    if not match:
        # print(f"DEBUG_FAKE_SIG_MAP: Cannot parse interval rule string: {rule_str}")
        raise ValueError(f"Cannot parse interval rule string for fake sigs: {rule_str}")

    lower_bracket, lower_val_str, upper_val_str, upper_bracket, group_num_str = match.groups()
    lower_bound = -np.inf if lower_val_str == '-inf' else float(lower_val_str)
    upper_bound = float(upper_val_str)
    lower_inclusive = (lower_bracket == '[')
    upper_inclusive = (upper_bracket == ']')
    group_index = int(group_num_str)
    return lower_bound, upper_bound, lower_inclusive, upper_inclusive, group_index

# Helper function to apply parsed interval rules to a numeric data series
def _apply_numeric_interval_mapping_for_fake_sigs(numeric_data_series, rule_series):
    """
    Applies interval mapping rules to a numeric pandas Series.
    numeric_data_series: pd.Series of numeric data to be mapped.
    rule_series: pd.Series of interval rule strings (e.g., "(0,10]=0").
    Returns a pd.Series with mapped group indices.
    """
    parsed_rules = []
    for rule_str in rule_series.dropna():
        try:
            parsed_rules.append(_parse_interval_rule_string_for_fake_sigs(rule_str))
        except ValueError:
            # print(f"DEBUG_FAKE_SIG_MAP: Skipping unparsable rule for data mapping: {rule_str}")
            pass # Skip rules that can't be parsed by our helper
    
    if not parsed_rules:
        # print(f"DEBUG_FAKE_SIG_MAP: No valid rules parsed for column {numeric_data_series.name}. Returning NaNs.")
        return pd.Series(np.nan, index=numeric_data_series.index, dtype=np.float64)

    # Sort rules by lower bound, then upper bound (optional, but good practice)
    parsed_rules.sort(key=lambda x: (x[0], x[1]))

    mapped_values = pd.Series(np.nan, index=numeric_data_series.index, dtype=np.float64)
    
    # Ensure numeric_data_series is indeed numeric, coercing errors
    data_to_map = pd.to_numeric(numeric_data_series, errors='coerce')
    valid_data_mask = data_to_map.notna()

    for lower, upper, l_incl, u_incl, group_idx in parsed_rules:
        condition = pd.Series(True, index=data_to_map.index)
        if l_incl:
            condition &= (data_to_map >= lower)
        else:
            condition &= (data_to_map > lower)
        if u_incl:
            condition &= (data_to_map <= upper)
        else:
            condition &= (data_to_map < upper)
        
        final_condition = condition & valid_data_mask
        mapped_values.loc[final_condition] = group_idx
        
    return mapped_values

# Helper function for parallel calculation of single signature contribution
def _calculate_single_signature_contribution(sig_id, alerts_df_subset_cols, anomalous_indices_set, total_anomalous_alerts_count):
    """Calculates recall contribution for a single signature ID."""
    # Recreate alerts_df from the necessary columns passed
    # This is to avoid passing large DataFrames if only a subset is needed and pickling issues.
    # However, alerts_df is filtered by sig_id, so passing the relevant part or whole might be fine.
    # For simplicity here, assuming alerts_df_subset_cols is already filtered for the current sig_id OR we filter it here.
    # The original code did: sig_alerts = alerts_df[alerts_df['signature_id'] == sig_id]
    # This implies that alerts_df should be passed fully, or tasks should pre-filter.
    # For starmap, it's better if the worker function gets exactly what it needs.
    # Option 1: Pass full alerts_df and filter inside (less ideal for many tasks if alerts_df is huge)
    # Option 2: Pre-filter alerts_df for each sig_id before making tasks (more setup but cleaner worker)

    # Assuming alerts_df_subset_cols IS alerts_df (the full one, or a view with 'signature_id' and 'alert_index')
    # This will be re-evaluated based on how tasks are prepared.
    # For now, let's stick to the logic from the original loop:
    sig_alerts = alerts_df_subset_cols[alerts_df_subset_cols['signature_id'] == sig_id]
    
    detected_by_sig = anomalous_indices_set.intersection(set(sig_alerts['alert_index']))
    contribution = 0.0
    if total_anomalous_alerts_count > 0:
        contribution = len(detected_by_sig) / total_anomalous_alerts_count
    return sig_id, contribution

# ===== Helper Function: Calculate Recall Contribution Per Signature =====
def calculate_recall_contribution(group_mapped_df, alerts_df, signature_map):
    """
    Calculates the recall contribution for each signature using parallel processing.

    Args:
        group_mapped_df (pd.DataFrame): DataFrame with original data and 'label' column.
        alerts_df (pd.DataFrame): DataFrame from apply_signatures_to_dataset (covering all signatures).
        signature_map (dict): Dictionary mapping signature_id to signature rule dict.

    Returns:
        dict: Dictionary mapping signature_id to its recall contribution (0.0 to 1.0).
              Returns empty dict if errors occur.
    """
    recall_contributions = {}
    if 'label' not in group_mapped_df.columns:
        print("Error: 'label' column not found in group_mapped_df for recall contribution.")
        return recall_contributions
    if 'alert_index' not in alerts_df.columns or 'signature_id' not in alerts_df.columns:
         print("Error: 'alert_index' or 'signature_id' column not found in alerts_df for recall contribution.")
         return recall_contributions

    anomalous_indices = set(group_mapped_df[group_mapped_df['label'] == 1].index)
    total_anomalous_alerts = len(anomalous_indices)

    if total_anomalous_alerts == 0:
        print("Warning: No anomalous alerts found in group_mapped_df for recall contribution.")
        return {sig_id: 0.0 for sig_id in signature_map.keys()} # All contribute 0

    print(f"\nCalculating recall contribution for {len(signature_map)} signatures using parallel processing...")

    # Prepare tasks for parallel execution
    # Each task will be (sig_id, alerts_df, anomalous_indices, total_anomalous_alerts)
    # Pass alerts_df directly. Pandas DataFrames are picklable.
    tasks = [
        (sig_id, alerts_df[['signature_id', 'alert_index']], anomalous_indices, total_anomalous_alerts)
        for sig_id in signature_map.keys()
    ]

    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes for recall contribution calculation.")
    
    results = []
    if tasks: # Proceed only if there are signatures to process
        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Results will be a list of (sig_id, contribution) tuples
                results = pool.starmap(_calculate_single_signature_contribution, tasks)
        except Exception as e:
            print(f"An error occurred during parallel recall contribution calculation: {e}")
            # Fallback to sequential calculation or return empty/partial
            print("Falling back to sequential calculation for recall contribution...")
            for sig_id in signature_map.keys():
                sig_alerts = alerts_df[alerts_df['signature_id'] == sig_id]
                detected_by_sig = anomalous_indices.intersection(set(sig_alerts['alert_index']))
                contribution = 0.0
                if total_anomalous_alerts > 0:
                    contribution = len(detected_by_sig) / total_anomalous_alerts
                recall_contributions[sig_id] = contribution
                # Optional: print contribution per signature
                # print(f"  - {sig_id}: {contribution:.4f} (sequential)")
            return recall_contributions # Return sequentially computed results

    # Populate recall_contributions from parallel results
    for sig_id, contribution in results:
        recall_contributions[sig_id] = contribution
        # Optional: print contribution per signature
        # print(f"  - {sig_id}: {contribution:.4f} (parallel)")

    return recall_contributions
# ====================================================================

def ensure_directory_exists(filepath):
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)

# ===== Recall Calculation Helper Functions =====
def calculate_overall_recall(group_mapped_df, alerts_df, signature_map, relevant_signature_ids=None):
    '''
    Calculates the overall recall for a given set of signatures.

    Args:
        group_mapped_df (pd.DataFrame): DataFrame with original data and 'label' column.
        alerts_df (pd.DataFrame): DataFrame returned by apply_signatures_to_dataset.
                                    Expected columns: 'alert_index', 'signature_id'.
        signature_map (dict): Dictionary mapping signature_id to signature rule dict.
        relevant_signature_ids (set, optional): Set of signature IDs to consider.
                                                If None, all signatures in alerts_df are considered.

    Returns:
        float: Overall recall value (0.0 to 1.0).
    '''
    if 'label' not in group_mapped_df.columns:
        print("Error: 'label' column not found in group_mapped_df for recall calculation.")
        return 0.0
    if 'alert_index' not in alerts_df.columns or 'signature_id' not in alerts_df.columns:
         print("Error: 'alert_index' or 'signature_id' column not found in alerts_df for recall calculation.")
         return 0.0

    total_anomalous_alerts = group_mapped_df['label'].sum()
    if total_anomalous_alerts == 0:
        print("Warning: No anomalous alerts found in group_mapped_df.")
        return 0.0 # Avoid division by zero

    # Get indices of anomalous alerts in the original data
    anomalous_indices = set(group_mapped_df[group_mapped_df['label'] == 1].index)

    # Filter alerts that correspond to anomalous original data
    anomalous_alerts_df = alerts_df[alerts_df['alert_index'].isin(anomalous_indices)].copy()

    # Filter by relevant signature IDs if provided
    if relevant_signature_ids is not None:
        print(f"Calculating recall based on {len(relevant_signature_ids)} signatures.")
        anomalous_alerts_df = anomalous_alerts_df[anomalous_alerts_df['signature_id'].isin(relevant_signature_ids)]
    else:
         print("Calculating recall based on all signatures present in alerts_df.")


    # Count unique anomalous alerts detected by the relevant signatures
    detected_anomalous_alerts = anomalous_alerts_df['alert_index'].nunique()

    recall = detected_anomalous_alerts / total_anomalous_alerts
    print(f"Total Anomalous Alerts: {total_anomalous_alerts}")
    print(f"Detected Anomalous Alerts (by relevant signatures): {detected_anomalous_alerts}")

    return recall