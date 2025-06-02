# Return needs to be received as 'data[label]='


def anomal_judgment_nonlabel(data_type, data):
    data_line = []
    result = None # Initialize result

    if data_type == "MiraiBotnet":
        print(f"DEBUG MiraiBotnet: Entered MiraiBotnet block. Data type: {type(data)}. Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}, Data empty: {data.empty if hasattr(data, 'empty') else 'N/A'}")
        data_line = ['reconnaissance', 'infection', 'action']
        print(f"DEBUG MiraiBotnet: Expected columns (data_line): {data_line}")
        actual_columns = data.columns.tolist() if hasattr(data, 'columns') else []
        print(f"DEBUG MiraiBotnet: Actual data columns: {actual_columns}")

        # Check if all indicator columns exist for MiraiBotnet
        if not all(col in actual_columns for col in data_line):
            missing_cols = [col for col in data_line if col not in actual_columns]
            print(f"DEBUG MiraiBotnet: Missing columns detected: {missing_cols}")
            raise ValueError(f"Missing required columns for MiraiBotnet: {missing_cols}. Available: {actual_columns}")
        
        print(f"DEBUG MiraiBotnet: All required columns found. Attempting to calculate result.")
        if hasattr(data, 'empty') and data.empty:
            print(f"DEBUG MiraiBotnet: Data is empty. Result of .any().astype(int) on empty data will be an empty Series.")
        
        try:
            result_intermediate = data[data_line].any(axis=1).astype(int)
            print(f"DEBUG MiraiBotnet: Intermediate result calculated. Type: {type(result_intermediate)}, Is None: {result_intermediate is None}, Shape: {result_intermediate.shape if hasattr(result_intermediate, 'shape') else 'N/A'}, Empty: {result_intermediate.empty if hasattr(result_intermediate, 'empty') else 'N/A'}")
            if hasattr(result_intermediate, 'head'):
                 print(f"DEBUG MiraiBotnet: Intermediate result head: {result_intermediate.head().tolist() if not result_intermediate.empty else 'Empty Series'}")
            result = result_intermediate # Assign to the function's result variable
            print(f"DEBUG MiraiBotnet: 'result' variable assigned. Is None: {result is None}")
        except Exception as e:
            print(f"DEBUG MiraiBotnet: Error during result calculation: {e}")
            # Optionally re-raise or handle, for now, it might fall through to result is None check
            # Re-raising might be better to see the direct cause
            raise # Re-raise the specific exception that occurred during the operation

    elif data_type in ['NSL-KDD', 'NSL_KDD']:
        label_col_name = None
        if 'label' in data.columns: # As confirmed by user debug output
            label_col_name = 'label'
        elif 'class' in data.columns: # Fallback
            label_col_name = 'class'
        elif 'Class' in data.columns: # Fallback
            label_col_name = 'Class'
        
        if label_col_name:
            data_line = [label_col_name]
            result = data[label_col_name].copy() # Return the original string label Series for NSL-KDD
                                                 # This will be converted to numeric 0/1 later in Main_Association_Rule.py
        else:
            raise ValueError(f"For NSL-KDD, no 'label', 'class', or 'Class' column found. Available columns: {data.columns.tolist()}")
    else:
        # Fallback for unhandled data_types
        raise NotImplementedError(f"Label judgment for data_type '{data_type}' is not implemented in anomal_judgment_nonlabel.")

    # Debug print to check the result being returned (this will run AFTER the specific block)
    print(f"DEBUG anom_judgment_nonlabel (POST-PROCESSING): data_type={data_type}, result type={type(result)}, data_line={data_line}")
    if result is not None and hasattr(result, 'head'): # Check if it's Series-like
        print(f"DEBUG anom_judgment_nonlabel (POST-PROCESSING): first 5 of result: {result.head().tolist() if not result.empty else 'Result Series is empty'}")
    else:
        print(f"DEBUG anom_judgment_nonlabel (POST-PROCESSING): result: {result}")

    if result is None: # Should ideally not be reached if logic is exhaustive for supported types
        raise ValueError(f"Resulting label Series is None for data_type '{data_type}'. This indicates an issue in label processing logic.")

    return result, data_line
    

def anomal_judgment_label(data):
    if 'Label' in data.columns:
        return data['Label']
    elif 'label' in data.columns:
        return data['label']
    else:
        print("label data error! No 'Label' or 'label' column found in the dataset")
        return None
    