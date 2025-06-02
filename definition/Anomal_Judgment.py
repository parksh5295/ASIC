# Return needs to be received as 'data[label]='


def anomal_judgment_nonlabel(data_type, data):
    data_line = []
    result = None # Initialize result

    if data_type == "MiraiBotnet":
        data_line = ['reconnaissance', 'infection', 'action']
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

    # Debug print to check the result being returned
    if result is not None and hasattr(result, 'head'):
        print(f"DEBUG anom_judgment_nonlabel: data_type={data_type}, result type={type(result)}, data_line={data_line}, first 5 of result: {result.head().tolist() if not result.empty else 'Result Series is empty'}")
    else:
        print(f"DEBUG anom_judgment_nonlabel: data_type={data_type}, result type={type(result)}, data_line={data_line}, result: {result}")

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
    