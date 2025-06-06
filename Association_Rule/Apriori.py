# Input: dataframe (Common table shapes)
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def Apriori_rule(df, min_support=0.5, min_confidence=0.8, association_metric='confidence'):  # default; min_support=0.5, min_confidence=0.8
    print(f"    [Debug Apriori Init] Algorithm: Apriori, Input df shape: {df.shape}, min_support={min_support}, min_confidence={min_confidence}, metric={association_metric}")
    # Decide on a metrics method
    metric = association_metric

    # One-Hot Encoding Conversion - Improve memory efficiency with sparse=True
    print(f"    [Debug Apriori PreEncode] Performing One-Hot Encoding...")
    start_encode_time = pd.Timestamp.now()
    df_encoded = pd.get_dummies(df.astype(str), prefix_sep="=", sparse=True)
    encode_duration = (pd.Timestamp.now() - start_encode_time).total_seconds()
    print(f"    [Debug Apriori PostEncode] One-Hot Encoding finished. df_encoded shape: {df_encoded.shape}. Time taken: {encode_duration:.2f}s")

    # Applying Apriori
    print(f"    [Debug Apriori PreApriori] Calling mlxtend.apriori to find frequent itemsets...")
    start_apriori_time = pd.Timestamp.now()
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    apriori_duration = (pd.Timestamp.now() - start_apriori_time).total_seconds()
    print(f"    [Debug Apriori PostApriori] mlxtend.apriori finished. Found {len(frequent_itemsets)} frequent itemsets. Time taken: {apriori_duration:.2f}s")
    if frequent_itemsets.empty:
        print("    [Debug Apriori PostApriori] No frequent itemsets found. Returning empty list.")
        return []
    
    # Create association rules
    print(f"    [Debug Apriori PreRules] Calling mlxtend.association_rules to generate rules...")
    start_rules_time = pd.Timestamp.now()
    rules = association_rules(frequent_itemsets, metric=metric, 
                            min_threshold=min_confidence, 
                            num_itemsets=len(frequent_itemsets))
    rules_duration = (pd.Timestamp.now() - start_rules_time).total_seconds()
    print(f"    [Debug Apriori PostRules] mlxtend.association_rules finished. Generated {len(rules)} rules. Time taken: {rules_duration:.2f}s")
    if rules.empty:
        print("    [Debug Apriori PostRules] No rules generated. Returning empty list.")
        return []

    # Pre-split column names for faster processing
    column_map = {col: col.split("=") for col in df_encoded.columns}
    
    # Convert to set for faster duplicate checking
    rule_dicts = set()
    
    # Use to_dict('records') instead of iterrows() for faster processing
    for rule in rules[['antecedents', 'consequents']].to_dict('records'):
        combined_rule = {}
        
        # Process antecedents and consequents together
        for items in (rule['antecedents'], rule['consequents']):
            for item in items:
                key, value = column_map[item]
                combined_rule[key] = int(value)
        
        # Convert to sorted tuple for faster set addition
        rule_tuple = tuple(sorted(combined_rule.items()))
        rule_dicts.add(rule_tuple)

    # Convert set to final result format
    final_rules_list = [dict(rule) for rule in rule_dicts]
    print(f"    [Debug Apriori Finish] Apriori rule processing finished. Total unique rule antecedents/consequents combinations found: {len(final_rules_list)}")
    return final_rules_list
