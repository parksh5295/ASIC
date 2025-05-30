# Algorithm: H-Mine (H-Structure Mining)
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

from collections import defaultdict
from itertools import combinations
import pandas as pd # For pd.Timestamp
import multiprocessing # Add multiprocessing

# Helper function for parallel support calculation (similar to RARM's)
# Needs to be defined at the top level or be picklable by multiprocessing
def calculate_support_for_candidate_hmine(item_tids, transaction_count, candidate_itemset):
    if not candidate_itemset:
        return 0, candidate_itemset
    # Ensure all items in candidate_itemset are in item_tids to prevent KeyError
    if not all(item in item_tids for item in candidate_itemset):
        # print(f"Warning: Item in {candidate_itemset} not found in TIDs for HMine. Skipping.")
        return 0, candidate_itemset 
    common_tids = set.intersection(*(item_tids[item] for item in candidate_itemset))
    support = len(common_tids) / transaction_count if transaction_count > 0 else 0
    return support, candidate_itemset

class HStructure:
    def __init__(self):
        self.item_counts = defaultdict(int)
        self.transaction_count = 0
        self.item_tids = defaultdict(set)  # Save transaction IDs where each item appears
    
    def add_transaction(self, tid, items):
        self.transaction_count += 1
        for item in items:
            self.item_counts[item] += 1
            self.item_tids[item].add(tid)
    
    def get_support(self, items):
        if not items:
            return 0
        # Calculate the number of transactions where all items appear simultaneously
        # Ensure all items are in self.item_tids
        if not all(item in self.item_tids for item in items):
            # print(f"Warning (HStructure.get_support): Item in {items} not found. Returning 0 support.")
            return 0
        common_tids = set.intersection(*[self.item_tids[item] for item in items])
        return len(common_tids) / self.transaction_count if self.transaction_count > 0 else 0

# Helper function for parallel rule generation for H-Mine
def generate_hmine_rules_for_itemset_task(f_itemset, min_conf, h_struct_get_support_func):
    found_strong_rule = False
    if len(f_itemset) > 1:
        support_f_itemset = h_struct_get_support_func(f_itemset)
        if support_f_itemset == 0:
            return None

        for i in range(1, len(f_itemset)):
            for antecedent_tuple in combinations(f_itemset, i):
                antecedent = frozenset(antecedent_tuple)
                support_antecedent = h_struct_get_support_func(antecedent)
                confidence = 0
                if support_antecedent > 0:
                    confidence = support_f_itemset / support_antecedent
                
                if confidence >= min_conf:
                    found_strong_rule = True
                    break
            if found_strong_rule:
                break
    
    return f_itemset if found_strong_rule else None

def h_mine(df, min_support=0.5, min_confidence=0.8, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    print(f"    [Debug H-Mine Init] Algorithm: H-Mine, Input df shape: {df.shape}, min_support={min_support}, min_confidence={min_confidence}, num_processes={num_processes}")
    start_time_total = pd.Timestamp.now()

    # Initialize H-Structure
    h_struct = HStructure()
    
    # Convert data and build H-Structure
    print(f"    [Debug H-Mine BuildStruct] Building H-Structure...")
    start_time_build = pd.Timestamp.now()
    transaction_items = []
    for tid, row in enumerate(df.itertuples(index=False, name=None)):
        items = set(f"{col}={row[idx]}" for idx, col in enumerate(df.columns))
        transaction_items.append(items)  # Keep full transactions for later use
        h_struct.add_transaction(tid, items)
    build_duration = (pd.Timestamp.now() - start_time_build).total_seconds()
    print(f"    [Debug H-Mine BuildStruct] H-Structure built. Total transactions: {h_struct.transaction_count}. Time: {build_duration:.2f}s")
    
    # Find frequent 1-itemsets
    print(f"    [Debug H-Mine Freq1] Finding frequent 1-itemsets...")
    frequent_items = {\
        item for item, count in h_struct.item_counts.items()\
        if count / h_struct.transaction_count >= min_support\
    }
    print(f"    [Debug H-Mine Freq1] Found {len(frequent_items)} frequent 1-itemsets.")
    if not frequent_items:
        print("    [Debug H-Mine Freq1] No frequent 1-items found. Returning empty list.")
        return []
    
    # Use set for rule storage (optimized for duplicate removal)
    rule_set = set()
    
    # Generate frequent itemsets and extract rules
    current_level_itemsets = [frozenset([item]) for item in frequent_items] # Changed variable name for clarity
    
    level_count = 1
    max_itemset_size = len(df.columns) if not df.empty else len(frequent_items)

    while current_level_itemsets and len(current_level_itemsets[0]) < max_itemset_size:
        if not current_level_itemsets: 
            print(f"    [Debug H-Mine Loop-{level_count}] current_level_itemsets is empty. Breaking.")
            break
        current_itemset_size = len(current_level_itemsets[0]) 
        print(f"    [Debug H-Mine Loop-{level_count}] Processing itemsets of size: {current_itemset_size}. Num itemsets in current_level: {len(current_level_itemsets)}")
        
        # Generate all potential next_level candidates first (Apriori-gen)
        potential_next_level_candidates = set()
        # Convert current_level_itemsets to a set for efficient lookup during subset checking if it's not already
        current_level_set_for_lookup = set(current_level_itemsets) 

        # Candidate Generation (Apriori-gen: join + prune)
        # Sort current_level_itemsets to ensure consistent pairing for candidate generation (optional but good practice)
        # For frozensets, direct sorting is not possible, but list of them can be sorted if elements are comparable
        # However, the itemset_idx < other_itemset_idx handles unique pairs.
        for i in range(len(current_level_itemsets)):
            for j in range(i + 1, len(current_level_itemsets)):
                itemset1 = current_level_itemsets[i]
                itemset2 = current_level_itemsets[j]
                
                # Join step: Check if they share k-1 items
                # For frozensets, intersection size then union is fine.
                # Or, convert to list, sort, compare first k-1, then merge.
                # union_len = len(itemset1.union(itemset2))
                # if union_len == current_itemset_size + 1: # A simpler check if they differ by one item, implies k-1 shared
                
                # More robust join: check if first k-1 items are the same (assuming items within frozenset are somehow ordered or comparable for this)
                # A common way for Apriori-gen with frozensets:
                if len(itemset1.intersection(itemset2)) == current_itemset_size - 1:
                    new_candidate = itemset1.union(itemset2)
                    if len(new_candidate) == current_itemset_size + 1:
                        # Pruning step: check if all (k)-subsets are in current_level_itemsets
                        all_subsets_frequent = True
                        if current_itemset_size > 0: # For candidates of size > 1
                            for subset_to_check_tuple in combinations(new_candidate, current_itemset_size):
                                if frozenset(subset_to_check_tuple) not in current_level_set_for_lookup:
                                    all_subsets_frequent = False
                                    break
                        if all_subsets_frequent:
                            potential_next_level_candidates.add(new_candidate)

        print(f"    [Debug H-Mine Loop-{level_count}] Generated {len(potential_next_level_candidates)} potential candidates for next level.")
        if not potential_next_level_candidates:
            print(f"    [Debug H-Mine Loop-{level_count}] No potential candidates generated. Breaking loop.")
            break

        # Parallel support calculation for candidates
        actual_next_level_itemsets = set()
        tasks = [(h_struct.item_tids, h_struct.transaction_count, cand) for cand in potential_next_level_candidates]

        if tasks:
            print(f"    [Debug H-Mine Loop-{level_count}] Calculating support for {len(tasks)} candidates using {num_processes} processes...")
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.starmap(calculate_support_for_candidate_hmine, tasks)
            
            for support, itemset_cand in results:
                if support >= min_support:
                    actual_next_level_itemsets.add(itemset_cand)
                    if len(actual_next_level_itemsets) % 1000 == 0 and len(actual_next_level_itemsets) > 0:
                        print(f"        [Debug H-Mine Loop-{level_count}] Found frequent candidate (total {len(actual_next_level_itemsets)}): {itemset_cand}, Support: {support:.4f}")

        print(f"    [Debug H-Mine Loop-{level_count}] Found {len(actual_next_level_itemsets)} frequent itemsets for the next level.")

        # Rule generation from current_level_itemsets - NOW PARALLELIZED
        if current_level_itemsets: # Process rules if current_level is not empty
            print(f"    [Debug H-Mine Loop-{level_count}] Generating rules from {len(current_level_itemsets)} (k={current_itemset_size}) frequent itemsets using {num_processes} processes...")
            rule_gen_tasks_hmine = [
                (item_set, min_confidence, h_struct.get_support) 
                for item_set in current_level_itemsets # Rules from current level's frequent itemsets
            ]
            if rule_gen_tasks_hmine:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    results_validated_fitemsets_for_hmine_rules = pool.starmap(generate_hmine_rules_for_itemset_task, rule_gen_tasks_hmine)
                
                for f_itemset_with_strong_rule in results_validated_fitemsets_for_hmine_rules:
                    if f_itemset_with_strong_rule:
                        rule_dict = {}
                        for rule_item_str in f_itemset_with_strong_rule:
                            key, value = rule_item_str.split('=', 1)
                            try:
                                val_float = float(value)
                                rule_dict[key] = int(val_float) if val_float.is_integer() else val_float
                            except ValueError:
                                rule_dict[key] = value
                        rule_tuple = tuple(sorted(rule_dict.items()))
                        rule_set.add(rule_tuple)
            print(f"    [Debug H-Mine Loop-{level_count}] Rule set size after level {level_count} rule generation: {len(rule_set)}")

        # Update current_level_itemsets for the next iteration
        if not actual_next_level_itemsets:
            print(f"    [Debug H-Mine Loop-{level_count}] Next_level (actual frequent) is empty. Breaking loop.")
            break
        current_level_itemsets = list(actual_next_level_itemsets) # Convert set to list for next iteration's indexed access
        level_count +=1
    
    total_duration = (pd.Timestamp.now() - start_time_total).total_seconds()
    print(f"    [Debug H-Mine Finish] H-Mine processing finished. Total unique rules/itemsets recorded: {len(rule_set)}. Total time: {total_duration:.2f}s")
    # Convert final result to dictionary list
    return [dict(rule) for rule in rule_set]
