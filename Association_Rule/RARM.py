# Algorithm: RARM (Rapid Association Rule Mining)
# Improve speed by reducing the search space by eliminating unnecessary candidates, and reduce memory usage by minimizing the set of intermediate candidates
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

from collections import defaultdict
from itertools import combinations
import multiprocessing # Add multiprocessing
import pandas as pd # Add pandas for Timestamp if not already there (it seems to be missing in the provided RARM snippet but likely present)

# Helper function for parallel support calculation
# Needs to be defined at the top level or be picklable by multiprocessing
# It will take the miner's item_tids and transaction_count along with a candidate itemset
def calculate_support_for_candidate(item_tids, transaction_count, candidate_itemset):
    if not candidate_itemset:
        return 0, candidate_itemset
    common_tids = set.intersection(*(item_tids[item] for item in candidate_itemset if item in item_tids))
    support = len(common_tids) / transaction_count if transaction_count > 0 else 0
    return support, candidate_itemset

class RARMiner:
    def __init__(self):
        self.transaction_count = 0
        self.item_tids = defaultdict(set)  # TID list
        self.item_counts = defaultdict(int)  # Item count
        
    def add_transaction(self, tid, items):
        # Process single transaction
        self.transaction_count += 1
        for item in items:
            self.item_tids[item].add(tid)
            self.item_counts[item] += 1
    
    def get_support_from_tids(self, tids):
        # Calculate support from TID set
        return len(tids) / self.transaction_count
    
    def get_support(self, items):
        # Calculate support for itemset
        if not items:
            return 0
        # Calculate TID intersection
        # Ensure all items in 'items' are present in self.item_tids to avoid KeyError
        common_tids = set.intersection(*(self.item_tids[item] for item in items if item in self.item_tids))
        return self.get_support_from_tids(common_tids)
    
    def get_confidence(self, base_items, full_items):
        # Calculate confidence
        base_support = self.get_support(base_items)
        if base_support == 0:
            return 0
        return self.get_support(full_items) / base_support


def rarm(df, min_support=0.5, min_confidence=0.8, num_processes=None): # Added num_processes
    # Initialize RARM miner
    miner = RARMiner()
    
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    print(f"    [Debug RARM Init] Initializing RARM. df shape: {df.shape}, min_support={min_support}, min_confidence={min_confidence}, num_processes={num_processes}")

    # Convert data and build initial structure (streaming approach)
    for tid, row in enumerate(df.itertuples(index=False, name=None)):
        items = set(f"{col}={val}" for col, val in zip(df.columns, row))
        miner.add_transaction(tid, items)
    
    print(f"    [Debug RARM DataLoad] Finished loading {miner.transaction_count} transactions.")

    # Find frequent 1-itemset (items with minimum support)
    frequent_items = {
        item for item, count in miner.item_counts.items()
        if count / miner.transaction_count >= min_support
    }
    print(f"    [Debug RARM Freq1] Found {len(frequent_items)} frequent 1-items.")
    if not frequent_items:
        print("    [Debug RARM Freq1] No frequent 1-items found. Returning empty list.")
        return []

    # Set for rule storage
    rule_set = set()
    
    # Process level by level (memory efficient)
    current_level = {frozenset([item]) for item in frequent_items}
    
    level_count = 1
    # Limit max itemset size to avoid excessively long runs if frequent_items is huge.
    # This limit can be df.shape[1] (number of columns) or a practical limit.
    max_itemset_size = len(df.columns) if not df.empty else len(frequent_items)

    while current_level and len(next(iter(current_level))) < max_itemset_size:
        current_itemset_size = len(next(iter(current_level)))
        print(f"    [Debug RARM Loop-{level_count}] Processing itemsets of size: {current_itemset_size}. Num itemsets in current_level: {len(current_level)}")
        
        # Generate all potential next_level candidates first
        potential_next_level_candidates = set()
        if not frequent_items or not current_level: # Safety check
            break

        # Candidate Generation (Apriori-gen like from RARM's perspective)
        # itemset is k, item is 1, candidate is k+1
        for itemset in current_level:
            # Only try to extend with items that are 'larger' than any item in itemset (lexicographical or other consistent order)
            # to avoid duplicate candidates like {A,B} and {B,A}.
            # Or, more simply, ensure `item` is not already in `itemset`.
            # The original `frequent_items - itemset` handles this.
            for item in frequent_items - itemset: 
                candidate = itemset | {item}
                if len(candidate) == current_itemset_size + 1:
                    subsets_are_frequent = True
                    if current_itemset_size > 0: # Check subsets only if k > 0 (i.e., candidate size > 1)
                        for subset_to_check in combinations(candidate, current_itemset_size):
                            if frozenset(subset_to_check) not in current_level:
                                subsets_are_frequent = False
                                break
                    if subsets_are_frequent:
                        potential_next_level_candidates.add(candidate)
        
        print(f"    [Debug RARM Loop-{level_count}] Generated {len(potential_next_level_candidates)} potential candidates for next level.")
        if not potential_next_level_candidates:
            print(f"    [Debug RARM Loop-{level_count}] No potential candidates generated. Breaking loop.")
            break

        next_level_frequent_itemsets = set()
        
        # Parallel support calculation for candidates
        # Prepare arguments for the worker function.
        # miner.item_tids and miner.transaction_count are shared.
        # Convert set to list for chunking/distribution if necessary, though starmap handles iterables.
        tasks = [(miner.item_tids, miner.transaction_count, cand) for cand in potential_next_level_candidates]

        if tasks:
            print(f"    [Debug RARM Loop-{level_count}] Calculating support for {len(tasks)} candidates using {num_processes} processes...")
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Results will be a list of (support, itemset) tuples
                results = pool.starmap(calculate_support_for_candidate, tasks)
            
            for support, itemset_cand in results:
                if support >= min_support:
                    next_level_frequent_itemsets.add(itemset_cand)
                    if len(next_level_frequent_itemsets) % 1000 == 0 and len(next_level_frequent_itemsets) > 0:
                         print(f"        [Debug RARM Loop-{level_count}] Found frequent candidate for next level (count {len(next_level_frequent_itemsets)}): {itemset_cand}, Support: {support:.4f}")

        print(f"    [Debug RARM Loop-{level_count}] Found {len(next_level_frequent_itemsets)} frequent itemsets for the next level.")

        # Rule generation from newly found frequent itemsets (next_level_frequent_itemsets)
        # This part is kept sequential for now but can also be parallelized if it's a bottleneck.
        if next_level_frequent_itemsets:
            print(f"    [Debug RARM Loop-{level_count}] Generating rules from {len(next_level_frequent_itemsets)} frequent itemsets...")
            for f_itemset in next_level_frequent_itemsets: # f_itemset is a frequent itemset from the current discovery
                if len(f_itemset) > 1:
                    for i in range(1, len(f_itemset)):
                        for antecedent_tuple in combinations(f_itemset, i):
                            antecedent = frozenset(antecedent_tuple)
                            # consequent = f_itemset - antecedent # Not directly used for dict key
                            
                            confidence = miner.get_confidence(antecedent, f_itemset) # Uses the original miner.get_support
                            if confidence >= min_confidence:
                                rule_dict = {}
                                for rule_item_str in antecedent:
                                    key, value = rule_item_str.split('=', 1) # Split only on the first '='
                                    try:
                                        # Attempt to convert to float, then to int if it's a whole number
                                        val_float = float(value)
                                        if val_float.is_integer():
                                            rule_dict[key] = int(val_float)
                                        else:
                                            rule_dict[key] = val_float
                                    except ValueError:
                                        rule_dict[key] = value # Keep as string
                                
                                rule_tuple_representation = tuple(sorted(rule_dict.items()))
                                rule_set.add(rule_tuple_representation)
            print(f"    [Debug RARM Loop-{level_count}] Rule set size after level {level_count}: {len(rule_set)}")
        
        if not next_level_frequent_itemsets:
            print(f"    [Debug RARM Loop-{level_count}] Next_level (frequent) is empty. Breaking loop.")
            break
        current_level = next_level_frequent_itemsets # Move to next level
        level_count += 1
    
    print(f"    [Debug RARM Finish] RARM processing finished. Total rules found: {len(rule_set)}")
    # Convert results
    return [dict(rule) for rule in rule_set]

