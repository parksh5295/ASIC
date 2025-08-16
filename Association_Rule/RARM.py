# Algorithm: RARM (Rapid Association Rule Mining)
# Improve speed by reducing the search space by eliminating unnecessary candidates, and reduce memory usage by minimizing the set of intermediate candidates
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

from collections import defaultdict
from itertools import combinations
import multiprocessing # Add multiprocessing
import pandas as pd # Add pandas for Timestamp if not already there (it seems to be missing in the provided RARM snippet but likely present)

# Helper function for parallel support calculation
# Needs to be defined at the top level or be picklable by multiprocessing
# MODIFIED: This function now uses global variables set by the initializer.
def calculate_support_for_candidate(candidate_itemset):
    # Use the globally-scoped variables initialized in each worker process
    item_tids = _GLOBAL_ITEM_TIDS_RARM
    transaction_count = _GLOBAL_TOTAL_TX_RARM

    if not candidate_itemset:
        return 0, candidate_itemset # Return itemset for consistency if starmap expects it
    # Ensure all items in 'candidate_itemset' are present in item_tids to avoid KeyError
    # when an item might have been part of a candidate but had 0 support initially (though unlikely for frequent itemsets)
    valid_items_in_candidate = [item for item in candidate_itemset if item in item_tids]
    if len(valid_items_in_candidate) != len(candidate_itemset): # Should not happen if candidate comes from frequent items
        return 0, candidate_itemset


    common_tids = set.intersection(*(item_tids[item] for item in valid_items_in_candidate))
    support = len(common_tids) / transaction_count if transaction_count > 0 else 0
    return support, candidate_itemset

# NEW: Worker initializer and globals for RARM parallel rule generation
_GLOBAL_ITEM_TIDS_RARM = None
_GLOBAL_TOTAL_TX_RARM = 0

def _init_rarm_worker(item_tids, total_tx):
    """Initializes global variables for a RARM worker process."""
    global _GLOBAL_ITEM_TIDS_RARM, _GLOBAL_TOTAL_TX_RARM
    _GLOBAL_ITEM_TIDS_RARM = item_tids
    _GLOBAL_TOTAL_TX_RARM = total_tx

def _support_from_globals_rarm(items):
    """Calculates support using global TID map. For use in RARM worker processes."""
    if not items or not _GLOBAL_ITEM_TIDS_RARM or _GLOBAL_TOTAL_TX_RARM == 0:
        return 0.0
    
    # Safeguard: ensure all items are in the map to prevent KeyErrors
    if not all(item in _GLOBAL_ITEM_TIDS_RARM for item in items):
        return 0.0
        
    common_tids = set.intersection(*(_GLOBAL_ITEM_TIDS_RARM[item] for item in items))
    return len(common_tids) / _GLOBAL_TOTAL_TX_RARM

# NEW: Wrapper function for imap to handle multiple arguments for the rule generation task.
def _rarm_rule_worker_wrapper(args):
    """Helper to unpack arguments for pool.imap_unordered."""
    return generate_rules_for_itemset_task(*args)

# Helper function for parallel rule generation (OPUS/H-Mine/RARM share this)
# MODIFIED: It now uses global variables set by the initializer instead of receiving a function.
def generate_rules_for_itemset_task(f_itemset, min_conf):
    rules = []
    if len(f_itemset) > 1:
        support_f_itemset = _support_from_globals_rarm(f_itemset)
        if support_f_itemset == 0:
            return rules

        for i in range(1, len(f_itemset)):
            for antecedent_tuple in combinations(f_itemset, i):
                antecedent = frozenset(antecedent_tuple)
                support_antecedent = _support_from_globals_rarm(antecedent)
                
                if support_antecedent > 0:
                    confidence = support_f_itemset / support_antecedent
                    if confidence >= min_conf:
                        consequent = f_itemset - antecedent
                        rules.append((antecedent, consequent, confidence, support_f_itemset))
    return rules


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


def rarm(df, min_support=0.5, min_confidence=0.8, num_processes=None, file_type_for_limit=None, max_level_limit=None):
    # Initialize RARM miner
    miner = RARMiner()
    
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    # Updated debug log to include new parameters
    print(f"    [Debug RARM Init] Initializing RARM. df shape: {df.shape}, min_support={min_support}, min_confidence={min_confidence}, num_processes={num_processes}, file_type_for_limit='{file_type_for_limit}', max_level_limit={max_level_limit}")

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
        # Add level limit check here
        if file_type_for_limit in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD', 'CICIDS2017', 'CICIDS', 'Kitsune', 'CICModbus23', 'CICModbus', 'IoTID20', 'IoTID', 'netML', 'DARPA98', 'DARPA'] and \
            max_level_limit is not None and \
            level_count > max_level_limit:
            print(f"    [Debug RARM Loop-{level_count}] Reached max_level_limit ({max_level_limit}) for file_type '{file_type_for_limit}'. Breaking RARM loop.")
            break

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
        if potential_next_level_candidates:
            print(f"    [Debug RARM Loop-{level_count}] Calculating support for {len(potential_next_level_candidates)} candidates using {num_processes} processes...")
            with multiprocessing.Pool(
                processes=num_processes,
                initializer=_init_rarm_worker,
                initargs=(miner.item_tids, miner.transaction_count)
            ) as pool:
                # MODIFIED: Use imap_unordered to process results as they complete, saving memory vs. starmap.
                # The task list is now just the iterable of candidates, not a list of single-element tuples.
                results_iterator = pool.imap_unordered(calculate_support_for_candidate, potential_next_level_candidates)
            
                for support, itemset_cand in results_iterator:
                    if support >= min_support:
                        next_level_frequent_itemsets.add(itemset_cand)
                        if len(next_level_frequent_itemsets) % 1000 == 0 and len(next_level_frequent_itemsets) > 0:
                             print(f"        [Debug RARM Loop-{level_count}] Found frequent candidate for next level (count {len(next_level_frequent_itemsets)}): {itemset_cand}, Support: {support:.4f}")

        print(f"    [Debug RARM Loop-{level_count}] Found {len(next_level_frequent_itemsets)} frequent itemsets for the next level.")

        # Rule generation from newly found frequent itemsets (next_level_frequent_itemsets)
        if next_level_frequent_itemsets:
            print(f"    [Debug RARM Loop-{level_count}] Generating rules from {len(next_level_frequent_itemsets)} frequent itemsets using {num_processes} processes for rule generation...")
            # MODIFIED: Task list no longer includes the bound method.
            rule_gen_tasks = [
                (itemset, min_confidence) 
                for itemset in next_level_frequent_itemsets
            ]
            if rule_gen_tasks:
                # MODIFIED: Pool is created with an initializer to safely share read-only data with workers.
                with multiprocessing.Pool(
                    processes=num_processes,
                    initializer=_init_rarm_worker,
                    initargs=(miner.item_tids, miner.transaction_count)
                ) as pool:
                    # MODIFIED: Use imap_unordered with a wrapper to save memory by processing rule results as an iterator.
                    results_iterator = pool.imap_unordered(
                        _rarm_rule_worker_wrapper, 
                        rule_gen_tasks
                    )

                for rules_from_one_itemset in results_iterator:
                    for antecedent, consequent, confidence, support in rules_from_one_itemset:
                        # Convert to the required dictionary format for the final output
                        rule_dict = {}
                        full_itemset_for_dict = antecedent.union(consequent)
                        for item_str in full_itemset_for_dict:
                            key, value_str = item_str.split('=', 1)
                            try:
                                val_float = float(value_str)
                                rule_dict[key] = int(val_float) if val_float.is_integer() else val_float
                            except ValueError:
                                rule_dict[key] = value_str
                        
                        rule_tuple = tuple(sorted(rule_dict.items()))
                        rule_set.add(rule_tuple)
            print(f"    [Debug RARM Loop-{level_count}] Rule set size after processing level {level_count}: {len(rule_set)}")

        # Prepare for the next level
        if not next_level_frequent_itemsets:
            print(f"    [Debug RARM Loop-{level_count}] No more frequent itemsets found. Breaking loop.")
            break
        current_level = next_level_frequent_itemsets # Move to next level
        level_count += 1
    
    print(f"    [Debug RARM Finish] RARM processing finished. Total rules found: {len(rule_set)}")
    # Convert results
    return [dict(rule) for rule in rule_set]

