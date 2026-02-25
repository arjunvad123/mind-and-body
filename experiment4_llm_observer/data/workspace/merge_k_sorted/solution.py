import heapq
from typing import List

def merge_k_sorted(lists: List[List[int]]) -> List[int]:
    """
    Merge K sorted lists into a single sorted list.
    
    Time Complexity: O(N*log(K)) where N is total number of elements, K is number of lists
    Space Complexity: O(K) for the heap
    
    Args:
        lists: List of K sorted lists
        
    Returns:
        Single sorted list containing all elements
    """
    if not lists:
        return []
    
    # Filter out empty lists
    non_empty_lists = [lst for lst in lists if lst]
    
    if not non_empty_lists:
        return []
    
    if len(non_empty_lists) == 1:
        return non_empty_lists[0][:]
    
    # Min heap to store (value, list_index, element_index)
    heap = []
    result = []
    
    # Initialize heap with first element from each non-empty list
    for i, lst in enumerate(non_empty_lists):
        if lst:  # Double check for non-empty
            heapq.heappush(heap, (lst[0], i, 0))
    
    # Process elements until heap is empty
    while heap:
        value, list_idx, elem_idx = heapq.heappop(heap)
        result.append(value)
        
        # If there are more elements in the same list, add the next one
        if elem_idx + 1 < len(non_empty_lists[list_idx]):
            next_value = non_empty_lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_value, list_idx, elem_idx + 1))
    
    return result