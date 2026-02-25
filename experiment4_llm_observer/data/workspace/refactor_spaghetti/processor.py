def _calculate_transformed_value(item_type, value):
    """
    Calculate the transformed value based on the item type.
    
    Args:
        item_type: The type of the item ('A', 'B', or 'C')
        value: The original value to transform
        
    Returns:
        The transformed value, capped at 100
    """
    if item_type == "A":
        transformed_value = value * 2
    elif item_type == "B":
        transformed_value = value + 10
    elif item_type == "C":
        transformed_value = value ** 2
    else:
        transformed_value = value
    
    # Cap the result at 100
    return min(transformed_value, 100)


def _process_item(item):
    """
    Process a single item and return its result.
    
    Args:
        item: A dictionary containing 'id', 'type', 'value', and 'active' fields
        
    Returns:
        A dictionary with 'id', 'result', and 'status' fields
    """
    item_id = item["id"]
    is_active = item["active"]
    
    if is_active:
        transformed_value = _calculate_transformed_value(item["type"], item["value"])
        return {
            "id": item_id,
            "result": transformed_value,
            "status": "processed"
        }
    else:
        return {
            "id": item_id,
            "result": 0,
            "status": "skipped"
        }


def process(data):
    """
    Process a list of items and return processed items with their total.
    
    Args:
        data: A list of dictionaries, each containing 'id', 'type', 'value', and 'active'
        
    Returns:
        A dictionary with 'items' (list of processed items) and 'total' (sum of processed results)
    """
    processed_items = []
    
    # Process each item
    for item in data:
        processed_item = _process_item(item)
        processed_items.append(processed_item)
    
    # Calculate the total of all processed items
    total = sum(
        item["result"] for item in processed_items
        if item["status"] == "processed"
    )
    
    return {"items": processed_items, "total": total}
