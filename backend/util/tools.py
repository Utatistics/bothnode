
def convert_to_serializable(value):
    if isinstance(value, bytes):
        return value.hex()
    elif isinstance(value, list):
        return [convert_to_serializable(item) for item in value] # Recursively convert list items
    elif isinstance(value, dict):
        return {k: convert_to_serializable(v) for k, v in value.items()} # Recursively convert dictionary values
    else:
        return value