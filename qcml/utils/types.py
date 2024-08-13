def convert_to_hashable(obj):
    """Helper function to convert unhashable types to hashable ones."""
    if isinstance(obj, dict):
        return tuple(sorted((k, convert_to_hashable(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        return tuple(convert_to_hashable(x) for x in obj)
    return obj