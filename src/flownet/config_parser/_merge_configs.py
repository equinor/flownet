from typing import Dict, Any
from collections.abc import Mapping


def merge_configs(base: Dict[Any, Any], update: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Merges two dictionaries (of arbitrary depth).

    Args:
        base: Initial dictionary to begin with.
        update: Dictionary with values to update with.

    Returns:
        Merged dictionary.

    """
    for key, value in update.items():
        if isinstance(value, Mapping):
            base[key] = merge_configs(base.get(key, {}), value)  # type: ignore
        else:
            base[key] = value
    return base
