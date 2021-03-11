from typing import List, Union

import configsuite


@configsuite.transformation_msg("Convert integer to list")
def _integer_to_list(input_data: Union[List, int]) -> List:
    """
    Converts integer to list with single item.

    Args:
        input_data (Union[List, int]):

    Returns:
        The input_data. If it wasn't a list yet is will be turned into a list.
    """
    if isinstance(input_data, int):
        input_data = [input_data]
    return input_data


@configsuite.transformation_msg("Convert 'None' to None")
def _str_none_to_none(
    input_data: Union[str, int, float, None]
) -> Union[str, int, float, None]:
    """
    Converts "None" to None
    Args:
        input_data (Union[str, int, float, None]):

    Returns:
        The input_data. If the input is "None" or "none" it is converted to None (str to None)
    """
    if isinstance(input_data, str):
        if input_data.lower() == "none":
            return None

    return input_data


@configsuite.transformation_msg("Convert string to lower case")
def _to_lower(input_data: Union[List[str], str]) -> Union[List[str], str]:
    if isinstance(input_data, str):
        return input_data.lower()

    return [x.lower() for x in input_data]


@configsuite.transformation_msg("Convert string to upper case")
def _to_upper(input_data: Union[List[str], str]) -> Union[List[str], str]:
    if isinstance(input_data, str):
        return input_data.upper()

    return [x.upper() for x in input_data]
