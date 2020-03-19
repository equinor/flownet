from typing import Tuple, Union, List, Any

# Coordinate can be a tuple with floats, or a list with floats, or a Tuple with Numpy floats.
# Ideally, we remove the possibility of it being a list and convert Numpy floats to Python floats.
Coordinate = Union[Tuple[float, float, float], List[float], Tuple[Any, ...]]

IJK = Union[(Tuple[int, int, int],)]
