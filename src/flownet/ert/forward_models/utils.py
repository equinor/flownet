import glob
from typing import List, Tuple


def normalize_ert_parameters(parameters: dict) -> dict:
    """Normalize legacy and current ERT parameter JSON formats."""
    parameters = parameters.get("FLOWNET_PARAMETERS", parameters)
    return {
        name: value["value"] if isinstance(value, dict) and "value" in value else value
        for name, value in parameters.items()
    }


def get_last_iteration(path: str) -> Tuple[int, List]:
    """
    Function to collect the last iteration number for which the simulation has run
    and the associated runpaths of this last iteration of all simulations.

    Args:
        path: ERT runpath

    Returns:
        Tuple with integer of last iteration and list of runpaths of last iteration of all simulations.

    """
    iteration = sorted(
        [int(rel_iter.replace("/", "").split("-")[-1]) for rel_iter in glob.glob(path)]
    )[-1]
    runpath_list = glob.glob(path[::-1].replace("*", str(iteration)[::-1], 1)[::-1])

    return iteration, runpath_list
