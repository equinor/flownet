import json
import argparse
import pickle
import collections
from pathlib import Path
from typing import List, Dict

import pandas as pd

from flownet.parameters import Parameter
from flownet.realization._simulation_realization import SimulationRealization


def _ert_samples2simulation_input(
    random_samples: Path, parameters: List[Parameter], realization_index: int
) -> Dict:
    """
    Reads an ERT made json file with parameter values, or a previously stored parquet file
    for a whole ensemble. The parameter values are then plugged into the corresponding parameter,
    and finally the render method is called on the different parameters, returning the Flow
    INCLUDEs to ad to the simulation model.

    Args:
        random_samples: Path to the parameters.json or previously stored parquet file.
        parameters: List of Parameters.
        realization_index: Integer representing the realization number.

    Returns:
        Dictionary with include files per section

    """
    if random_samples.suffix == ".json":
        with open(random_samples, "r", encoding="utf8") as json_file:
            unsorted_random_samples = json.load(json_file)["FLOWNET_PARAMETERS"]
    elif random_samples.suffix == ".parquet":
        df = pd.read_parquet(random_samples)
        unsorted_random_samples = json.loads(
            df[df.index == realization_index].transpose().to_json()
        )[str(realization_index)]
    else:
        raise ValueError(f"Unknown file type {random_samples}")

    # ERT does not give the same parameter output order as given to ERT as input.
    # Sort on running index in random variable names.
    sorted_names = sorted(
        list(unsorted_random_samples.keys()), key=lambda x: int(x.split("_")[0])
    )
    sorted_random_samples = [unsorted_random_samples[name] for name in sorted_names]

    for parameter in parameters:
        n = len(parameter.random_variables)
        parameter.random_samples = sorted_random_samples[:n]
        del sorted_random_samples[:n]

    parameter_output = [parameter.render_output() for parameter in parameters]

    section_ordered_includes: collections.defaultdict = collections.defaultdict(str)
    for output in parameter_output:
        for section, include in output.items():
            section_ordered_includes[section] += include + "\n"

    return section_ordered_includes


def _dims2simulation_input(parameters: List[Parameter]) -> Dict:
    """
    Lists all dims required by all parameters and creates a dictionary that
    will be included in the simulation_input.

    Args:
        parameters: List with used parameters

    Returns:
        Dictionary with dim-name and value

    """
    merged_dims_dict = dict(
        j
        for i in [
            parameter.get_dims() for parameter in parameters if parameter.get_dims()
        ]
        for j in i.items()  # type: ignore[union-attr]
    )

    if len(merged_dims_dict) != len(set(merged_dims_dict.keys())):
        raise AssertionError(
            "Some memory allocation dimensions for Flow have been defined more than once."
        )

    return merged_dims_dict


def render_realization():
    """
    This function is called by a forward model in ERT, creating a simulation
    realization from drawn parameter values, and a previously stored network
    grid.

    Returns:
        Nothing

    """
    parser = argparse.ArgumentParser(
        prog=(
            "Creates a FlowNet realization model "
            "based on a previously calculated "
            "network model, but updating e.g. "
            "porosity and permeability"
        )
    )

    parser.add_argument("pickled_network", type=Path, help="Path to pickled network")
    parser.add_argument("pickled_schedule", type=Path, help="Path to pickled schedule")
    parser.add_argument(
        "pickled_parameters", type=Path, help="Path to pickled parameters"
    )
    parser.add_argument(
        "output_folder", type=Path, help="Folder to create the model in"
    )
    parser.add_argument(
        "random_samples",
        type=Path,
        help="Path to the stored random samples (either as json or parquet)",
    )
    parser.add_argument("realization_index", type=int, help="Realization index")
    parser.add_argument(
        "pred_schedule_file",
        type=Path,
        help="Path to an (optional) prediction schedule file",
    )

    args = parser.parse_args()

    with open(args.pickled_network, "rb") as fh:
        network = pickle.load(fh)

    with open(args.pickled_schedule, "rb") as fh:
        schedule = pickle.load(fh)

    with open(args.pickled_parameters, "rb") as fh:
        parameters = pickle.load(fh)

    simulation_input = {
        "DIMS": _dims2simulation_input(parameters),
        "INCLUDES": _ert_samples2simulation_input(
            args.random_samples, parameters, args.realization_index
        ),
    }

    pred_schedule_file = (
        None if str(args.pred_schedule_file) == "None" else args.pred_schedule_file
    )

    realization = SimulationRealization(
        network, schedule, simulation_input, pred_schedule_file=pred_schedule_file
    )

    realization.create_model(args.output_folder)
