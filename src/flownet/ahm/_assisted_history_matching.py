import argparse
import concurrent.futures
import glob
import json
import os
import pathlib
import re
import shutil
from typing import List, Dict, Tuple

from configsuite import ConfigSuite
import jinja2
import numpy as np
import pandas as pd

from ..ert import create_ert_setup, run_ert_subprocess
from ..realization import Schedule
from ..network_model import NetworkModel
from ..parameters import Parameter
from ..parameters.probability_distributions import (
    LogUniformDistribution,
    UniformDistribution,
    NormalDistribution,
    TruncatedNormalDistribution,
)

_TEMPLATE_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.PackageLoader("flownet", "templates"),
    undefined=jinja2.StrictUndefined,
)
_TEMPLATE_ENVIRONMENT.globals["isnan"] = np.isnan


class AssistedHistoryMatching:
    """
    A class facilitating assisted history matching. Takes in a network of grid
    cells together with a dictionary of parameters with prior distributions.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        network: NetworkModel,
        schedule: Schedule,
        parameters: List[Parameter],
        config: ConfigSuite.snapshot,
    ):
        """
        Initialize an Assisted History Matching Class

        Args:
            network: NetworkModel instance
            schedule: Schedule instance
            parameters: List of Parameter objects
            config: Information from the FlowNet config YAML
        """
        self._network: NetworkModel = network
        self._schedule: Schedule = schedule
        self._parameters: List[Parameter] = parameters
        self._config: ConfigSuite.snapshot = config

    def create_ert_setup(self, args: argparse.Namespace, training_set_fraction: float):
        # pylint: disable=attribute-defined-outside-init
        """
        Creates an ERT setup, for the assisted history matching method.

        Args:
            args: The input argparse namespace
            training_set_fraction: Fraction of observations in schedule to use in training set

        Returns:
            Nothing

        """
        self._training_set_fraction = training_set_fraction
        self.output_folder = args.output_folder

        create_ert_setup(
            args,
            self._network,
            self._schedule,
            config=self._config,
            parameters=self._parameters,
            training_set_fraction=training_set_fraction,
        )

    def run_ert(self, weights: List[float]):
        """
        This function will start running ert (assumes create_ert_setup has been called).

        Currently, if you want to stop a previously started ERT run, it is not
        enough to stop the Python script. You will currently in addition need to
        manually run

        `killall ert`

        in the terminal.

        Args:
            weights: Weights for the iterated ensemble smoother to use.

        Returns:
            Nothing

        """
        with open(self.output_folder / "webviz_config.yml", "w") as fh:
            fh.write(
                _TEMPLATE_ENVIRONMENT.get_template(
                    "webviz_ahm_config.yml.jinja2"
                ).render(
                    {
                        "output_folder": self.output_folder,
                        "iterations": range(len(weights) + 1),
                        "runpath": self._config.ert.runpath,
                    }
                )
            )

        run_ert_subprocess(
            f"ert es_mda --weights {','.join(map(str, weights))!r} ahm_config.ert",
            cwd=self.output_folder,
            runpath=self._config.ert.runpath,
        )

    def report(self):
        """
        Prints relevant information of the AHM setup to stdout.

        Returns:
            Nothing

        """

        # pylint: disable=protected-access
        print(
            f"Degrees of freedom:     {sum([len(parameter._random_variables) for parameter in self._parameters]):>20}"
        )
        print(
            f"Number of observations: {self._schedule.get_nr_observations(self._training_set_fraction):>20}"
        )
        print(
            f"Number of realizations: {self._config.ert.realizations.num_realizations:>20}"
        )

        distributions = {
            (distribution.__class__)
            for parameter in self._parameters
            for distribution in parameter.random_variables
        }

        print("Unique parameter distributions:")
        print(
            "\nDistribution             Minimum             Mean              Stddev           Max"
        )
        print(
            "-------------------------------------------------------------------------------------"
        )
        for parameter in self._parameters:
            for rv in parameter.random_variables:
                #TODO: Add more distributions
                if rv.distribution == LogUniformDistribution:
                    print(
                        "Loguniform".ljust(15),
                        f"{rv.minimum:16.8f}",
                        f"{(rv.maximum - rv.minimum) / np.log(rv.maximum / rv.minimum):16.8f}",
                        f"{np.sqrt((np.log(rv.maximum / rv.minimum) * (np.power(rv.maximum,2)-np.power(rv.minimum,2)) - 2 * np.power(rv.maximum-rv.minimum,2))/(2*np.power(np.log(rv.maximum/rv.minimum),2))):16.8f}"
                        f"{rv.maximum:16.8f}",
                    )
                elif rv.distribution == UniformDistribution:
                    print(
                        "Uniform".ljust(15),
                        f"{rv.minimum:16.8f}",
                        f"{(rv.maximum + rv.minimum) / 2.0:16.8f}",
                        f"{np.sqrt(np.power(rv.maximum-rv.minimum,2)/12):16.8f}"
                        f"{rv.maximum:16.8f}",
                    )
        print("")


def delete_simulation_output():
    """
    This function is called by a forward model in ERT, deleting unnecessary
    simulation output files.

    Returns:
        Nothing

    """
    parser = argparse.ArgumentParser(prog="Delete simulation output.")

    parser.add_argument(
        "ecl_base", type=str, help="Base name of the simulation DATA file"
    )

    args = parser.parse_args()

    for suffix in ["EGRID", "INIT", "UNRST", "LOG", "PRT"]:
        if os.path.exists(f"{args.ecl_base}.{suffix}"):
            os.remove(f"{args.ecl_base}.{suffix}")


def _load_parameters(runpath: str) -> Tuple[int, Dict]:
    """
    Internal helper function to load parameter.json files in
    parallel.

    Args:
        runpath: Path to where the realization is run.

    Returns:
        Dictionary with the realizations' parameters.

    """
    realization = int(re.findall(r"[0-9]+", runpath)[-2])
    parameters = json.loads((pathlib.Path(runpath) / "parameters.json").read_text())

    return realization, parameters["FLOWNET_PARAMETERS"]


def save_iteration_parameters():
    """
    This function is called as a pre-simulation workflow in ERT, saving all
    parameters of an iteration to a file.

    The resulting dataframe is saved as a gzipped parquet file using a PyArrow table
    and has the following format (example for 5 realizations and 2 parameters):

    | index = realization | parameter 1 | parameter 2 |
    |=====================|=============|=============|
    |                   1 |         x.x |         x.x |
    |                   3 |         x.x |         x.x |
    |                   5 |         x.x |         x.x |
    |                   4 |         x.x |         x.x |
    |                   2 |         x.x |         x.x |

    Mind that the dataframe is not ordered.

    Returns:
        Nothing

    """
    parser = argparse.ArgumentParser(prog="Save iteration parameters to a file.")
    parser.add_argument("runpath", type=str, help="Path to the ERT runpath.")
    args = parser.parse_args()
    args.runpath = args.runpath.replace("%d", "*")

    print("Saving ERT parameters to file...", end=" ")

    iteration = int(re.findall(r"[0-9]+", sorted(glob.glob(args.runpath))[-1])[-1])
    runpath_list = glob.glob(args.runpath[::-1].replace("*", str(iteration), 1)[::-1])
    realizations_dict = {}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(_load_parameters, runpath_list):
            realizations_dict[result[0]] = result[1]

    pd.DataFrame(
        [parameters for _, parameters in realizations_dict.items()],
        index=realizations_dict.keys(),
    ).to_parquet(
        f"parameters_iteration-{iteration}.parquet.gzip",
        index=True,
        engine="pyarrow",
        compression="gzip",
    )

    shutil.copyfile(
        f"parameters_iteration-{iteration}.parquet.gzip",
        "parameters_iteration-latest.parquet.gzip",
    )
    print("[Done]")
