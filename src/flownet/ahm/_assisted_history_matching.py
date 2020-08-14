import argparse
import concurrent.futures
import glob
import json
import os
import pathlib
import re
import shutil
import subprocess
from typing import List, Dict, Optional, Tuple

import jinja2
import numpy as np
import pandas as pd
import requests
from configsuite import ConfigSuite

from ..ert import create_ert_setup
from ..realization import Schedule
from ..network_model import NetworkModel
from ..parameters import Parameter
from ..parameters.probability_distributions import LogUniformDistribution

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
        case_name: str,
        config: Dict,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize an Assisted History Matching Class

        Args:
            network: NetworkModel instance
            schedule: Schedule instance
            parameters: List of Parameter objects
            case_name: Name of simulation case
            config: FlowNet config
            random_seed: Random seed to control reproducibility of FlowNet

        """
        self._network: NetworkModel = network
        self._schedule: Schedule = schedule
        self._parameters: List[Parameter] = parameters
        self._config: ConfigSuite.snapshot = config
        self._case_name: str = case_name
        self._random_seed: Optional[int] = random_seed

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
            random_seed=self._random_seed,
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

        try:
            # Ignore deprecation warnings (ERT as of August 2019 has a lot of them
            # due to transition to Python3)
            subprocess.run(
                "export PYTHONWARNINGS=ignore::DeprecationWarning;"
                f"ert es_mda --weights {','.join(map(str, weights))!r} ahm_config.ert",
                cwd=self.output_folder,
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            error_files = glob.glob(
                str(
                    self.output_folder
                    / self._config.ert.runpath.replace("%d", "*")
                    / "ERROR"
                )
            )
            raise RuntimeError(pathlib.Path(error_files[0]).read_text())

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
            (distribution.__class__, distribution.minimum, distribution.maximum)
            for parameter in self._parameters
            for distribution in parameter.random_variables
        }

        print("Unique parameter distributions:")
        print("\nDistribution             Minimum             Mean              Max")
        print("------------------------------------------------------------------")

        for distribution in distributions:
            if distribution[0] == LogUniformDistribution:
                print(
                    "Loguniform".ljust((15)),
                    f"{distribution[1]:16.8f}",
                    f"{(distribution[2] - distribution[1]) / np.log(distribution[2] / distribution[1]):16.8f}",
                    f"{distribution[2]:16.8f}",
                )
            else:
                print(
                    "Uniform".ljust((15)),
                    f"{distribution[1]:16.8f}",
                    f"{(distribution[2] + distribution[1]) / 2.0:16.8f}",
                    f"{distribution[2]:16.8f}",
                )
        print("")


def delete_simulation_output():
    """
    This function is called by a forward model in ERT, deleting unnecessary
    simulation output files.

    Returns:
        Nothing

    """
    parser = argparse.ArgumentParser(prog=("Delete simulation output."))

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

    return (realization, parameters["FLOWNET_PARAMETERS"])


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
    parser = argparse.ArgumentParser(prog=("Save iteration parameters to a file."))
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


def run_via_cloud_engine():
    """
    This function can be called to outsource forward models to a cloud compute engine.

    Returns:
        Nothing
    """
    parser = argparse.ArgumentParser(
        prog=("Run FlowNet forward models via a cloud engine.")
    )
    parser.add_argument("cloud_api_url", type=str, help="Path to the ERT runpath.")
    parser.add_argument(
        "ecl_base", type=str, help="Base name of the simulation DATA file."
    )
    args = parser.parse_args()

    print("Running forward models on the cloud...", end=" ", flush=True)

    # Pack Results
    simulation_runpath = str(pathlib.Path(args.ecl_base + ".DATA").parents[0])
    subprocess.check_output(["tar", "-czvf", "model.tar.gz", simulation_runpath])

    # Send request
    files = {"upload_file": open("model.tar.gz", "rb")}
    values = {"ecl_base": args.ecl_base}
    response = requests.post(
        args.cloud_api_url.strip("/") + "/RunFlowSimulation/", data=values, files=files
    )

    # Unpack results
    open("result.tar.gz", "wb").write(response.content)
    subprocess.check_output(["tar", "-xzf", "result.tar.gz", "-C", simulation_runpath])

    # Remove tar.gz's
    pathlib.Path("model.tar.gz").unlink()
    pathlib.Path("result.tar.gz").unlink()

    print("[Done]")
