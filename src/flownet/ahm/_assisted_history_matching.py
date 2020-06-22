import argparse
import concurrent.futures
import glob
import json
import os
import pathlib
import re
import shutil
import subprocess
from typing import List, Dict, Optional, Tuple, Union

import jinja2
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from configsuite import ConfigSuite

from ..ert import create_ert_setup
from ..realization import Schedule
from ..network_model import NetworkModel
from ..parameters import Parameter
from ..parameters.probability_distributions import LogUniformDistribution
from ..parameters import (
    PorvPoroTrans,
    RockCompressibility,
    RelativePermeability,
    Aquifer,
    Equilibration,
    FaultTransmissibility,
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
        case_name: str,
        ert_config: Dict,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize an Assisted History Matching Class

        Args:
            network: NetworkModel instance
            schedule: Schedule instance
            parameters: List of Parameter objects
            case_name: Name of simulation case
            ert_config: Dictionary containing information about queue (system, name, server and max_running)
                and realizations (num_realizations, required_success_percent and max_runtime)
            random_seed: Random seed to control reproducibility of FlowNet

        """
        self._network: NetworkModel = network
        self._schedule: Schedule = schedule
        self._parameters: List[Parameter] = parameters
        self._ert_config: dict = ert_config
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
            ert_config=self._ert_config,
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
                        "runpath": self._ert_config["runpath"],
                    }
                )
            )

        # Ignore deprecation warnings (ERT as of August 2019 has a lot of them
        # due to transition to Python3)
        subprocess.run(
            "export PYTHONWARNINGS=ignore::DeprecationWarning;"
            f"ert es_mda --weights {','.join(map(str, weights))!r} ahm_config.ert",
            cwd=self.output_folder,
            shell=True,
            check=True,
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
            f"Number of realizations: {self._ert_config['realizations'].num_realizations:>20}"
        )

        distributions = {
            (distribution.__class__, distribution.minimum, distribution.maximum)
            for parameter in self._parameters
            for distribution in parameter.random_variables
        }

        print(f"Unique parameter distributions:")
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


def find_training_set_fraction(
    schedule: Schedule, config: ConfigSuite.snapshot
) -> float:
    """
    Args:
        schedule: FlowNet Schedule instance
        config: Information from the FlowNet config yaml
    Returns:
        Fraction of the observations to be used as a training set

    """
    training_set_fraction = 1.0

    if config.flownet.training_set_end_date is not None:
        if config.flownet.training_set_fraction is not None:
            print(
                "\nTraining set fraction and training set end date are both defined in config file.\n"
                "The input given for training set fraction will be ignored.\n"
                "The training set end date will be used to calculate the training set fraction.\n"
            )
        if (
            not schedule.get_first_date()
            <= config.flownet.training_set_end_date
            <= schedule.get_dates()[-1]
        ):
            raise AssertionError("Training set end date outside of date range")
        training_set_fraction = float(
            sum(
                date < config.flownet.training_set_end_date
                for date in schedule.get_dates()
            )
            / len(schedule.get_dates())
        )
    elif config.flownet.training_set_fraction is not None:
        training_set_fraction = config.flownet.training_set_fraction

    return training_set_fraction


def _get_distribution(
    parameters: Union[str, List[str]], parameters_config: Dict, index: list
) -> pd.DataFrame:
    """
    Create the distribution min-max for one or more parameters

    Args:
        parameters: which parameter(s) should be outputted in the dataframe
        parameters_config: the parameters definition from the configuration file
        index: listing used to determine how many times to repeat the distribution

    Returns:
        A dataframe with distributions for the requested parameter(s)

    """
    if not isinstance(parameters, list):
        parameters = [parameters]

    df = pd.DataFrame(index=index)

    for parameter in parameters:
        parameter_config = getattr(parameters_config, parameter)

        if parameter_config.mean is not None:
            mean = parameter_config.mean

            if parameter_config.loguniform is True:
                # pylint: disable=cell-var-from-loop
                if parameter_config.max is not None:
                    dist_max = parameter_config.max
                    dist_min = minimize(
                        lambda x: (mean - ((dist_max - x) / np.log(dist_max / x))) ** 2,
                        x0=mean,
                        tol=1e-9,
                        method="L-BFGS-B",
                        bounds=[(1e-9, mean)],
                    ).x[0]
                else:
                    dist_min = parameter_config.min
                    dist_max = minimize(
                        lambda x: (mean - ((x - dist_min) / np.log(x / dist_min))) ** 2,
                        x0=mean,
                        tol=1e-9,
                        method="L-BFGS-B",
                        bounds=[(mean, None)],
                    ).x[0]
            else:
                if parameter_config.max is not None:
                    dist_max = parameter_config.max
                    dist_min = mean - (dist_max - mean)
                else:
                    dist_min = parameter_config.min
                    dist_max = mean + (mean - dist_min)
        else:
            dist_min = parameter_config.min
            dist_max = parameter_config.max

        df[f"minimum_{parameter}"] = dist_min
        df[f"maximum_{parameter}"] = dist_max
        df[f"loguniform_{parameter}"] = parameter_config.loguniform

    return df


def create_parameter_distributions(
    config: ConfigSuite.snapshot, network: NetworkModel
) -> List[
    Union[
        PorvPoroTrans,
        RockCompressibility,
        RelativePermeability,
        Aquifer,
        Equilibration,
        FaultTransmissibility,
    ]
]:
    """
    Creates the parameter distribution dataframe used in history matching.

    Args:
        parameters: which parameter(s) should be outputted in the dataframe
        parameters_config: the parameters definition from the configuration file
        index: listing used to determine how many times to repeat the distribution

    Returns:
        A list of parameters with their respective distributions

    """
    # pylint: disable=too-many-branches

    #########################################
    # Set the range on uncertain parameters #
    #########################################

    ##########################################
    # Pore volume, porosity and permeability #
    ##########################################

    # Create a tube index to cell index dataframe:
    ti2ci = pd.DataFrame(data=network.grid.index, index=network.grid.model)

    porv_poro_trans_dist_values = _get_distribution(
        ["bulkvolume_mult", "porosity", "permeability"],
        config.model_parameters,
        network.grid.model.unique(),
    )

    #########################################
    # Relative Permeability                 #
    #########################################

    # Create a Pandas dataframe with all SATNUMs based on the chosen scheme
    if config.model_parameters.relative_permeability.scheme == "individual":
        df_satnum = pd.DataFrame(
            range(1, len(network.grid.model.unique()) + 1), columns=["SATNUM"]
        )
    elif config.model_parameters.relative_permeability.scheme == "global":
        df_satnum = pd.DataFrame(
            [1] * len(network.grid.model.unique()), columns=["SATNUM"]
        )
    else:
        raise ValueError(
            f"The relative permeability scheme "
            f"'{config.model_parameters.relative_permeability.scheme}' is not valid.\n"
            f"Valid options are 'global' or 'individual'."
        )

    # Create a pandas dataframe with all parameter definition for each individual tube
    relperm_dist_values = pd.DataFrame(
        columns=["parameter", "minimum", "maximum", "loguniform", "satnum"]
    )

    relperm_dict = {
        key: value
        for key, value in config.model_parameters.relative_permeability._asdict().items()
        if value is not None
    }

    relperm_parameters = {
        key: relperm_dict[key] for key in relperm_dict if key != "scheme"
    }

    for i in df_satnum["SATNUM"].unique():
        info = [
            relperm_parameters.keys(),
            [relperm_parameters[key].min for key in relperm_parameters],
            [relperm_parameters[key].max for key in relperm_parameters],
            [False] * len(relperm_parameters),
            [i] * len(relperm_parameters),
        ]

        relperm_dist_values = relperm_dist_values.append(
            pd.DataFrame(
                list(map(list, zip(*info))),
                columns=["parameter", "minimum", "maximum", "loguniform", "satnum"],
            ),
            ignore_index=True,
        )

    #########################################
    # Equilibration                         #
    #########################################

    # Create a Pandas dataframe with all EQLNUM based on the chosen scheme
    if config.model_parameters.equil.scheme == "individual":
        df_eqlnum = pd.DataFrame(
            range(1, len(network.grid.model.unique()) + 1), columns=["EQLNUM"]
        )
    elif config.model_parameters.equil.scheme == "global":
        df_eqlnum = pd.DataFrame(
            [1] * len(network.grid.model.unique()), columns=["EQLNUM"]
        )
    else:
        raise ValueError(
            f"The equilibration scheme "
            f"'{config.model_parameters.relative_permeability.scheme}' is not valid.\n"
            f"Valid options are 'global' or 'individual'."
        )

    # Create a pandas dataframe with all parameter definition for each individual tube
    equil_dist_values = pd.DataFrame(
        columns=["parameter", "minimum", "maximum", "loguniform", "eqlnum"]
    )

    equil_config = config.model_parameters.equil
    for i in df_eqlnum["EQLNUM"].unique():
        info = [
            ["datum_pressure", "owc_depth", "gwc_depth", "goc_depth"],
            [
                equil_config.datum_pressure.min,
                None if equil_config.owc_depth is None else equil_config.owc_depth.min,
                None if equil_config.gwc_depth is None else equil_config.gwc_depth.min,
                None if equil_config.goc_depth is None else equil_config.goc_depth.min,
            ],
            [
                equil_config.datum_pressure.max,
                None if equil_config.owc_depth is None else equil_config.owc_depth.max,
                None if equil_config.gwc_depth is None else equil_config.gwc_depth.max,
                None if equil_config.goc_depth is None else equil_config.goc_depth.max,
            ],
            [False] * 4,
            [i] * 4,
        ]

        equil_dist_values = equil_dist_values.append(
            pd.DataFrame(
                list(map(list, zip(*info))),
                columns=["parameter", "minimum", "maximum", "loguniform", "eqlnum"],
            ),
            ignore_index=True,
        )

    equil_dist_values.dropna(inplace=True)

    #########################################
    # Aquifer                               #
    #########################################

    aquifer_config = config.model_parameters.aquifer

    # Create a Pandas dataframe with parameters for all aquifers, based on the chosen scheme
    if aquifer_config.scheme == "individual":
        df_aquid = pd.DataFrame(
            range(1, len(network.aquifers_xyz) + 1), columns=["AQUID"]
        )
    elif aquifer_config.scheme == "global":
        df_aquid = pd.DataFrame([1] * len(network.aquifers_xyz), columns=["AQUID"])
    else:
        raise ValueError(
            f"The aquifer scheme "
            f"'{aquifer_config['scheme']}' is not valid.\n"
            f"Valid options are 'global' or 'individual'."
        )

    # Create a pandas dataframe with all parameter definition for each individual tube
    aquifer_dist_values = pd.DataFrame(
        columns=["parameter", "minimum", "maximum", "loguniform", "aquid"]
    )

    aquifer_parameters = {
        key: value
        for key, value in aquifer_config._asdict().items()
        if key not in ("scheme", "type", "fraction", "delta_depth", "datum_depth")
    }

    for i in df_aquid["AQUID"].unique():
        info = [
            aquifer_parameters.keys(),
            [param.min for param in aquifer_parameters.values()],
            [param.max for param in aquifer_parameters.values()],
            [param.loguniform for param in aquifer_parameters.values()],
            [i] * len(aquifer_parameters),
        ]

        aquifer_dist_values = aquifer_dist_values.append(
            pd.DataFrame(
                list(map(list, zip(*info))),
                columns=["parameter", "minimum", "maximum", "loguniform", "aquid"],
            ),
            ignore_index=True,
        )

    # ******************************************************************************

    parameters = [
        PorvPoroTrans(porv_poro_trans_dist_values, ti2ci, network),
        RelativePermeability(
            relperm_dist_values,
            ti2ci,
            df_satnum,
            fast_pyscal=config.flownet.fast_pyscal,
        ),
        Equilibration(
            equil_dist_values,
            network,
            ti2ci,
            df_eqlnum,
            equil_config.datum_depth,
            config.flownet.pvt.rsvd,
        ),
        RockCompressibility(
            config.model_parameters.rock_compressibility.reference_pressure,
            config.model_parameters.rock_compressibility.min,
            config.model_parameters.rock_compressibility.max,
        ),
    ]

    if config.model_parameters.aquifer.fraction > 0:
        parameters.append(
            Aquifer(aquifer_dist_values, network, scheme=aquifer_config.scheme)
        )

    #########################################
    # Fault transmissibility                #
    #########################################

    if config.model_parameters.fault_mult:
        if isinstance(network.faults, Dict):
            fault_mult_dist_values = _get_distribution(
                ["fault_mult"], config.model_parameters, list(network.faults.keys()),
            )
            parameters.append(FaultTransmissibility(fault_mult_dist_values, network))

    return parameters
