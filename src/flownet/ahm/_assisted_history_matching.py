import argparse
from typing import List
from configsuite import ConfigSuite
import jinja2
import numpy as np

from ..ert import create_ert_setup, run_ert_subprocess
from ..realization import Schedule
from ..network_model import NetworkModel
from ..parameters import Parameter

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
        with open(self.output_folder / "webviz_config.yml", "w", encoding="utf8") as fh:
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
            timeout=self._config.ert.timeout,
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

        print("Unique parameter distributions:")
        print(
            "\nDistribution            Minimum             Mean          Stddev            Max"
        )
        print(
            "-------------------------------------------------------------------------------------"
        )
        for parameter in self._parameters:
            for random_var in parameter.random_variables:

                print(
                    f"{random_var.name}".ljust(17),
                    f"{random_var.minimum:16.8f}"
                    if random_var.minimum is not None
                    else "      None      ",
                    f"{random_var.mean:16.8f}",
                    f"{random_var.stddev:16.8f}"
                    if random_var.stddev is not None
                    else "      None      ",
                    f"{random_var.maximum:16.8f}"
                    if random_var.maximum is not None
                    else "      None      ",
                )
        print("")
