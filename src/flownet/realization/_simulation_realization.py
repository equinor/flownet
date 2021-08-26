"""This module helps with the creation of single simulation model, which later can be simulated
using OPM Flow or a commercial reservoir simulator
"""

import os
import pathlib
from shutil import copyfile
from typing import Dict, Optional
import datetime

import numpy as np
import jinja2

from ..network_model import NetworkModel, create_egrid
from ._schedule import Schedule
from ..utils import write_grdecl_file


MODULE_FOLDER = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


class SimulationRealization:
    """
    Class which facilitates creating a valid Flow simulation case
    that can be run within the flownet framework.

    """

    def __init__(
        self,
        network: NetworkModel,
        schedule: Schedule,
        simulation_input: Dict[str, str],
        pred_schedule_file: Optional[pathlib.Path] = None,
    ):
        """
        Initialization method for for a simulation realization

        Args:
            network: FlowNet NetworkModel instance
            schedule: FlowNet Schedule instance
            simulation_input: dictionary containing simulation input
            pred_schedule_file: Path to an optional prediciton schedule include file

        """
        self._network: NetworkModel = network
        self._schedule: Schedule = schedule
        self._dims: Dict = (
            simulation_input["DIMS"]  # type: ignore[assignment]
            if "DIMS" in simulation_input
            else {}
        )
        self._includes: Dict = (
            simulation_input["INCLUDES"]  # type: ignore[assignment]
            if "INCLUDES" in simulation_input
            else {}
        )

        self._start_date: datetime.date = self._schedule.get_first_date()
        self._pred_schedule_file = pred_schedule_file

    def create_model(self, output_folder: pathlib.Path):
        """
        Function that creates a Flow simulation model

        Args:
            output_folder: Path where to store the simulation model files to disk

        Returns:
            Nothing

        """

        def isnan(value: float) -> float:
            """
            Helper function that is used as a Jinja filter while creating a SCHEDULE.

            Args:
                value: Object to be tested

            Returns:
                0 if nan, otherwise the value object

            """
            if np.isnan(value):
                return 0
            return value

        template_environment = jinja2.Environment(
            loader=jinja2.PackageLoader("flownet", "templates"),
            undefined=jinja2.StrictUndefined,
        )
        template_environment.filters["isnan"] = isnan

        # Create output folders if they don't exist
        output_folder_path = pathlib.Path(output_folder)
        os.makedirs(output_folder_path / "include", exist_ok=True)

        # Create EGRID file
        create_egrid(
            self._network.grid, output_folder_path / "FLOWNET_REALIZATION.EGRID"
        )

        # Write ACTNUM include file (necessary as long as
        # https://github.com/OPM/opm-common/issues/903 is an open issue)
        write_grdecl_file(
            self._network.grid,
            "ACTNUM",
            output_folder_path / "include" / "ACTNUM.grdecl",
            int_type=True,
        )

        if self._schedule:
            # Render SCHEDULE include file
            template = template_environment.get_template("HISTORY_SCHEDULE.inc.jinja2")
            with open(
                output_folder_path / "include" / "HISTORY_SCHEDULE.inc",
                "w",
                encoding="utf8",
            ) as fh:
                fh.write(
                    template.render(
                        {"schedule": self._schedule, "startdate": self._start_date}
                    )
                )

        for section in self._includes:
            with open(
                output_folder_path / "include" / f"{section}_PARAMETERS.inc",
                "w",
                encoding="utf8",
            ) as fh:
                fh.write(self._includes[section])

        configuration = {
            "nx": len(self._network.grid.index),
            "startdate": self._start_date,
            "welldims": {
                "number_wells": self._schedule.num_wells(),
                "max_connections": self._schedule.max_connections(),
            },
            "schedule": self._schedule,
            "sections_with_include": list(self._includes.keys()),
            "dims": self._dims,
            "pred_schedule_file": self._pred_schedule_file,
        }

        # Render main Flow .DATA file
        template = template_environment.get_template("TEMPLATE_MODEL.DATA.jinja2")
        with open(
            output_folder_path / "FLOWNET_REALIZATION.DATA", "w", encoding="utf8"
        ) as fh:
            fh.write(template.render(configuration))

        # Copy static files
        copyfile(
            MODULE_FOLDER / ".." / "static" / "SUMMARY.inc",
            output_folder_path / "include" / "SUMMARY.inc",
        )
        copyfile(
            MODULE_FOLDER / ".." / "static" / "PROPS.inc",
            output_folder_path / "include" / "PROPS.inc",
        )
        copyfile(
            MODULE_FOLDER / ".." / "static" / "SOLUTION.inc",
            output_folder_path / "include" / "SOLUTION.inc",
        )
        if self._pred_schedule_file is not None:
            copyfile(
                self._pred_schedule_file,
                output_folder_path / "include" / "PREDICTION_SCHEDULE.inc",
            )
