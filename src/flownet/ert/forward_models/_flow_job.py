import os
import argparse
import subprocess
from pathlib import Path
import shutil

from ert_shared.plugins.plugin_response import plugin_response
from ert_shared.plugins.plugin_manager import hook_implementation


@hook_implementation
@plugin_response(plugin_name="flow")  # pylint: disable=no-value-for-parameter
def installable_jobs():
    return {"FLOW_SIMULATION": str(Path(__file__).resolve().parent / "FLOW_SIMULATION")}


def run_flow():
    """
    This is what the FLOW_SIMULATION forward model actually run.

    Returns:
        Nothing

    """
    parser = argparse.ArgumentParser()

    parser.add_argument("data_file", type=str)
    args = parser.parse_args()

    flow_path = shutil.which("flow")
    if "FLOW_PATH" in os.environ:
        flow_path = os.environ.get("FLOW_PATH")
        if not os.path.isfile(flow_path):
            raise FileNotFoundError(
                r"$FLOW_PATH does not point at a file that exists.\n \
                Please, use the environment variable $FLOW_PATH to indicate a path for OPM Flow"
            )
    elif flow_path is None:
        raise RuntimeError(
            r"OPM Flow could not be found.\n \
            Follow the instructions on https://opm-project.org/ to install OPM Flow.\n \
            If OPM Flow is already installed, make sure it is available in $PATH,\n \
            or alternatively use the environment variable $FLOW_PATH."
        )

    subprocess.run([flow_path, args.data_file], check=True)

    Path("FLOW_SIMULATION.OK").write_text("", encoding="utf8")
