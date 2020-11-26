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

    if "FLOW_PATH" in os.environ:
        flow_path = os.environ.get("FLOW_PATH")
        if os.path.isfile(flow_path):
            # Runing flow from ENV variable
            subprocess.run([flow_path, args.data_file], check=True)
        else:
            print("Error: FLOW_PATH point to a path that doesn't exist")
    elif shutil.which("flow") is None:
        print("Error: flow is not installed")
    else:
        flow_path = shutil.which("flow")
        if os.path.isfile(flow_path):
            # Runing flow from Installation variable
            subprocess.run(
                [os.environ.get("FLOW_PATH", shutil.which("flow")), args.data_file],
                check=True,
            )
        else:
            print("Error: Instalation FLOW_PATH points to a path that doesn't exist")

    Path("FLOW_SIMULATION.OK").write_text("")
