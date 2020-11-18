import os
import argparse
import subprocess
from pathlib import Path

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

    subprocess.run(
        [os.environ.get("FLOW_PATH", "/usr/bin/flow"), args.data_file], check=True
    )

    Path("FLOW_SIMULATION.OK").write_text("")
