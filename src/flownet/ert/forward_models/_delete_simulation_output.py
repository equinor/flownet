import os
import argparse


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
