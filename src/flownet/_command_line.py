import sys
import argparse
import shutil
import pathlib
import subprocess

from .config_parser import parse_config, parse_pred_config, parse_hyperparam_config
from .ahm import run_flownet_history_matching, run_flownet_history_matching_from_restart
from .prediction import run_flownet_prediction
from .hyperparameter import run_flownet_hyperparameter


def create_webviz(output_folder: pathlib.Path, start_webviz: bool = True):
    """
    This function will spawn a webviz process to generated a webviz
    portable app using the FlowNet simulation results.

    Args:
        output_folder: Output folder where the webviz portable app will be stored.
        start_webviz: Start webviz?

    Returns:
        Nothing

    """
    print("Started running fmu-ensemble and webviz")

    generated_folder = "generated_app"

    subprocess.run(
        f"webviz build ./webviz_config.yml --portable { generated_folder } --theme equinor",
        cwd=output_folder,
        shell=True,
        check=True,
    )

    print(f"Webviz application created in { output_folder / generated_folder }.")

    if start_webviz:
        print("Starting webviz locally now...")
        subprocess.run(
            "python webviz_app.py",
            cwd=output_folder / generated_folder,
            check=True,
            shell=True,
        )


def flownet_ahm(args: argparse.Namespace) -> None:
    """
    Entrypoint for running flownet in AHM mode.

    Args:
        args: input namespace from argparse

    Returns:
        Nothing

    """
    if args.output_folder.exists():
        if args.overwrite:
            shutil.rmtree(args.output_folder)
        else:
            raise ValueError(
                f"{args.output_folder} already exists. Add --overwrite or change output folder."
            )

    config = parse_config(args.config, args.update_config)
    if hasattr(args, "restart_folder") and args.restart_folder is not None:
        if args.restart_folder.exists():
            # check for pickled files and zipped file
            if (
                pathlib.Path(args.restart_folder / "network.pickled").is_file()
                and pathlib.Path(args.restart_folder / "schedule.pickled").is_file()
                and pathlib.Path(args.restart_folder / "parameters.pickled").is_file()
                and pathlib.Path(
                    args.restart_folder / "parameters_iteration-latest.parquet.gzip"
                ).is_file()
            ):
                run_flownet_history_matching_from_restart(config, args)
            else:
                raise ValueError(
                    f"Case in {args.restart_folder} is not complete! The following files "
                    "are needed: network.pickled, schedule.pickled, parameters.pickled "
                    "and parameters_iteration-latest.parquet.gzip."
                )
        else:
            raise ValueError(f"{args.restart_folder} does not exist!")
    else:
        run_flownet_history_matching(config, args)

    if not args.skip_postprocessing:
        create_webviz(args.output_folder, start_webviz=args.start_webviz)


def flownet_pred(args: argparse.Namespace) -> None:
    """
    Entrypoint for running flownet in prediction mode.

    Args:
        args: input namespace from argparse

    Returns:
        Nothing

    """
    if args.output_folder.exists():
        if args.overwrite:
            shutil.rmtree(args.output_folder)
        else:
            raise ValueError(
                f"{args.output_folder} already exists. Add --overwrite or change output folder."
            )

    config = parse_pred_config(args.config, args.update_config)
    run_flownet_prediction(config, args)

    if not args.skip_postprocessing:
        create_webviz(args.output_folder, start_webviz=args.start_webviz)


def flownet_hyperparam(args: argparse.Namespace) -> None:
    """
    Entrypoint for the hyperparameter exploration and optimization mode.

    Args:
        args: input namespace from argparse

    Returns:
        Nothing

    """
    if args.output_folder.exists():
        if args.overwrite:
            shutil.rmtree(args.output_folder)
        else:
            raise ValueError(
                f"{args.output_folder} already exists. Add --overwrite or change output folder."
            )

    hyper_parameters = parse_hyperparam_config(args.config)
    run_flownet_hyperparameter(args, hyper_parameters)


def main():
    """
    Main functionality run when the 'flownet' command-line tool is called.

    The following will be performed:
        - the input parameters will be read;
        - the configuration file interpreted;
        - the assisted history match process will be started;
        - Post-processing will be performed if required.

    Returns:
        Nothing

    """
    parser = argparse.ArgumentParser(
        description=("Run Flownet in an assisted history matching setting.")
    )

    subparsers = parser.add_subparsers(
        help="The options available. "
        'Type e.g. "flownet --help" '
        "to get help on that particular "
        "option."
    )

    # Add assisted history matching argument parser:
    parser_ahm = subparsers.add_parser(
        "ahm", help="Run flownet in an assisted history matching setting."
    )

    parser_ahm.add_argument(
        "config", type=pathlib.Path, help="Configuration file with AHM settings to use."
    )
    parser_ahm.add_argument(
        "output_folder", type=pathlib.Path, help="Folder to store AHM output."
    )
    parser_ahm.add_argument(
        "--update-config",
        type=pathlib.Path,
        default=None,
        help="Optional configuration file which values will update the main config. "
        "Any relative paths in this file will also be assumed relative to the main config.",
    )
    parser_ahm.add_argument(
        "--restart-folder",
        type=pathlib.Path,
        default=None,
        help="Optional folder containing results from a previous history match.",
    )
    parser_ahm.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists",
    )
    parser_ahm.add_argument(
        "--skip-postprocessing",
        action="store_true",
        help="Do not run any postprocessing after ERT is finished",
    )
    parser_ahm.add_argument(
        "--start-webviz",
        action="store_true",
        help="Start webviz automatically. This flag has no effect if also --skip-postprocessing is used",
    )
    parser_ahm.add_argument(
        "--debug",
        action="store_true",
        help="Keeps all files generated by ERT for easier inspection of results during debugging",
    )

    parser_ahm.set_defaults(func=flownet_ahm)

    # Add assisted history matching argument parser:
    parser_pred = subparsers.add_parser(
        "pred", help="Run flownet in prediction based mode."
    )

    parser_pred.add_argument(
        "config",
        type=pathlib.Path,
        help="Configuration file with prediction settings to use.",
    )
    parser_pred.add_argument(
        "output_folder", type=pathlib.Path, help="Folder to store prediction output."
    )
    parser_pred.add_argument(
        "ahm_folder",
        type=pathlib.Path,
        help="Folder to where the AHM run to be base on is located.",
    )
    parser_pred.add_argument(
        "--update-config",
        type=pathlib.Path,
        default=None,
        help="Optional configuration file which values will update the main config.",
    )
    parser_pred.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists",
    )
    parser_pred.add_argument(
        "--skip-postprocessing",
        action="store_true",
        help="Do not run any postprocessing after ERT is finished",
    )
    parser_pred.add_argument(
        "--start-webviz",
        action="store_true",
        help="Start webviz automatically. This flag has no effect if also --skip-postprocessing is used",
    )

    parser_pred.set_defaults(func=flownet_pred)

    # Add hyperparameter tuning/sensitivity checks:
    parser_hyperparam = subparsers.add_parser(
        "hyperparam",
        help="Run flownet in hyperparameter exploration or optimization mode.",
    )

    parser_hyperparam.set_defaults(func=flownet_hyperparam)

    parser_hyperparam.add_argument(
        "config",
        type=pathlib.Path,
        help="Configuration file with hyperparameter ranges to use.",
    )
    parser_hyperparam.add_argument(
        "output_folder",
        type=pathlib.Path,
        help="Folder to hyperparameter exploration or optimization results.",
    )
    parser_hyperparam.add_argument(
        "--update-config",
        type=pathlib.Path,
        default=None,
        help="Optional configuration file which values will update the main config.",
    )
    parser_hyperparam.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists",
    )

    args = parser.parse_args()

    if len(sys.argv) <= 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
