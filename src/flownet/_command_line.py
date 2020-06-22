import argparse
import shutil
import pathlib
import subprocess

from .config_parser import parse_config, parse_pred_config
from .ahm import run_esmda
from .ahm import run_hyperopt
from .prediction import run_flownet_prediction


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


def flownet_ahm_esmda(args: argparse.Namespace) -> None:
    """
    Entrypoint for running flownet in AHM mode using ES-MDA.

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

    config = parse_config(args.config)
    run_esmda(config, args)

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

    config = parse_pred_config(args.config)
    run_flownet_prediction(config, args)

    if not args.skip_postprocessing:
        create_webviz(args.output_folder, start_webviz=args.start_webviz)


def flownet_ahm_hyperopt(args: argparse.Namespace) -> None:
    """
    Entrypoint for running flownet in AHM mode using hyperopt.

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

    config = parse_config(args.config)
    run_hyperopt(config, args)

    if not args.skip_postprocessing:
        create_webviz(args.output_folder, start_webviz=args.start_webviz)


def flownet_hyperparam(args: argparse.Namespace):
    raise NotImplementedError


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

    subparsers_ahm = parser_ahm.add_subparsers(
        help="The options available. "
        'Type e.g. "flownet --help" '
        "to get help on that particular "
        "option."
    )

    # Add assisted history matching esmda argument parser:
    parser_esmda = subparsers_ahm.add_parser(
        "esmda",
        help="Run flownet in an assisted history matching setting using ES-MDA via ERT.",
    )

    parser_esmda.add_argument(
        "config", type=pathlib.Path, help="Configuration file with AHM settings to use."
    )
    parser_esmda.add_argument(
        "output_folder", type=pathlib.Path, help="Folder to store AHM output."
    )
    parser_esmda.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists",
    )
    parser_esmda.add_argument(
        "--skip-postprocessing",
        action="store_true",
        help="Do not run any postprocessing after ERT is finished",
    )
    parser_esmda.add_argument(
        "--start-webviz",
        action="store_true",
        help="Start webviz automatically. This flag has no effect if also --skip-postprocessing is used",
    )
    parser_esmda.add_argument(
        "--debug",
        action="store_true",
        help="Keeps all files generated by ERT for easier inspection of results during debugging",
    )

    parser_esmda.set_defaults(func=flownet_ahm_esmda)

    # Add assisted history matching hyperopt argument parser:
    parser_hyperopt = subparsers_ahm.add_parser(
        "hyperopt",
        help="Run flownet in an assisted history matching setting using a direct hyperopt approach.",
    )

    parser_hyperopt.add_argument(
        "config", type=pathlib.Path, help="Configuration file with AHM settings to use."
    )
    parser_hyperopt.add_argument(
        "output_folder", type=pathlib.Path, help="Folder to store AHM output."
    )
    parser_hyperopt.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists",
    )
    parser_hyperopt.add_argument(
        "--skip-postprocessing",
        action="store_true",
        help="Do not run any postprocessing after ERT is finished",
    )
    parser_hyperopt.add_argument(
        "--start-webviz",
        action="store_true",
        help="Start webviz automatically. This flag has no effect if also --skip-postprocessing is used",
    )
    parser_hyperopt.add_argument(
        "--debug",
        action="store_true",
        help="Keeps all files generated by ERT for easier inspection of results during debugging",
    )

    parser_hyperopt.set_defaults(func=flownet_ahm_hyperopt)

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

    # Add hyper parameter tuning/sensitivity checks:
    parser_hyperparam = subparsers.add_parser(
        "hyperparam", help="Run flownet in prediction based mode."
    )

    parser_hyperparam.set_defaults(func=flownet_hyperparam)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
