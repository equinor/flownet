import glob
import pathlib
import subprocess


def run_ert_subprocess(command: str, cwd: pathlib.Path, runpath: str) -> None:
    """
    Helper function to run a ERT setup.

    Should revert here to use the much simpler subprocess.run when
    https://github.com/equinor/libres/issues/984 is closed. See
    https://github.com/equinor/flownet/pull/119 on changes to revert.

    Args:
        command: Command to run.
        cwd: The folder to run the command from.
        runpath: Runpath variable given to ERT.

    Returns:
        Nothing

    """

    with subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    ) as process:
        for line in process.stdout:  # type: ignore
            print(line, end="")
            if (
                "active realisations left, which is less than "
                "the minimum specified - stopping assimilation." in line
                or "All realizations failed!" in line
            ):
                process.terminate()
                error_files = glob.glob(str(cwd / runpath.replace("%d", "*") / "ERROR"))
                raise RuntimeError(pathlib.Path(error_files[0]).read_text())
