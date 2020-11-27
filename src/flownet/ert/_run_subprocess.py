import glob
import pathlib
import subprocess
import signal

import psutil

TIMEOUT = 900  # Kill ERT if no new output to stdout for 15 minutes.


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

        def _handler(*args):  # pylint: disable=unused-argument
            main_proc = psutil.Process(process.pid)
            for child_proc in main_proc.children(recursive=True):
                child_proc.kill()
            main_proc.kill()

            raise subprocess.SubprocessError(
                f"The ERT process has not returned any output for {TIMEOUT} seconds.\n"
                "FlowNet assumes that something fishy has happened and will kill\n"
                "ERT and all suprocesses. Check the logs for details."
            )

        signal.signal(signal.SIGALRM, _handler)

        for line in process.stdout:  # type: ignore
            signal.alarm(TIMEOUT)

            print(line, end="")
            if (
                "active realisations left, which is less than "
                "the minimum specified - stopping assimilation." in line
                or "All realizations failed!" in line
            ):
                process.terminate()
                error_files = glob.glob(str(cwd / runpath.replace("%d", "*") / "ERROR"))
                raise subprocess.SubprocessError(
                    pathlib.Path(error_files[0]).read_text()
                )

    if process.returncode != 0:
        raise subprocess.SubprocessError(
            "The ERT workflow failed. Check the ERT log for more details."
        )
