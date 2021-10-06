import glob
import pathlib
import subprocess
import signal

import psutil


def run_ert_subprocess(
    command: str, cwd: pathlib.Path, runpath: str, timeout: int
) -> None:
    """
    Helper function to run a ERT setup.

    Should revert here to use the much simpler subprocess.run when
    https://github.com/equinor/libres/issues/984 is closed. See
    https://github.com/equinor/flownet/pull/119, and
    https://github.com/equinor/flownet/pull/271,
    on possible changes to revert.

    Args:
        command: Command to run.
        cwd: The folder to run the command from.
        runpath: Runpath variable given to ERT.
        timeout: inactivity time out for killing FlowNet.

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
                f"The ERT process has not returned any output for {timeout} seconds.\n"
                "FlowNet assumes that something fishy has happened and will kill\n"
                "ERT and all suprocesses. Check the logs for details."
            )

        signal.signal(signal.SIGALRM, _handler)

        for line in process.stdout:  # type: ignore
            signal.alarm(timeout)

            print(line, end="")
            if (
                "active realisations left, which is less than "
                "the minimum specified - stopping assimilation." in line
                or "All realizations failed!" in line
            ):
                process.terminate()
                error_files = glob.glob(str(cwd / runpath.replace("%d", "*") / "ERROR"))
                raise subprocess.SubprocessError(
                    pathlib.Path(error_files[0]).read_text(encoding="utf8")
                )

    if process.returncode != 0:
        raise subprocess.SubprocessError(
            "The ERT workflow failed. Check the ERT log for more details."
        )
