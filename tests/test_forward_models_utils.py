import os
import shutil

from flownet.ert.forward_models.utils import get_last_iteration


def generate_example_runpath_directory(
    runpath: str, num_realizations: int, _numiterations: int
) -> None:
    for i in range(0, num_realizations):
        realization_folder = "-".join(("realization", str(i)))
        for j in range(0, num_iterations):
            iteration_folder = "-".join(("iter", str(j)))
            os.makedirs(os.path.join(runpath, realization_folder, iteration_folder))


def remove_example_runpath_directory(runpath: str) -> None:
    shutil.rmtree(runpath)


def test_get_last_iteration() -> None:
    path = "./tests/data/runpath/realization-*/iter-*"
    runpath = "./tests/data/runpath"
    num_realizations = 11
    num_iterations = 11

    generate_example_runpath_directory(runpath, num_realizations, num_iterations)

    last_iteration, runpath_list = get_last_iteration(path)

    remove_example_runpath_directory(runpath)

    assert last_iteration == (num_iterations - 1)
    assert len(runpath_list) == realizations
