import traceback
import shutil
import os
import pathlib
from subprocess import check_output
import mimetypes
import uuid

import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:

    # Create tmp storage location
    unique_folder = pathlib.Path("/tmp/" + str(uuid.uuid4()))
    os.mkdir(unique_folder)

    try:
        # Read function arguments
        pickled_network = pathlib.Path(unique_folder / req.form["pickled_network"])
        pickled_schedule = pathlib.Path(unique_folder / req.form["pickled_schedule"])
        output_folder = pathlib.Path(unique_folder / req.form["output_folder"])
        pickled_parameters = pathlib.Path(
            unique_folder / req.form["pickled_parameters"]
        )
        random_samples = pathlib.Path(unique_folder / req.form["random_samples"])
        realization_index = req.form["realization_index"]
        if (
            not req.form["pred_schedule_file"]
            or req.form["pred_schedule_file"] == "None"
        ):
            pred_schedule_file = "None"
        else:
            pred_schedule_file = str(
                pathlib.Path(unique_folder / req.form["pred_schedule_file"])
            )

        # Retrieve input files
        for input_file in req.files.values():
            filename = pathlib.Path(unique_folder / input_file.filename)
            contents = input_file.stream.read()
            open(filename, "wb").write(contents)

        # Render realization
        check_output(
            [
                "flownet_render_realization",
                str(pickled_network),
                str(pickled_schedule),
                str(pickled_parameters),
                str(output_folder),
                str(random_samples),
                str(realization_index),
                pred_schedule_file,
            ]
        )

        # Remove input files
        for input_file in req.files.values():
            filename = pathlib.Path(unique_folder / input_file.filename)
            filename.unlink()

        # Pack results
        out_file = pathlib.Path(unique_folder / "results.tar.gz")

        check_output(
            ["touch", str(out_file),]
        )

        check_output(
            [
                "tar",
                "-C",
                str(unique_folder),
                "-czvf",
                str(out_file),
                f"--exclude={out_file.name}",
                ".",
            ]
        )

        # Guess filetype
        mimetype = mimetypes.guess_type(str(out_file))[0]

        # Load results in memory
        with open(out_file, "rb") as fh:
            binary_output = fh.read()

        # Remove temporary files
        shutil.rmtree(unique_folder)

        # Send back to client
        return func.HttpResponse(binary_output, mimetype=mimetype)

    # pylint: disable=broad-except
    except Exception as err:
        try:
            shutil.rmtree(unique_folder)
        except FileNotFoundError:
            pass

        with open("log.txt", "a") as fh:
            fh.write(str(err))
            fh.write(traceback.format_exc())

        return func.HttpResponse(
            f"The RenderFlownetRealization Azure Function could not be executed.",
            status_code=500,
        )
