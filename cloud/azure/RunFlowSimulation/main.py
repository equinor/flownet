import traceback
import uuid
import shutil
import os
import pathlib
from subprocess import check_output
import mimetypes

import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:

    unique_folder = pathlib.Path("/tmp/" + str(uuid.uuid4()))
    os.mkdir(unique_folder)

    try:

        ecl_base = pathlib.Path(req.form["ecl_base"])

        for input_file in req.files.values():
            filename = unique_folder / pathlib.Path(input_file.filename)
            contents = input_file.stream.read()
            open(filename, "wb").write(contents)
            out_file = unique_folder / "results.tar.gz"

            # Unzip
            check_output(["tar", "-xzf", str(filename), "-C", str(unique_folder)])

            # Run Flow
            check_output(
                [
                    "/opm-simulators/build/bin/flow",
                    str(
                        unique_folder / ecl_base.parents[0] / "FLOWNET_REALIZATION.DATA"
                    ),
                ]
            )

            # Pack Results
            check_output(
                [
                    "tar",
                    "-czvf",
                    str(out_file),
                    "-C",
                    str(unique_folder / ecl_base.parents[0]),
                    "FLOWNET_REALIZATION.SMSPEC",
                    "FLOWNET_REALIZATION.UNRST",
                    "FLOWNET_REALIZATION.UNSMRY",
                ]
            )

            # Load result file in memory
            with open(out_file, "rb") as fh:
                binary_output = fh.read()

            # Remove temporary files
            shutil.rmtree(unique_folder)

            # Send back to client
            return func.HttpResponse(
                binary_output, mimetype=mimetypes.guess_type(str(out_file))[0]
            )

        return func.HttpResponse(
            f"The FlowNet Cloud Engine request for '{ecl_base.name}' could not be handled.",
            status_code=400,
        )

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
