import logging
import pathlib
from subprocess import check_output
import mimetypes

import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    ecl_base = req.form["ecl_base"]
    if not ecl_base:
        try:
            req_body = req.get_json()
        except ValueError:
            return func.HttpResponse(
                f"Please specify an 'ecl_base' value in your request.", status_code=400,
            )
        else:
            ecl_base = req_body.get("ecl_base")

    for input_file in req.files.values():
        filename = input_file.filename
        contents = input_file.stream.read()
        open(filename, "wb").write(contents)
        out_file = "results.tar.gz"

        # Unzip
        check_output(["tar", "-xzf", filename])

        # Run Flow
        check_output(
            [
                "/opm-simulators/build/bin/flow",
                str(pathlib.Path(ecl_base).parents[0] / "FLOWNET_REALIZATION.DATA"),
            ]
        )

        # Pack Results
        check_output(
            [
                "tar",
                "-czvf",
                out_file,
                "-C",
                pathlib.Path(ecl_base).parents[0],
                "FLOWNET_REALIZATION.SMSPEC",
                "FLOWNET_REALIZATION.UNRST",
                "FLOWNET_REALIZATION.UNSMRY",
            ]
        )

        # Send back to browser
        with open(out_file, "rb") as f:
            mimetype = mimetypes.guess_type(out_file)
            return func.HttpResponse(f.read(), mimetype=mimetype[0])

    return func.HttpResponse(
        f"The FlowNet Cloud Engine request for '{ecl_base}' could not be handled.",
        status_code=400,
    )
