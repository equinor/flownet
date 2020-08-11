import logging
from subprocess import check_output

import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    name = req.params.get("name")
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get("name")

    if name:
        output = check_output(["/opm-simulators/build/bin/flow", "--version"])

        return func.HttpResponse(f"<p>{name}</p><p>{output}</p>")
    else:
        return func.HttpResponse(
            "Please pass a name on the query string or in the request body",
            status_code=400,
        )
