import tempfile
import pathlib
from typing import Union, Optional

import cwrap
import pandas as pd

from ..network_model._create_egrid import construct_kw


def write_grdecl_file(
    df_prop: pd.DataFrame,
    column_name: str,
    filename: Optional[Union[pathlib.Path, str]] = None,
    int_type: bool = False,
) -> Optional[str]:
    """
    Writes a list of values as a Flow grid file. The values need to be ordered in the
    "Flow way" (e.g. looping over X then Y then Z).

    Args:
        df_prop: A dataframe with as many rows as there are grid cells.
        column_name: Column in the dataframe to use as values.
        filename: Output filename to write to (e.g. "poro.grdecl"). If not given, return output as string.
        int_type: If the output file is to use integers (if False, save as floats).

    Returns:
        Output as string or None when output is written to a file

    """
    values = (
        df_prop[column_name]
        .astype(int if int_type else float)
        .values.flatten()
        .tolist()
    )

    outputfilename: Union[pathlib.Path, str] = ""

    if filename is None:
        _, outputfilename = tempfile.mkstemp()
    else:
        outputfilename = filename

    with cwrap.open(str(outputfilename), "w") as fh:
        construct_kw(column_name, values, int_type=int_type).write_grdecl(fh)

    if filename is None:
        outputfile = pathlib.Path(outputfilename)
        content = outputfile.read_text(encoding="utf8")
        outputfile.unlink()
        return content
    return None
