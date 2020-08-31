from typing import Optional, Dict

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pykrige import UniversalKriging3D


def execute(
    data: np.ndarray,
    n: int = 20,
    variogram_model: str = "spherical",
    variogram_parameters: Optional[Dict] = None,
    nlags: int = 6,
    anisotropy_scaling_z: float = 10.0,
    **kwargs
) -> RegularGridInterpolator:
    """
    Executes spatial kriging of input data and returns an scipy.interpolate.RegularInterpolator object.

    Returns:
        RegularInterpolator for mthe kriged 3D volume

    """
    if variogram_parameters is None:
        variogram_parameters = {"sill": 0.75, "range": 1000, "nugget": 0}

    gridx = np.arange(
        data[:, 0].min(), data[:, 0].max(), (data[:, 0].max() - data[:, 0].min()) / n,
    )
    gridy = np.arange(
        data[:, 1].min(), data[:, 1].max(), (data[:, 1].max() - data[:, 1].min()) / n,
    )
    gridz = np.arange(
        data[:, 2].min(), data[:, 2].max(), (data[:, 2].max() - data[:, 2].min()) / n,
    )

    uk3d = UniversalKriging3D(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        np.log10(data[:, 3]),
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        nlags=nlags,
        anisotropy_scaling_z=anisotropy_scaling_z,
        **kwargs
    )

    k3d3, ss3d = uk3d.execute("grid", gridx, gridy, gridz)

    k3d3_interpolator = RegularGridInterpolator(
        (gridx, gridy, gridz), k3d3, bounds_error=False
    )
    ss3d_interpolator = RegularGridInterpolator(
        (gridx, gridy, gridz), ss3d, bounds_error=False
    )

    return k3d3_interpolator, ss3d_interpolator
