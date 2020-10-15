from typing import Optional, Dict, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pykrige import UniversalKriging3D


def execute(
    data: np.ndarray,
    n: int = 20,
    variogram_model: str = "spherical",
    variogram_parameters: Optional[Dict] = None,
    n_lags: int = 6,
    anisotropy_scaling_z: float = 10.0,
    **kwargs
) -> Tuple[RegularGridInterpolator, RegularGridInterpolator]:
    """
    Executes spatial kriging of input data and returns an scipy.interpolate.RegularInterpolator object.

    Args:
        data: 3xN np.ndarray with columns X, Y, Z, MEASUREMENT
        n: Number of kriged values in each direct. E.g, n = 10 -> 10x10x10 = 1000 values
        variogram_model: Variogram options as allowed in pykrige
        nlags: nlag as defined in pykrige
        anisotropy_scaling_z: anisotropy in the z-scale
        ...

    Returns:
        Tuple of RegularInterpolators for the kriged measurement and uncertainty in a 3D volume

    """
    gridx = np.arange(
        data[:, 0].min(),
        data[:, 0].max(),
        (data[:, 0].max() - data[:, 0].min()) / n,
    )
    gridy = np.arange(
        data[:, 1].min(),
        data[:, 1].max(),
        (data[:, 1].max() - data[:, 1].min()) / n,
    )
    gridz = np.arange(
        data[:, 2].min(),
        data[:, 2].max(),
        (data[:, 2].max() - data[:, 2].min()) / n,
    )

    uk3d = UniversalKriging3D(
        *data.T,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        nlags=n_lags,
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
