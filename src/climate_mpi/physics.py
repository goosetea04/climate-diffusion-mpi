from __future__ import annotations
import numpy as np

def diffusion_step(
    T: np.ndarray,
    kappa: float,
    dt: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Original diffusion function (no halos), still used in tests or
    serial runs.
    """
    if T.ndim != 2:
        raise ValueError("T must be a 2D array (nlat, nlon)")

    nlat, nlon = T.shape
    T_new = T.copy()

    # latitude indices with reflecting boundaries
    ip = np.arange(nlat) + 1
    im = np.arange(nlat) - 1
    ip[-1] = nlat - 1
    im[0] = 0

    # longitude indices with periodic boundaries
    jp = np.arange(nlon) + 1
    jm = np.arange(nlon) - 1
    jp[-1] = 0
    jm[0] = nlon - 1

    T_center = T
    T_north = T[ip, :]
    T_south = T[im, :]
    T_east = T[:, jp]
    T_west = T[:, jm]

    laplacian = ((T_north - 2.0 * T_center + T_south) / dy**2 +
                 (T_east - 2.0 * T_center + T_west) / dx**2)

    T_new += kappa * dt * laplacian
    return T_new


def diffusion_step_with_halos(
    T_ext: np.ndarray,
    kappa: float,
    dt: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Diffusion step for a local domain with halo columns.

    Parameters
    ----------
    T_ext : np.ndarray
        Temperature field of shape (nlat, local_nlon + 2), where
        column 0 and -1 are halo columns from neighbour ranks.
    kappa, dt, dx, dy : float
        Same physical parameters as diffusion_step.

    Returns
    -------
    T_new_local : np.ndarray
        Updated local field of shape (nlat, local_nlon), i.e. without halos.
    """
    if T_ext.ndim != 2:
        raise ValueError("T_ext must be a 2D array (nlat, local_nlon+2)")

    nlat, nlon_ext = T_ext.shape
    local_nlon = nlon_ext - 2

    # interior (local) columns are 1..-2
    T_center = T_ext[:, 1:-1] 

    # latitude indices with reflecting boundaries
    ip = np.arange(nlat) + 1
    im = np.arange(nlat) - 1
    ip[-1] = nlat - 1
    im[0] = 0

    # neighbours in latitude
    T_north = T_ext[ip, 1:-1]
    T_south = T_ext[im, 1:-1]

    # neighbours in longitude using halos
    T_west = T_ext[:, 0:-2] # west neighbour: columns 0..local_nlon-1 (includes left halo)
    T_east = T_ext[:, 2:] # east neighbour: columns 2..local_nlon+1 (includes right halo)

    laplacian = ((T_north - 2.0 * T_center + T_south) / dy**2 +
                 (T_east - 2.0 * T_center + T_west) / dx**2)

    T_new_local = T_center + kappa * dt * laplacian
    return T_new_local

def simple_radiative_forcing(T: np.ndarray, dt: float) -> np.ndarray:
    return T + 0.01 * dt
