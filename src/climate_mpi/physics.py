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
    Single explicit 2D diffusion step using a 5-point stencil.

    T_new = T + kappa * dt * Laplacian(T)

    Boundary conditions:
    - Latitude: reflecting (zero flux across poles)
    - Longitude: periodic

    Parameters
    ----------
    T : np.ndarray
        Temperature field of shape (nlat, nlon).
    kappa : float
        Diffusion coefficient.
    dt : float
        Time step.
    dx : float
        Grid spacing in x (lon) direction (arbitrary units).
    dy : float
        Grid spacing in y (lat) direction.

    Returns
    -------
    T_new : np.ndarray
        Updated temperature field with the same shape as T.
    """
    if T.ndim != 2:
        raise ValueError("T must be a 2D array (nlat, nlon)")

    nlat, nlon = T.shape
    T_new = T.copy()

    # latitude indices with reflecting boundaries
    ip = np.arange(nlat) + 1
    im = np.arange(nlat) - 1
    ip[-1] = nlat - 1  # clamp at south pole
    im[0] = 0          # clamp at north pole

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


def simple_radiative_forcing(T: np.ndarray, dt: float) -> np.ndarray:
    """
    Very simple uniform warming forcing term.

    Adds a small linear warming tendency everywhere.

    Parameters
    ----------
    T : np.ndarray
        Temperature field (K).
    dt : float
        Time step (arbitrary).

    Returns
    -------
    np.ndarray
        Updated field.
    """
    return T + 0.01 * dt  # toy: 0.01 K per unit time