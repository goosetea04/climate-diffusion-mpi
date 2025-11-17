from __future__ import annotations

import numpy as np


def create_lat_lon_grid(nlat: int, nlon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create 1D latitude and longitude arrays.

    Parameters
    ----------
    nlat : int
        Number of latitude points (from -90 to 90 inclusive).
    nlon : int
        Number of longitude points (from 0 to 360, exclusive).

    Returns
    -------
    lat : np.ndarray
        Array of shape (nlat,) with latitude in degrees.
    lon : np.ndarray
        Array of shape (nlon,) with longitude in degrees.
    """
    if nlat <= 1 or nlon <= 1:
        raise ValueError("nlat and nlon must be > 1")

    lat = np.linspace(-90.0, 90.0, nlat)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    return lat, lon


def initial_temperature_field(nlat: int, nlon: int) -> np.ndarray:
    """
    Construct a simple initial temperature field.

    T(lat) = T0 - alpha * |lat|
    (warm equator, cold poles), repeated zonally.

    Parameters
    ----------
    nlat : int
    nlon : int

    Returns
    -------
    field : np.ndarray
        Array of shape (nlat, nlon) in Kelvin.
    """
    lat, _ = create_lat_lon_grid(nlat, nlon)
    T0 = 288.0  # ~15°C
    alpha = 0.5  # K per degree latitude (toy value)

    t_lat = T0 - alpha * np.abs(lat)
    field = np.repeat(t_lat[:, None], nlon, axis=1)
    return field.astype("float64")