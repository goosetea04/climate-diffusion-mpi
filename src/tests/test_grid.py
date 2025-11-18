import numpy as np
from climate_mpi.grid import create_lat_lon_grid, initial_temperature_field
def test_create_lat_lon_grid_shapes():
    nlat, nlon = 10, 20
    lat, lon = create_lat_lon_grid(nlat, nlon)

    assert lat.shape == (nlat,)
    assert lon.shape == (nlon,)


def test_initial_temperature_field_shape_and_equator_warmest():
    nlat, nlon = 9, 16
    T = initial_temperature_field(nlat, nlon)

    assert isinstance(T, np.ndarray)
    assert T.shape == (nlat, nlon)

    # Middle latitude (equator) should be the warmest band
    mid = nlat // 2
    equator_row = T[mid, :]
    assert np.allclose(equator_row, equator_row[0])  # zonally uniform
    assert equator_row[0] == T.max()