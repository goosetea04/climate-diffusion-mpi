import numpy as np
from climate_mpi.physics import diffusion_step, simple_radiative_forcing

def test_diffusion_step_smooths_sharp_gradient():
    nlat, nlon = 10, 10
    T = np.zeros((nlat, nlon))
    T[:, -1] = 100.0  # sharp gradient on the right boundary

    T_new = diffusion_step(T, kappa=1.0, dt=0.1, dx=1.0, dy=1.0)

    assert T_new.max() < T.max() # Max should go down due to diffusion
    assert T_new[:, -2].mean() > 0.0 # Neighbouring column should have warmed up


def test_simple_radiative_forcing_warms_field():
    T = np.zeros((5, 5))
    T_new = simple_radiative_forcing(T, dt=1.0)

    assert np.all(T_new > T)
