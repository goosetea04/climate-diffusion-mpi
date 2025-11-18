import numpy as np
import pytest
from mpi4py import MPI
from climate_mpi.parallel import compute_decomposition, exchange_halos


def test_compute_decomposition_single_rank():
    """
    With a single MPI rank, the decomposition should give the full domain
    to rank 0.
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    if size != 1:
        pytest.skip("This test is intended for size == 1 runs without mpiexec.")

    nlon = 17
    decomp = compute_decomposition(nlon, comm)

    assert len(decomp.starts) == 1
    assert len(decomp.counts) == 1
    assert decomp.starts[0] == 0
    assert decomp.counts[0] == nlon


def test_exchange_halos_single_rank_self_consistent():
    """
    For size == 1, exchange_halos should wrap around to itself:
    - left halo column == last column
    - right halo column == first column
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    if size != 1:
        pytest.skip("This test is intended for size == 1 runs without mpiexec.")
    T_local = np.array([[0.0, 1.0, 2.0], # Construct a simple local field with easily recognisable columns
                        [10.0, 11.0, 12.0]])

    T_ext = exchange_halos(T_local, comm)

    assert T_ext.shape == (2, 5)
    np.testing.assert_allclose(T_ext[:, 0], T_local[:, -1])
    np.testing.assert_allclose(T_ext[:, 1:4], T_local)
    np.testing.assert_allclose(T_ext[:, 4], T_local[:, 0])
