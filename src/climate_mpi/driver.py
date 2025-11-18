from __future__ import annotations
import numpy as np
from mpi4py import MPI
from .grid import initial_temperature_field
from .physics import diffusion_step_with_halos, simple_radiative_forcing
from .parallel import compute_decomposition, exchange_halos


def run_simulation(
    nlat: int = 64,
    nlon: int = 128,
    nt: int = 100,
    kappa: float = 1.0,
    dt: float = 0.01,
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray | None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if nlat <= 1 or nlon <= 1:
        raise ValueError("nlat and nlon must be > 1")

    # 1. Global initial condition on rank 0
    if rank == 0:
        T_global = initial_temperature_field(nlat, nlon)
    else:
        T_global = None

    # 2. Broadcast initial field to all ranks
    T_global = comm.bcast(T_global, root=0)

    # 3. Decomposition along longitude
    decomp = compute_decomposition(nlon, comm)
    start = decomp.starts[rank]
    local_nlon = decomp.counts[rank]
    end = start + local_nlon

    # local slice
    T_local = T_global[:, start:end].copy()  # (nlat, local_nlon)

    # 4. Time integration loop with halo exchange
    for _ in range(nt):
        # get extended field with halos from neighbours
        T_ext = exchange_halos(T_local, comm)

        # perform diffusion using halos for correct stencils
        T_local = diffusion_step_with_halos(
            T_ext,
            kappa=kappa,
            dt=dt,
            dx=dx,
            dy=dy,
        )

        # apply forcing locally
        T_local = simple_radiative_forcing(T_local, dt=dt)

    # 5. Gather results back to rank 0 using Gatherv (unchanged)

    sendbuf = T_local.ravel()

    counts_cols = decomp.counts
    counts_el = [nlat * c for c in counts_cols]

    displs_cols = []
    running = 0
    for c in counts_cols:
        displs_cols.append(running)
        running += c
    displs_el = [nlat * d for d in displs_cols]

    if rank == 0:
        recvbuf = np.empty(nlat * nlon, dtype=T_local.dtype)
    else:
        recvbuf = None

    comm.Gatherv(
        sendbuf,
        (recvbuf, counts_el, displs_el, MPI.DOUBLE),
        root=0,
    )

    if rank == 0 and recvbuf is not None:
        T_final = recvbuf.reshape(nlat, nlon)
        return T_final

    return None
