from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
from mpi4py import MPI

@dataclass(frozen=True)
class Decomposition:
    """
    Description of a 1D domain decomposition along longitude.

    Attributes
    ----------
    starts : list[int]
        Starting column index (inclusive) for each rank.
    counts : list[int]
        Number of columns owned by each rank.
    """
    starts: list[int]
    counts: list[int]


def compute_decomposition(nlon: int, comm: MPI.Comm) -> Decomposition:
    """
    Compute a 1D decomposition of the longitude dimension across ranks.

    We distribute columns as evenly as possible, with some ranks getting
    one extra column if nlon is not divisible by size.
    """
    size = comm.Get_size()

    base = nlon // size
    extra = nlon % size

    starts: List[int] = []
    counts: List[int] = []
    current_start = 0

    for r in range(size):
        local_nlon = base + (1 if r < extra else 0)
        starts.append(current_start)
        counts.append(local_nlon)
        current_start += local_nlon

    assert sum(counts) == nlon
    return Decomposition(starts=starts, counts=counts)


def exchange_halos(T_local: np.ndarray, comm: MPI.Comm) -> np.ndarray:
    """
    Exchange one-column halos in longitude between neighbouring ranks.

    Parameters
    ----------
    T_local : np.ndarray
        Local temperature field of shape (nlat, local_nlon). This is the
        portion of the global grid owned by this rank.

    Returns
    -------
    T_ext : np.ndarray
        Extended array of shape (nlat, local_nlon + 2), where:
        - column 0   : halo column from left neighbour
        - columns 1:-1: original T_local
        - column -1 : halo column from right neighbour
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    nlat, local_nlon = T_local.shape

    left = (rank - 1) % size
    right = (rank + 1) % size

    # Allocate halos
    recv_left = np.empty(nlat, dtype=T_local.dtype)
    recv_right = np.empty(nlat, dtype=T_local.dtype)

    # Send rightmost column to right, receive left halo from left
    send_right = T_local[:, -1].copy()
    comm.Sendrecv(
        sendbuf=send_right,
        dest=right,
        sendtag=0,
        recvbuf=recv_left,
        source=left,
        recvtag=0,
    )

    # Send leftmost column to left, receive right halo from right
    send_left = T_local[:, 0].copy()
    comm.Sendrecv(
        sendbuf=send_left,
        dest=left,
        sendtag=1,
        recvbuf=recv_right,
        source=right,
        recvtag=1,
    )

    # Build extended array
    T_ext = np.empty((nlat, local_nlon + 2), dtype=T_local.dtype)
    T_ext[:, 0] = recv_left
    T_ext[:, 1:-1] = T_local
    T_ext[:, -1] = recv_right

    return T_ext
