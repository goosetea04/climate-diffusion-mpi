from __future__ import annotations
from dataclasses import dataclass
from typing import List
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

    Parameters
    ----------
    nlon : int
        Total number of longitude points.
    comm : MPI.Comm

    Returns
    -------
    Decomposition
        starts[r], counts[r] define the slice for rank r.
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