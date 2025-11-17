from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

from .driver import run_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mini parallel climate diffusion model (toy)."
    )
    parser.add_argument("--nlat", type=int, default=64, help="Number of latitude points.")
    parser.add_argument("--nlon", type=int, default=128, help="Number of longitude points.")
    parser.add_argument("--nt", type=int, default=200, help="Number of time steps.")
    parser.add_argument(
        "--kappa", type=float, default=1.0, help="Diffusion coefficient (toy units)."
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Time step size.")
    parser.add_argument(
        "--output",
        type=str,
        default="T_final.npy",
        help="Output filename for final temperature field (NumPy .npy).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # basic validation (only rank 0 prints errors)
    if args.nlat <= 1 or args.nlon <= 1:
        if rank == 0:
            raise SystemExit("nlat and nlon must be > 1")
        else:
            return

    if args.nt <= 0:
        if rank == 0:
            raise SystemExit("nt must be > 0")
        else:
            return

    T_final = run_simulation(
        nlat=args.nlat,
        nlon=args.nlon,
        nt=args.nt,
        kappa=args.kappa,
        dt=args.dt,
        dx=1.0,
        dy=1.0,
    )

    # Only rank 0 writes results
    if rank == 0 and T_final is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, T_final)
        print(f"[rank 0] Saved final temperature field to {out_path}")


if __name__ == "__main__":
    main()