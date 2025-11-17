# Mini Climate Model

A tiny climate model with big ambitions.

This project is a parallel 2D surface temperature diffusion model built with Python, NumPy, and mpi4py. It simulates a very simplified Earth: warm at the equator, cold at the poles, and a little bit of physics in between.

Think of it as a "hello world" for climate modelling + high-performance computing. Not a full Earth System Model, but it does demonstrate the core ideas used in the real ones:

- domain decomposition
- MPI parallelism
- diffusion-based energy transport
- toy radiative forcing
- scientific code structure
- clean code & reproducibility

…and it's actually fun to play with.

## What Does This Model Actually Do?

This mini-model represents the Earth's surface on a latitude × longitude grid:
```
lat: -90° … 90°      (nlat points)
lon: 0° … 360°       (nlon points)
```

Every grid point has a temperature `T(lat, lon)`. We evolve this temperature field forward in time using:

### 1. Horizontal Diffusion

A classic 5-point Laplacian stencil (north/south/east/west neighbours), which smooths out sharp gradients—similar to how energy diffuses in real climate systems.

### 2. Simple Radiative Forcing

A tiny uniform warming term ("Earth is heating up, sorry").

### 3. Boundary Conditions

- **Periodic in longitude** (wrap-around Earth)
- **Reflecting in latitude** (energy doesn't cross the poles)

### 4. Parallel Execution with MPI

The longitude dimension is split across MPI ranks. Each rank evolves its "slice" of the planet, then we `MPI_Gatherv` the pieces back together.

Domain decomposition looks like this:
```
Global grid      | Rank 0 | Rank 1 | Rank 2 | Rank 3 |
-----------------------------------------------
<---  nlon columns divided evenly across ranks  --->
```

Every rank does its part. Rank 0 collects the finished world.

## Why Build This?

Three reasons:

1. To showcase core climate-model concepts without needing a supercomputer.
2. To demonstrate real HPC patterns (domain decomposition, collectives, mpi4py).
3. To write a tidy, well-structured scientific codebase, the kind RSE teams expect.

Basically:

> "What if I built something small enough to run on a laptop but structured like something you'd find running on a cluster?"

This is that.

## Project Structure
```
mini-climate-mpi/
  src/climate_mpi/
    grid.py        # latitude/longitude grids & initial conditions
    physics.py     # diffusion + forcing
    parallel.py    # domain decomposition logic
    driver.py      # time-stepping loop, MPI communication
    cli.py         # command-line interface
  tests/
    test_grid.py   # basic grid tests
    test_physics.py
```

Everything is modular, typed, documented, and testable.

## Installation

You need Python 3.11+ and an MPI implementation (MS-MPI on Windows, OpenMPI/MPICH on Linux/macOS).
```bash
pip install -e .
```

(Optional) run the tests:
```bash
pytest
```

## Running the Model

Example: 4 processes, 64×128 grid, 300 timesteps:
```bash
mpiexec -n 4 python -m climate_mpi.cli --nlat 64 --nlon 128 --nt 300
```

Rank 0 will save the final global temperature field:
```
T_final.npy
```

Load and inspect it in Python:
```python
import numpy as np
T = np.load("T_final.npy")
print(T.shape, T.mean(), T.min(), T.max())
```

Plot it however you want—matplotlib, xarray, or your favourite tool.

## Example Output

After ~300 steps, you'll typically see:

- smoother temperature gradients
- gradual warming everywhere
- poles still cold-ish
- equator still warm-ish
- the planet looking a bit more "blurry" (diffusion does that)

Not physically accurate, but recognisably climate-model-ish.

## 🧬 Key Features (in human words)

- **Parallel toy climate physics**: The bare-bones mechanics behind real models.
- **MPI made friendly**: Clean patterns for broadcast, scatter/gather, and array slicing.
- **Code you'd actually want to maintain**: modular, documented, tested.
- **Educational**: great starting point for climate/HPC newcomers.
- **Extensible**: easy to add:
  - halo exchanges
  - variable diffusivity
  - atmospheric layers
  - ocean/land masks
  - real radiative forcing
  - netCDF I/O

You can take this surprisingly far.

## 💡 Future Extensions

If I had more time (or caffeine), I'd add:

- halo exchange for correct inter-rank stencils
- OpenMP or numba acceleration
- actual netCDF output
- equatorially enhanced solar forcing
- multi-layer temperature profiles
- simple atmospheric circulation toy model
- GPU backend (CuPy)

## 🧑‍💻 About This Project

I built this to demonstrate:

- knowledge of climate/earth-system model structure
- experience with high-performance / parallel computing
- comfort with scientific software engineering
- ability to write clean, maintainable RSE-grade code

It's intentionally small. But the patterns scale—just like real climate models started small once.

## 🌱 License

MIT — feel free to fork, extend, teach with it, or run it on a cluster.