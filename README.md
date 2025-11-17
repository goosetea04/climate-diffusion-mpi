# Mini Climate Model (MPI Parallel)

This project implements a **parallel 2D climate diffusion model** using Python, NumPy, `mpi4py`, and optional NetCDF output.  
It demonstrates how surface temperature fields can be evolved over time using a simple physical scheme and MPI-based domain decomposition.

## What Problem It Solves

Climate and Earth system models rely on **parallel numerical computation** to evolve large grids of physical variables.  
This project provides a **minimal, understandable example** of that workflow:

- Represents Earth as a latitude–longitude grid  
- Evolves temperature using diffusion + simple radiative forcing  
- Splits the longitude dimension across MPI processes  
- Gathers the final global field on rank 0  
- Outputs results in both `.npy` and NetCDF formats  

It shows how HPC and climate modelling techniques intersect at a fundamental level.

## What is NetCDF?

**NetCDF** (Network Common Data Form) is the **standard file format** used in climate, weather, ocean, and environmental modelling.

It is designed for:

- **Large, multi-dimensional arrays** (e.g., temperature(lat, lon, time))  
- **Self-describing** scientific datasets (metadata included in the file)  
- **Portability across supercomputers and HPC systems**  
- Integration with tools like **xarray**, MATLAB, Panoply, and climate workflows  

Your output file `T_final.nc` contains:

- `lat(lat)`
- `lon(lon)`
- `temperature(lat, lon)`

This makes the model output compatible with the same tools used for real climate model diagnostics.

## How to Run

### 1. Install dependencies

```bash
pip install -e .
pip install netCDF4
You also need an MPI implementation (MS-MPI, MPICH, OpenMPI).
```
### 2. Run the model
Example with 4 MPI processes:

```mpiexec -n 4 python -m climate_mpi.cli --nlat 64 --nlon 128 --nt 300```

### 3. Outputs
Rank 0 writes two files:

```T_final.npy``` — raw NumPy array of shape (nlat, nlon)

```T_final.nc``` — NetCDF file with lat, lon, and temperature variables

These represent the final temperature field after the simulation.

### Expected Result
- The model produces a smoothed temperature distribution:
- warm near the equator
- cold near the poles
- gradually warming everywhere due to forcing
- realistic diffusion-driven smoothing over time
- The output is small but structurally similar to real climate model diagnostics.

## License
MIT License.