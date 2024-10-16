# Efficient Stochastic Simulation of Epidemics

This repository contains a lean implementation of Gillespie's algorithm for simulating epidemic spread through networks using the SIS, SIR, and SI (if recovery rate = 0) models. The main file can be found in the main branch, and the code is optimized for performance by exploiting the sparsity of graphs and using incremental updates for infection and recovery rates.

## New in this Version

- **Gillespie’s Algorithm**: Implemented for SIS, SIR, and SI models.
- **Optimized for Sparse Networks**: The implementation takes advantage of network sparsity, making it much faster when the number of infected nodes is small.
- **Incremental Rate Updates**: Infection and recovery rates are updated incrementally, avoiding redundant calculations and improving efficiency.
- **Exact Timing via Exponential Distributions**: Both infection and recovery times follow exponential distributions, adhering to the exact timing mechanics without any workarounds.

### Limitations

- **Numba Integration**: Numba optimization was not successful, so the code contains loops and does not leverage vectorized computations as in previous versions.
- **Sorted List Optimization**: Integration of sorted lists for managing infection weights was not included due to issues with updating weights efficiently. This might still offer potential improvement, but it does not affect the worst-case time complexity.

## Features

- **Graph-Based Simulation**: Uses NetworkX to handle complex network structures.
- **Efficient Performance**: By leveraging the sparsity of graphs and maintaining incremental updates, the simulation avoids repeated computations.
- **Exponential Time Dynamics**: The timing for both infection and recovery events follows exponential distributions, giving an exact stochastic simulation.

## Installation

Before running the simulation, install the required Python libraries using `pip`:

```bash
pip install networkx numpy
