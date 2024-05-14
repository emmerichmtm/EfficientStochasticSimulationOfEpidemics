# Epidemic Simulation Using NetworkX

This project simulates the spread of an epidemic through a network using Python and NetworkX. The simulation models the time dynamics where the time until the next infection follows an exponential distribution with parameters based on the infection rates and the connectivity of the nodes.

## Features

- **Graph-Based Simulation**: Utilizes NetworkX to handle complex network structures.
- **Exponential Time Dynamics**: Models time until next infection using exponential distribution influenced by node connectivity.
- **Efficient Updates**: Incrementally updates weights to minimize computation, enhancing performance for larger networks.

## Installation

Before running the simulation, you need to install the required Python libraries. You can install them using `pip`:

```bash
pip install networkx numpy
