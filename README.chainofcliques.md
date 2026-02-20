# Efficient Stochastic Simulation of Epidemics

This repository contains a lean implementation of Gillespie's algorithm for simulating epidemic spread through networks using the **SIS**, **SIR**, and **SI** (if recovery rate = 0) models. The main file can be found in the main branch, and the code is optimized for performance by exploiting the sparsity of graphs and using incremental updates for infection and recovery rates.

# License

Copyright (c) 2025 Michael T.M. Emmerich  
Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)  
You are free to share and adapt this work with attribution.  
Provided "as is", without warranty of any kind.

## New in this Version

- **Gillespie’s Algorithm**: Implemented for SIS, SIR, and SI models, using specifics of epidemic processes (linear number of state transitions) to avoid state space explosion.
- **Optimized for Sparse Networks**: The implementation takes advantage of network sparsity, making it much faster when the number of neighbors per node is small.
- **Optimized for Sparse Networks**: The implementation maintains a list of infected nodes, making it faster when the number of infected nodes is small.
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
pip install networkx numpy matplotlib
```

---

# Demo: Chain of Cliques (micro vs macro + statistics)

This repository also contains a demonstration script:

- **`main_chain_of_cliques.py`**

It compares two stochastic simulation strategies **on the same chain-of-cliques network** and produces plots saved as timestamped PNG files.

## Network used in the demo

- **Chain of 10 cliques**, each clique has **100 nodes**
- Neighboring cliques are connected by **one bridge edge**
- “Bridge-to-next” node of clique *k* is the **last node** in that clique (except the last clique)

## How to run

From the repository root:

```bash
python main_chain_of_cliques.py
```

### Output files

Depending on your environment (terminal, remote, CI, etc.), plots may not pop up interactively.  
`main_chain_of_cliques.py` therefore **saves plots as PNG files** using a timestamp in the filename.

Typical output filenames look like:

- `chain_cliques_micro_vs_meta_YYYYMMDD-HHMMSS.png`
- `chain_cliques_micro_macro_with_stats_YYYYMMDD-HHMMSS.png`

(They are saved into the current working directory unless you change the output path in the script.)

## What the plots show

The script generates a combined figure with:

1. **Micro simulation panel**: per-clique infected counts vs simulated time (node-level Gillespie on the full network)
2. **Macro/metapopulation panel**: per-clique infected counts vs simulated time (well-mixed cliques + explicit bridge nodes)
3. **Statistical panel**: total infected across **all nodes** over many runs (median + quantile band)

Per-clique panels annotate two event times per clique:

- **X**: time when the **first node** in that clique becomes infected  
- **^**: time when the **bridge-to-next-clique node** becomes infected  
  (only meaningful for cliques 1..9; clique 10 has no “next”)

The statistical panel shows:

- solid line = **median** total infected over many runs  
- shaded area = **quantile band** (default: **25%–75%**)

---

# Conceptual overview

## Micro simulation (node-level Gillespie)

This is the classic **event-based** Gillespie algorithm for epidemics on a graph:

- The system is in a discrete state (infected / susceptible / recovered per node).
- Next event time is sampled from an exponential distribution with rate equal to the **total hazard**.
- Events are:
  - **infection** along an edge from an infected node to a susceptible neighbor
  - **recovery** of an infected node (SIS/SIR only)

To be efficient, the micro implementation maintains and updates:

- the current list of infected nodes,
- for each infected node, the **sum of weights of SI edges** leaving that node,
- and a global sum of these SI weights (plus a global recovery hazard in SIS/SIR).

This allows fast incremental updates rather than recomputing hazards from scratch after each event.

## Macro / metapopulation simulation (clique-level approximation)

The chain-of-cliques graph has strong time-scale separation:

- within a clique (complete graph), infection spreads very quickly once it arrives,
- between cliques, spread is bottlenecked by a **single bridge edge**.

The macro model exploits this by:

- treating each clique as **well-mixed** internally (counts of S/I/R in the clique),
- while tracking the **bridge nodes explicitly** so cross-clique transmission still happens only across the single bridge edge between neighboring cliques.

Internal infection in clique *k* uses the complete-graph SI pair count:

- number of SI pairs = \( I_k \cdot S_k \)
- internal infection rate ≈ \( \beta\, I_k S_k \)

Cross-clique infection uses the single bridge edge:

- rate ≈ \( \beta\, w_{\text{bridge}} \) when an infected endpoint is connected to a susceptible endpoint.

This macro model is **faster** and often captures the correct qualitative “wave” of infection along the chain, but it is still an approximation: it replaces within-clique network randomness by the well-mixed assumption.

---

# Bugfixes in `main_chain_of_cliques.py` (relative to the original demo version)

The micro simulation in `main_chain_of_cliques.py` includes two key fixes compared to the earlier chain-of-cliques demo code:

1. **Infection rate β is now applied to the infection hazard**  
   In the original demo code, the waiting time and the infection-vs-recovery choice used the sum of SI-edge weights, but the parameter `infection_rate` (β) was not used to scale the infection hazard.  
   **Fix:** total infection hazard is computed as  
   \[
   \lambda_{\text{inf}} = \beta \sum_{(i,j)\in SI} w_{ij}
   \]
   so the exponential waiting time and event selection use the correct scaled hazard.

2. **Recovery event selection was corrected (SIS/SIR)**  
   In the original demo, the recovery selection logic used an incorrect cumulative initialization, which can bias which infected node is selected for recovery.  
   **Fix:** recovery selection is consistent with a per-node recovery hazard γ (uniform over infected nodes if all share the same γ).

Additionally, the demo terminates when **no events are possible** (e.g., SI once all nodes are infected), rather than continuing to append repeated time points.

---

# Tuning the statistical experiment

Inside `main_chain_of_cliques.py` you can adjust:

- `n_runs` (number of independent runs)
- the quantiles (default uses 25%, 50%, 75%)
- the resolution of the time grid used for computing quantiles
- model parameters: `infection_rate`, `recovery_rate`, `model` (SI/SIS/SIR), and clique count/size

If you want a “central 75% interval”, use quantiles 12.5% and 87.5% instead of 25% and 75%.
