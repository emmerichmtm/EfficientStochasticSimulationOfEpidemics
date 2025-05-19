# Two‑Strain SIR Epidemic Simulator

> **Author:** Michael T. M. Emmerich (2025)
> **License:** CC BY 4.0 – Commercial and non‑commercial use allowed with attribution.
> **File(s):** `two_strain_epidemic.py` (main code) + this **README**

---

## Overview

This project contains a stochastic, Gillespie‑exact implementation of a **two‑strain SIR epidemic model** on arbitrary networks.  Each mutant (A & B) has its own infection rate (β) while sharing full cross‑immunity: once a host recovers from *either* strain it is permanently immune to **both**.

Key features :

* **Separate transmission parameters** for mutants A & B (`βₐ`, `βᵦ`)
* **Single recovery parameter** `γ` (identical for both strains)
* **Gillespie algorithm** for exact event timing
* **Edge‑weighted networks** – contact strength modulates transmission rate
* **Flexible seeding strategies** (`high_degree` or `random`) to start the outbreak
* **ICU load estimator** with per‑strain severity factors

---

## Dependencies

| Package    | Tested Version |
| ---------- | -------------- |
| Python     | ≥ 3.9          |
| numpy      | 1.26+          |
| networkx   | 3.3+           |
| matplotlib | 3.9+           |

Install via **pip**:

```bash
pip install numpy networkx matplotlib
```

---

## Quick Start

```bash
# clone / download project
python two_strain_epidemic.py
```

Running the script without arguments executes:

1. A **5‑node toy demo** (prints counts to console).
2. A **5 000‑node Barabási–Albert ICU demo** with hub‑based seeding and default parameters.  A step‑plot shows infections for each strain and estimated ICU admissions.

> **Tip :** Want faster runs? Lower `n_nodes` or `max_events` inside `barabasi_albert_icu_demo()`.

---

## Using the Library

```python
from two_strain_epidemic import TwoStrainEpidemicGraph

G = TwoStrainEpidemicGraph((0.15, 0.14), recovery_rate=0.1)
# build graph, add nodes & edges …
G.infect_node(0, strain='A')
while G.simulate_step() != float('inf'):
    pass
```

All public methods are documented in‑line.

### High‑level ICU demo

```python
from two_strain_epidemic import barabasi_albert_icu_demo
barabasi_albert_icu_demo(
    n_nodes=10000,
    seed_strategy="random",
    icu_rates=(0.02, 0.03))
```

This builds the network, seeds two mutants, runs a simulation, and pops up a plot.

---

## Parameters

| Argument          | Description                                              | Default         |
| ----------------- | -------------------------------------------------------- | --------------- |
| `infection_rates` | `(βₐ, βᵦ)` per‑strain transmission coefficients          | `(0.12, 0.10)`  |
| `recovery_rate`   | γ — reciprocal of infectious period                      | `0.11`          |
| `icu_rates`       | `(p_A, p_B)` share of current infections admitted to ICU | `(0.05, 0.10)`  |
| `seed_strategy`   | `"high_degree"` (two largest hubs) or `"random"`         | `"high_degree"` |

---

## File Layout

```
├── two_strain_epidemic.py   # core simulator + demos
└── README.md                # project documentation (this file)
```

---

## Citation

Please cite the author and link back to this repository if you use or adapt the code :

```
Emmerich, M.T.M. (2025). Two‑Strain SIR Epidemic Simulator (v1.0). CC BY 4.0.
```

---

## License

Creative Commons **Attribution 4.0 International** (CC BY 4.0). See [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).
