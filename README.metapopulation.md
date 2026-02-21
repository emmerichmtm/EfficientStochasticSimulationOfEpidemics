# README.metapopulation.md

This note documents the **generalized metapopulation-of-cliques** experiment and the reference script:

- `main_chain_of_cliques_generalized.py`

The key feature is that **metapopulations (cliques) can be connected in arbitrary ways** (chain, ER, BA, Watts–Strogatz, custom graphs).

---

## Concept: “graph of cliques”

We start from a **metapopulation graph** `metaG = (V_meta, E_meta)`:

- each node `u ∈ V_meta` represents a *metapopulation* (a clique)
- each edge `(u,v) ∈ E_meta` represents a *connection* between metapopulations

### Expansion to a node-level graph (micro model)

For each metanode `u`, we create a clique of size `clique_size`.

For each meta-edge `(u,v)`, we create **one bridge edge** between:

- a unique bridge node inside clique(u), and
- a unique bridge node inside clique(v)

This preserves “sparse coupling” between dense internal subgraphs.

**Constraint:** to give each incident meta-edge its own bridge endpoint in a clique, you need:

- `clique_size > max_degree(metaG)`

The script checks this and raises an error if violated.

---

## Two simulation strategies

### 1) Micro simulation (node-level Gillespie)

- Runs on the full expanded graph (cliques + bridge edges)
- Exact continuous-time Markov process for SI/SIS/SIR
- Uses incremental bookkeeping of SI edge mass
- Infection hazard uses **β · (sum of SI weights)**

Outputs:
- per-clique infected counts vs simulated time
- markers for (X) first infection in a clique and (^) first infected bridge node in that clique

### 2) Macro simulation (metapopulation / clique-level Gillespie)

- Runs directly on `metaG`
- Each clique is assumed **well-mixed internally**
- Bridge nodes are **explicit** (one per incident meta-edge)
- Internal infection rate in clique u: `β · I_u · S_u`
- Bridge infection rate over meta-edge: `β · bridge_weight` when infected→susceptible across that bridge

Outputs:
- per-clique infected counts vs simulated time
- the same marker ideas (first infection / first infected bridge node)

---

## Example included: BA network of 10 cliques

The default main block runs:

- `metaG = nx.barabasi_albert_graph(meta_n=10, meta_m=2)`
- `clique_size = 50` (must exceed max degree)
- SI model by default (`recovery_rate=0.0`)

It produces a timestamped PNG file:

- `ba_metapop_cliques_micro_macro_YYYYMMDD-HHMMSS.png`

The plot contains 3 panels:
1. Micro per-clique trajectories
2. Macro per-clique trajectories
3. Total infected (median + 25–75% quantile band across many runs)

---

## Running

From your repo root:

```bash
python main_chain_of_cliques_generalized.py
```

To try a different metapopulation topology, edit the line in `run_ba_demo()`:

```python
metaG = nx.barabasi_albert_graph(meta_n, meta_m, seed=seed)
```

Examples:

```python
metaG = nx.path_graph(meta_n)                 # chain
metaG = nx.erdos_renyi_graph(meta_n, 0.2)     # ER
metaG = nx.watts_strogatz_graph(meta_n, 4, 0.2, seed=seed)
```

---

## Parameters to tune

- `meta_n`, `meta_m`: metapopulation graph size (BA parameters)
- `clique_size`: size of each clique (must exceed max meta-degree)
- `infection_rate`, `recovery_rate`, `model`: SI / SIS / SIR behavior
- `internal_weight`, `bridge_weight`: weights for internal edges and bridges
- `n_runs`: number of runs for quantile statistics
- `max_events`: event cap per run
