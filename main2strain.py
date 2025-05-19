# Copyright (c) 2025 Michael T.M. Emmerich
# Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
# You are free to share and adapt this work with attribution.
# Provided "as is", without warranty of any kind.

"""
Two‑Strain SIR Epidemic Model
=============================
This simulator tracks two mutants (A & B) that share full cross‑immunity but
have **slightly different contagiousness (β)**.  Defaults are now:

* βₐ = 0.12
* βᵦ = 0.10
* γ   = 0.11

The large‑scale demo lets you pick a **seeding strategy** so that strain B gets
an equally good start:

* `seed_strategy="high_degree"` (default) – put strains on the two highest‑degree hubs.
* `seed_strategy="random"` – pick two distinct random nodes.

Run the module to execute both the toy demo and the ICU demo, or call
`barabasi_albert_icu_demo()` directly with custom parameters.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from typing import Tuple, List


# ----------------------------------------------------------------------
# Core epidemic class
# ----------------------------------------------------------------------
class TwoStrainEpidemicGraph:
    """Stochastic SIR epidemic with two fully cross‑immunising mutants."""

    def __init__(self,
                 infection_rates: Tuple[float, float] = (0.12, 0.10),
                 recovery_rate: float = 0.11):
        self.beta = {'A': infection_rates[0], 'B': infection_rates[1]}
        self.gamma = recovery_rate

        self.G = nx.Graph()
        self.infected_nodes: List[int] = []
        self.total_infection_rate: float = 0.0
        self.total_recovery_rate: float = 0.0

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------
    def add_node(self, node_id: int):
        self.G.add_node(node_id,
                        infected=False,
                        strain=None,
                        recovered=False,
                        sum_of_weights_i=0.0)

    def add_edge(self, u: int, v: int, weight: float = 1.0):
        self.G.add_edge(u, v, weight=weight)

    # ------------------------------------------------------------------
    # Gillespie scheduler
    # ------------------------------------------------------------------
    def simulate_step(self) -> float:
        if not self.infected_nodes:
            return float('inf')

        total_rate = self.total_infection_rate + self.total_recovery_rate
        if total_rate < 1e-9:
            return 0.0

        wait_time = random.expovariate(total_rate)
        r = random.uniform(0, total_rate)

        if r < self.total_infection_rate:
            self._infection_event()
        else:
            self._recovery_event()
        return wait_time

    def _infection_event(self):
        target = random.uniform(0, self.total_infection_rate)
        cumulative = 0.0
        for node in self.infected_nodes:
            cumulative += self.G.nodes[node]['sum_of_weights_i']
            if cumulative > target:
                self._infect_neighbour(node)
                break

    def _recovery_event(self):
        target = random.uniform(0, self.total_recovery_rate)
        cumulative = 0.0
        for node in self.infected_nodes:
            cumulative += self.gamma
            if cumulative > target:
                self._recover_node(node)
                break

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    def infect_node(self, node: int, strain: str = 'A'):
        info = self.G.nodes[node]
        if info['infected'] or info['recovered']:
            return

        info.update(infected=True, strain=strain)
        self.infected_nodes.append(node)
        self.total_recovery_rate += self.gamma

        beta_s = self.beta[strain]
        s_w = 0.0
        for nbr in self.G.neighbors(node):
            w = self.G[node][nbr]['weight']
            nbr_info = self.G.nodes[nbr]
            if not nbr_info['infected'] and not nbr_info['recovered']:
                inc = beta_s * w
                s_w += inc
                self.total_infection_rate += inc
            elif nbr_info['infected']:
                beta_n = self.beta[nbr_info['strain']]
                dec = beta_n * w
                nbr_info['sum_of_weights_i'] -= dec
                self.total_infection_rate -= dec

        info['sum_of_weights_i'] = s_w

    def _infect_neighbour(self, node: int):
        strain = self.G.nodes[node]['strain']
        beta_s = self.beta[strain]
        neighbours = [n for n in self.G.neighbors(node)
                      if not self.G.nodes[n]['infected'] and not self.G.nodes[n]['recovered']]
        if not neighbours:
            return

        weights = np.array([self.G[node][n]['weight'] * beta_s for n in neighbours])
        target = random.uniform(0, weights.sum())
        cumulative = 0.0
        for n, w in zip(neighbours, weights):
            cumulative += w
            if cumulative > target:
                self.infect_node(n, strain=strain)
                break

    def _recover_node(self, node: int):
        info = self.G.nodes[node]
        self.infected_nodes.remove(node)
        self.total_recovery_rate -= self.gamma
        self.total_infection_rate -= info['sum_of_weights_i']
        info.update(infected=False, strain=None, recovered=True, sum_of_weights_i=0.0)

    # ------------------------------------------------------------------
    # Mini visual demo
    # ------------------------------------------------------------------
    def demo_small(self):
        for i in range(5):
            self.add_node(i)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 4)]
        for u, v in edges:
            self.add_edge(u, v, 1.0)
        self.infect_node(0, 'A')
        self.infect_node(3, 'B')
        for _ in range(6):
            print(f"I_A={sum(1 for n in self.infected_nodes if self.G.nodes[n]['strain']=='A')}, I_B={sum(1 for n in self.infected_nodes if self.G.nodes[n]['strain']=='B')}")
            self.simulate_step()


# ----------------------------------------------------------------------
# Large‑network ICU demo
# ----------------------------------------------------------------------

def barabasi_albert_icu_demo(n_nodes: int = 5000,
                             m: int = 2,
                             infection_rates: Tuple[float, float] = (0.12, 0.10),
                             recovery_rate: float = 0.11,
                             icu_rates: Tuple[float, float] = (0.05, 0.1),
                             max_events: int = 30000,
                             random_seed: int | None = 1,
                             seed_strategy: str = "high_degree"):
    """Simulate on a BA graph and plot infections & ICU load over time.

    Parameters
    ----------
    seed_strategy : "high_degree" | "random"
        * `high_degree` – seed A & B on the two highest‑degree nodes.
        * `random` – seed on two distinct random nodes.
    """

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    base_graph = nx.barabasi_albert_graph(n_nodes, m)
    epi = TwoStrainEpidemicGraph(infection_rates, recovery_rate)

    for node in base_graph.nodes:
        epi.add_node(node)
    for u, v in base_graph.edges:
        epi.add_edge(u, v, 1.0)

    # -------------------- choose seed nodes -------------------------------
    if seed_strategy == "high_degree":
        degree_sorted = sorted(base_graph.degree, key=lambda x: x[1], reverse=True)
        seed_A, seed_B = degree_sorted[0][0], degree_sorted[1][0]
    elif seed_strategy == "random":
        seed_A = random.randrange(n_nodes)
        nodes_remaining = list(base_graph.nodes)
        nodes_remaining.remove(seed_A)
        seed_B = random.choice(nodes_remaining)
    else:
        raise ValueError("seed_strategy must be 'high_degree' or 'random'")

    print(f"Seeding strain A at node {seed_A}; strain B at node {seed_B}")

    epi.infect_node(seed_A, 'A')
    epi.infect_node(seed_B, 'B')

    # -------------------- time‑series storage -----------------------------
    t_series: List[float] = []
    inf_A: List[int] = []
    inf_B: List[int] = []
    icu_load: List[float] = []

    current_time = 0.0
    for _ in range(max_events):
        n_A = sum(1 for n in epi.infected_nodes if epi.G.nodes[n]['strain'] == 'A')
        n_B = len(epi.infected_nodes) - n_A

        t_series.append(current_time)
        inf_A.append(n_A)
        inf_B.append(n_B)
        icu_load.append(icu_rates[0] * n_A + icu_rates[1] * n_B)

        dt = epi.simulate_step()
        if dt == float('inf'):
            break
        current_time += dt

    # -------------------- plot -------------------------------------------
    plt.figure()
    plt.step(t_series, inf_A, where='post', label='Infections – strain A', linewidth=1.2, color='red')
    plt.step(t_series, inf_B, where='post', label='Infections – strain B', linewidth=1.2, color='orange')
    plt.step(t_series, icu_load, where='post', label='ICU admissions', linewidth=1.2, color='purple')
    plt.xlabel('Simulated time')
    plt.ylabel('Number of individuals')
    plt.title(f'Epidemic on {n_nodes}-node Barabási–Albert network')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Example execution helper
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Toy sanity check
    demo = TwoStrainEpidemicGraph()
    demo.demo_small()

    # Large‑scale ICU demo with default (high‑degree) seeding
    barabasi_albert_icu_demo()
