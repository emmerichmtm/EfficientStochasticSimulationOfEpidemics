# Copyright (c) 2025 Michael T.M. Emmerich
# Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
# You are free to share and adapt this work with attribution.
# Provided "as is", without warranty of any kind.

# Chain of 10 cliques (100 nodes each), single bridge edge between neighbors.
# Plot per-clique infected counts over *SIMULATED TIME*,
# and mark:
#   (X)  time when the FIRST node in that clique becomes infected
#   (^)  time when the BRIDGE-TO-NEXT-CLIQUE node in that clique becomes infected
#
# (For clique 10, there is no "bridge to next", so only the first-infection marker applies.)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from itertools import combinations


class EpidemicGraph:
    def __init__(self, infection_rate=0.1, recovery_rate=0.0, model=0):
        # model: 0: SI, 1: SIS, 2: SIR
        self.G = nx.Graph()
        self.model = model
        self.infection_rate, self.recovery_rate = infection_rate, recovery_rate
        self.infected_nodes = []
        self.total_infection_rate, self.total_recovery_rate = 0.0, 0.0

        # hooks: callable(node_id, event_time, event_iter)
        self.on_infect = None
        self.on_recover = None

    def add_node(self, node_id):
        self.G.add_node(node_id, infected=False, recovered=False, sum_of_weights_i=0.0)

    def add_edge(self, node1, node2, weight):
        self.G.add_edge(node1, node2, weight=weight)

    def simulate_step(self, current_time: float, event_iter: int):
        if not self.infected_nodes:
            return float("inf")

        total_rate = self.total_infection_rate if self.model == 0 else (self.total_infection_rate + self.total_recovery_rate)
        if total_rate < 1e-12:
            return 0.0

        wait_time = random.expovariate(total_rate)
        event_time = current_time + wait_time

        r = random.uniform(0, total_rate)
        if r < self.total_infection_rate:  # infection event
            target = random.uniform(0, self.total_infection_rate)
            cumulative = 0.0
            for node in self.infected_nodes:
                cumulative += self.G.nodes[node]["sum_of_weights_i"]
                if cumulative > target:
                    self.infect_neighbor(node, event_time, event_iter)
                    break
        else:  # recovery event
            target = random.uniform(0, self.total_recovery_rate)
            cumulative = self.total_infection_rate
            for node in self.infected_nodes:
                cumulative += self.recovery_rate
                if cumulative > target:
                    self.recover_node(node, event_time, event_iter)
                    break

        return wait_time

    def recover_node(self, node, event_time: float, event_iter: int):
        self.infected_nodes.remove(node)
        self.total_recovery_rate -= self.recovery_rate

        if self.model == 1:  # SIS
            for neighbor in self.G.neighbors(node):
                self.G.nodes[neighbor]["sum_of_weights_i"] += self.G[node][neighbor]["weight"]
            self.total_infection_rate += self.G.nodes[node]["sum_of_weights_i"]
            self.G.nodes[node]["infected"] = False
            if self.on_recover:
                self.on_recover(node, event_time, event_iter)

        elif self.model == 2:  # SIR
            self.G.nodes[node]["recovered"] = True
            self.G.nodes[node]["infected"] = False
            if self.on_recover:
                self.on_recover(node, event_time, event_iter)

    def infect_neighbor(self, node, event_time: float, event_iter: int):
        neighbors = [
            n for n in self.G.neighbors(node)
            if (not self.G.nodes[n]["infected"]) and (not self.G.nodes[n]["recovered"])
        ]
        if not neighbors:
            return

        weights = np.array([self.G[node][n]["weight"] for n in neighbors], dtype=float)
        total_w = float(np.sum(weights))
        if total_w <= 0.0:
            return

        target = random.uniform(0, total_w)
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += float(w)
            if cumulative > target:
                self.infect_node(neighbors[i], event_time, event_iter)
                break

    def infect_node(self, node, event_time: float = 0.0, event_iter: int = 0):
        self.G.nodes[node]["infected"] = True
        self.infected_nodes.append(node)
        self.total_recovery_rate += self.recovery_rate

        if self.on_infect:
            self.on_infect(node, event_time, event_iter)

        # bookkeeping for infection rates
        for neighbor in self.G.neighbors(node):
            w = self.G[node][neighbor]["weight"]
            if not self.G.nodes[neighbor]["infected"]:
                self.G.nodes[node]["sum_of_weights_i"] += w
                self.total_infection_rate += w
            elif neighbor != node:
                self.G.nodes[neighbor]["sum_of_weights_i"] -= w
                self.total_infection_rate -= w


def build_chain_of_cliques(num_cliques=10, clique_size=100):
    G = nx.Graph()

    for k in range(num_cliques):
        start = k * clique_size
        nodes = list(range(start, start + clique_size))
        G.add_nodes_from(nodes)
        G.add_edges_from(combinations(nodes, 2))  # clique

    for k in range(num_cliques - 1):
        left_node = (k + 1) * clique_size - 1   # last node of clique k
        right_node = (k + 1) * clique_size      # first node of clique k+1
        G.add_edge(left_node, right_node)       # single bridge

    return G


def run_simulation(
    num_cliques=10,
    clique_size=100,
    edge_weight=1.0,
    steps=300000,
    initial_infected=0,
    infection_rate=0.2,
    recovery_rate=0.0,
    model=0,  # 0: SI, 1: SIS, 2: SIR
):
    baseG = build_chain_of_cliques(num_cliques=num_cliques, clique_size=clique_size)

    epi = EpidemicGraph(infection_rate=infection_rate, recovery_rate=recovery_rate, model=model)
    for node in baseG.nodes:
        epi.add_node(node)
    for u, v in baseG.edges:
        epi.add_edge(u, v, edge_weight)

    def clique_of(node_id: int) -> int:
        return node_id // clique_size

    # bridge-to-next node in clique k is the last node of that clique (except last clique has no "next")
    bridge_node = [(k + 1) * clique_size - 1 for k in range(num_cliques)]

    current_infected = [0] * num_cliques

    # markers: store (time, y, iter)
    first_infection = [None] * num_cliques
    bridge_infection = [None] * num_cliques  # meaningful only for k=0..num_cliques-2

    def on_infect(node_id: int, event_time: float, event_iter: int):
        c = clique_of(node_id)
        current_infected[c] += 1

        # first infection in that clique
        if first_infection[c] is None:
            first_infection[c] = (event_time, current_infected[c], event_iter)

        # infection of the bridge-to-next node (for cliques that have a next clique)
        if c < num_cliques - 1 and node_id == bridge_node[c] and bridge_infection[c] is None:
            bridge_infection[c] = (event_time, current_infected[c], event_iter)

    def on_recover(node_id: int, event_time: float, event_iter: int):
        c = clique_of(node_id)
        current_infected[c] -= 1

    epi.on_infect = on_infect
    epi.on_recover = on_recover

    # initial infection at time 0, iteration 0
    epi.infect_node(initial_infected, event_time=0.0, event_iter=0)

    # time series: record state after each event (including t=0)
    times = [0.0]
    series_current = [[0] for _ in range(num_cliques)]
    for k in range(num_cliques):
        series_current[k][0] = current_infected[k]

    total_time = 0.0
    for it in range(1, steps + 1):
        dt = epi.simulate_step(total_time, event_iter=it)
        total_time += dt
        times.append(total_time)
        for k in range(num_cliques):
            series_current[k].append(current_infected[k])

    # plot lines + markers
    colors = plt.cm.tab10(np.linspace(0, 1, num_cliques))

    for k in range(num_cliques):
        plt.plot(times, series_current[k], color=colors[k], label=f"Clique {k+1}")

        if first_infection[k] is not None:
            t0, y0, _ = first_infection[k]
            plt.scatter([t0], [y0], color=colors[k], marker="X", s=90, zorder=6)

        if bridge_infection[k] is not None:
            tb, yb, _ = bridge_infection[k]
            plt.scatter([tb], [yb], color=colors[k], marker="^", s=90, zorder=6)

    # add a tiny legend for the marker meanings (dummy points)
    plt.scatter([], [], color="black", marker="X", s=90, label="First infection in clique")
    plt.scatter([], [], color="black", marker="^", s=90, label="Bridge-node infected (to next clique)")

    plt.xlabel("Simulated Time")
    plt.ylabel("Currently infected in clique")
    plt.title("Per-clique infected vs time (X=first infection, ^=bridge-node infected)")
    #plt.legend(ncol=2, fontsize=9)
    plt.show()


if __name__ == "__main__":
    run_simulation(
        num_cliques=10,
        clique_size=100,
        edge_weight=1.0,
        steps=300000,       # increase if you want (e.g. 1_000_000)
        initial_infected=0, # starts in clique 1
        infection_rate=0.2,
        recovery_rate=0.0,
        model=0,
    )
