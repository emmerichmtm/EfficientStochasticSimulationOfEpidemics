from datetime import datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from itertools import combinations

# ============================================================
# Shared helpers
# ============================================================

def build_chain_of_cliques(num_cliques=10, clique_size=100):
    G = nx.Graph()
    for k in range(num_cliques):
        start = k * clique_size
        nodes = list(range(start, start + clique_size))
        G.add_nodes_from(nodes)
        G.add_edges_from(combinations(nodes, 2))  # clique
    for k in range(num_cliques - 1):
        left_node  = (k + 1) * clique_size - 1
        right_node = (k + 1) * clique_size
        G.add_edge(left_node, right_node)  # single bridge
    return G

def step_sample(times, values, grid):
    """Sample a right-continuous step function defined by (times, values) onto grid."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    idx = np.searchsorted(times, grid, side="right") - 1
    idx = np.clip(idx, 0, len(values)-1)
    return values[idx]

# ============================================================
# Micro model (node-level Gillespie) - fixed beta scaling
# ============================================================

class EpidemicGraphFixed:
    def __init__(self, infection_rate=0.1, recovery_rate=0.0, model=0, rng=None):
        self.G = nx.Graph()
        self.model = model
        self.infection_rate = float(infection_rate)   # beta
        self.recovery_rate = float(recovery_rate)     # gamma
        self.infected_nodes = []
        self.total_infection_rate = 0.0  # unscaled sum of SI edge-weights
        self.total_recovery_rate = 0.0   # gamma * I
        self.on_infect = None
        self.on_recover = None
        self.rng = rng or random.Random()

    def add_node(self, node_id):
        self.G.add_node(node_id, infected=False, recovered=False, sum_of_weights_i=0.0)

    def add_edge(self, node1, node2, weight):
        self.G.add_edge(node1, node2, weight=float(weight))

    def simulate_step(self, current_time: float, event_iter: int):
        if not self.infected_nodes:
            return float("inf")

        scaled_infection = self.infection_rate * self.total_infection_rate
        scaled_recovery  = 0.0 if self.model == 0 else self.total_recovery_rate
        total_rate = scaled_infection + scaled_recovery

        if total_rate <= 1e-12:
            return float("inf")

        wait_time = self.rng.expovariate(total_rate)
        event_time = current_time + wait_time

        r = self.rng.uniform(0.0, total_rate)
        if r < scaled_infection:
            target = self.rng.uniform(0.0, self.total_infection_rate)
            cumulative = 0.0
            for node in self.infected_nodes:
                cumulative += self.G.nodes[node]["sum_of_weights_i"]
                if cumulative > target:
                    self.infect_neighbor(node, event_time, event_iter)
                    break
        else:
            # recovery: uniform over infected nodes when all have same gamma
            if self.total_recovery_rate <= 1e-12:
                return wait_time
            target = self.rng.uniform(0.0, self.total_recovery_rate)
            cumulative = 0.0
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
            self.G.nodes[node]["sum_of_weights_i"] = 0.0
            if self.on_recover:
                self.on_recover(node, event_time, event_iter)

        elif self.model == 2:  # SIR
            self.G.nodes[node]["recovered"] = True
            self.G.nodes[node]["infected"] = False
            self.G.nodes[node]["sum_of_weights_i"] = 0.0
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

        target = self.rng.uniform(0.0, total_w)
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += float(w)
            if cumulative > target:
                self.infect_node(neighbors[i], event_time, event_iter)
                break

    def infect_node(self, node, event_time: float = 0.0, event_iter: int = 0):
        if self.G.nodes[node]["infected"] or self.G.nodes[node]["recovered"]:
            return

        self.G.nodes[node]["infected"] = True
        self.infected_nodes.append(node)
        self.total_recovery_rate += self.recovery_rate

        if self.on_infect:
            self.on_infect(node, event_time, event_iter)

        for neighbor in self.G.neighbors(node):
            w = self.G[node][neighbor]["weight"]
            if not self.G.nodes[neighbor]["infected"]:
                self.G.nodes[node]["sum_of_weights_i"] += w
                self.total_infection_rate += w
            elif neighbor != node:
                self.G.nodes[neighbor]["sum_of_weights_i"] -= w
                self.total_infection_rate -= w


def simulate_micro_chain_single(
    baseG, num_cliques=10, clique_size=100, edge_weight=1.0,
    initial_infected=0, infection_rate=0.2, recovery_rate=0.0, model=0,
    max_events=200000, seed=1
):
    rng = random.Random(seed)

    epi = EpidemicGraphFixed(infection_rate=infection_rate, recovery_rate=recovery_rate, model=model, rng=rng)
    for node in baseG.nodes:
        epi.add_node(node)
    for u, v in baseG.edges:
        epi.add_edge(u, v, edge_weight)

    def clique_of(node_id: int) -> int:
        return node_id // clique_size

    bridge_node = [(k + 1) * clique_size - 1 for k in range(num_cliques)]
    current_infected = [0] * num_cliques
    first_infection = [None] * num_cliques
    bridge_infection = [None] * num_cliques

    def on_infect(node_id: int, event_time: float, event_iter: int):
        c = clique_of(node_id)
        current_infected[c] += 1
        if first_infection[c] is None:
            first_infection[c] = (event_time, current_infected[c], event_iter)
        if c < num_cliques - 1 and node_id == bridge_node[c] and bridge_infection[c] is None:
            bridge_infection[c] = (event_time, current_infected[c], event_iter)

    def on_recover(node_id: int, event_time: float, event_iter: int):
        c = clique_of(node_id)
        current_infected[c] -= 1

    epi.on_infect = on_infect
    epi.on_recover = on_recover

    epi.infect_node(initial_infected, event_time=0.0, event_iter=0)

    times = [0.0]
    series = [[ci] for ci in current_infected]
    t = 0.0
    for it in range(1, max_events + 1):
        dt = epi.simulate_step(t, event_iter=it)
        if not np.isfinite(dt):
            break
        t += dt
        times.append(t)
        for k in range(num_cliques):
            series[k].append(current_infected[k])

    return np.array(times), [np.array(s) for s in series], first_infection, bridge_infection


def simulate_micro_total(
    baseG, clique_size=100,
    edge_weight=1.0, initial_infected=0,
    infection_rate=0.2, recovery_rate=0.0, model=0,
    max_events=200000, seed=1
):
    rng = random.Random(seed)
    epi = EpidemicGraphFixed(infection_rate=infection_rate, recovery_rate=recovery_rate, model=model, rng=rng)
    for node in baseG.nodes:
        epi.add_node(node)
    for u, v in baseG.edges:
        epi.add_edge(u, v, edge_weight)

    total_I = 0
    def on_infect(node_id, event_time, event_iter):
        nonlocal total_I
        total_I += 1
    def on_recover(node_id, event_time, event_iter):
        nonlocal total_I
        total_I -= 1

    epi.on_infect = on_infect
    epi.on_recover = on_recover

    epi.infect_node(initial_infected, event_time=0.0, event_iter=0)
    total_I = 1

    times = [0.0]
    totals = [total_I]
    t = 0.0
    for it in range(1, max_events + 1):
        dt = epi.simulate_step(t, event_iter=it)
        if not np.isfinite(dt):
            break
        t += dt
        times.append(t)
        totals.append(total_I)

    return np.array(times), np.array(totals)


# ============================================================
# Macro/metapopulation model (well-mixed cliques + explicit bridge nodes)
# ============================================================

def simulate_meta_chain_single(
    num_cliques=10, clique_size=100, edge_weight=1.0,
    initial_infected=0, infection_rate=0.2, recovery_rate=0.0, model=0,
    max_events=200000, seed=2
):
    rng = random.Random(seed)
    K, N = num_cliques, clique_size
    beta, gamma = float(infection_rate), float(recovery_rate)
    w_bridge = float(edge_weight)

    has_L = [k > 0 for k in range(K)]
    has_R = [k < K - 1 for k in range(K)]
    bulk_size = [N - int(has_L[k]) - int(has_R[k]) for k in range(K)]

    S_bulk = np.array(bulk_size, dtype=int)
    I_bulk = np.zeros(K, dtype=int)
    R_bulk = np.zeros(K, dtype=int)

    L_state = np.array([-1 if not has_L[k] else 0 for k in range(K)], dtype=int)  # -1 absent, 0 S, 1 I, 2 R
    R_state = np.array
