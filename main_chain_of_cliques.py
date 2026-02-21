from datetime import datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from itertools import combinations

# ------------------ Helpers ------------------

def build_chain_of_cliques(num_cliques=10, clique_size=100):
    G = nx.Graph()
    for k in range(num_cliques):
        start = k * clique_size
        nodes = list(range(start, start + clique_size))
        G.add_nodes_from(nodes)
        G.add_edges_from(combinations(nodes, 2))
    for k in range(num_cliques - 1):
        left_node  = (k + 1) * clique_size - 1
        right_node = (k + 1) * clique_size
        G.add_edge(left_node, right_node)
    return G

def step_sample(times, values, grid):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    idx = np.searchsorted(times, grid, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]

def quantile_bands(runs_times, runs_vals, grid, qs=(0.25, 0.5, 0.75)):
    M = np.vstack([step_sample(t, v, grid) for t, v in zip(runs_times, runs_vals)])
    return np.quantile(M, qs, axis=0)

# ------------------ Micro model (fixed beta scaling) ------------------

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
            cum = 0.0
            for node in self.infected_nodes:
                cum += self.G.nodes[node]["sum_of_weights_i"]
                if cum > target:
                    self.infect_neighbor(node, event_time, event_iter)
                    break
        else:
            # recovery selection (uniform over infected when all have same gamma)
            if self.total_recovery_rate <= 1e-12:
                return wait_time
            target = self.rng.uniform(0.0, self.total_recovery_rate)
            cum = 0.0
            for node in self.infected_nodes:
                cum += self.recovery_rate
                if cum > target:
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
        nbrs = [n for n in self.G.neighbors(node)
                if (not self.G.nodes[n]["infected"]) and (not self.G.nodes[n]["recovered"])]
        if not nbrs:
            return
        weights = np.array([self.G[node][n]["weight"] for n in nbrs], dtype=float)
        tot = float(np.sum(weights))
        if tot <= 0.0:
            return
        target = self.rng.uniform(0.0, tot)
        cum = 0.0
        for i, w in enumerate(weights):
            cum += float(w)
            if cum > target:
                self.infect_node(nbrs[i], event_time, event_iter)
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

def simulate_micro_chain_single(baseG, num_cliques, clique_size, edge_weight,
                               initial_infected, infection_rate, recovery_rate, model,
                               max_events, seed):
    rng = random.Random(seed)
    epi = EpidemicGraphFixed(infection_rate=infection_rate, recovery_rate=recovery_rate, model=model, rng=rng)
    for node in baseG.nodes:
        epi.add_node(node)
    for u, v in baseG.edges:
        epi.add_edge(u, v, edge_weight)

    def clique_of(node_id): return node_id // clique_size
    bridge_node = [(k + 1) * clique_size - 1 for k in range(num_cliques)]

    current = [0] * num_cliques
    first = [None] * num_cliques
    bridge = [None] * num_cliques

    def on_inf(node_id, t, it):
        c = clique_of(node_id)
        current[c] += 1
        if first[c] is None:
            first[c] = (t, current[c], it)
        if c < num_cliques - 1 and node_id == bridge_node[c] and bridge[c] is None:
            bridge[c] = (t, current[c], it)

    def on_rec(node_id, t, it):
        c = clique_of(node_id)
        current[c] -= 1

    epi.on_infect = on_inf
    epi.on_recover = on_rec

    epi.infect_node(initial_infected, event_time=0.0, event_iter=0)

    times = [0.0]
    series = [[ci] for ci in current]
    t = 0.0
    for it in range(1, max_events + 1):
        dt = epi.simulate_step(t, it)
        if not np.isfinite(dt):
            break
        t += dt
        times.append(t)
        for k in range(num_cliques):
            series[k].append(current[k])

    return np.array(times), [np.array(s) for s in series], first, bridge

def simulate_micro_total(baseG, edge_weight, initial_infected, infection_rate, recovery_rate, model, max_events, seed):
    rng = random.Random(seed)
    epi = EpidemicGraphFixed(infection_rate=infection_rate, recovery_rate=recovery_rate, model=model, rng=rng)
    for node in baseG.nodes:
        epi.add_node(node)
    for u, v in baseG.edges:
        epi.add_edge(u, v, edge_weight)

    total_I = 0
    def on_inf(node_id, t, it):
        nonlocal total_I
        total_I += 1
    def on_rec(node_id, t, it):
        nonlocal total_I
        total_I -= 1

    epi.on_infect = on_inf
    epi.on_recover = on_rec

    epi.infect_node(initial_infected, event_time=0.0, event_iter=0)
    total_I = 1

    times = [0.0]
    totals = [total_I]
    t = 0.0
    for it in range(1, max_events + 1):
        dt = epi.simulate_step(t, it)
        if not np.isfinite(dt):
            break
        t += dt
        times.append(t)
        totals.append(total_I)

    return np.array(times), np.array(totals)

# ------------------ Macro model ------------------

def simulate_meta_chain_single(num_cliques, clique_size, edge_weight,
                              initial_infected, infection_rate, recovery_rate, model,
                              max_events, seed):
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
    L_state = np.array([-1 if not has_L[k] else 0 for k in range(K)], dtype=int)
    R_state = np.array([-1 if not has_R[k] else 0 for k in range(K)], dtype=int)

    def I_k(k): return int(I_bulk[k] + (L_state[k] == 1) + (R_state[k] == 1))
    def S_k(k):
        s = int(S_bulk[k])
        if has_L[k] and L_state[k] == 0: s += 1
        if has_R[k] and R_state[k] == 0: s += 1
        return s

    first = [None] * K
    bridge = [None] * K

    k0 = initial_infected // N
    start, end = k0 * N, (k0 + 1) * N - 1
    if has_L[k0] and initial_infected == start:
        L_state[k0] = 1
    elif has_R[k0] and initial_infected == end:
        R_state[k0] = 1
        bridge[k0] = (0.0, I_k(k0), 0)
    else:
        S_bulk[k0] -= 1
        I_bulk[k0] += 1
    first[k0] = (0.0, I_k(k0), 0)

    times = [0.0]
    series = [[I_k(k)] for k in range(K)]
    t = 0.0

    for it in range(1, max_events + 1):
        rates, events = [], []

        for k in range(K):
            I, S = I_k(k), S_k(k)
            if I > 0 and S > 0:
                rates.append(beta * I * S)
                events.append(("inf_internal", k))
            if model != 0 and I > 0 and gamma > 0:
                rates.append(gamma * I)
                events.append(("recover", k))

        for k in range(K - 1):
            if R_state[k] == 1 and L_state[k + 1] == 0:
                rates.append(beta * w_bridge); events.append(("bridge_k_to_k1", k))
            if L_state[k + 1] == 1 and R_state[k] == 0:
                rates.append(beta * w_bridge); events.append(("bridge_k1_to_k", k))

        total_rate = float(np.sum(rates))
        if total_rate <= 1e-12:
            break

        dt = rng.expovariate(total_rate)
        t += dt

        r = rng.uniform(0.0, total_rate)
        cum = 0.0
        chosen = events[-1]
        for rate, ev in zip(rates, events):
            cum += rate
            if cum > r:
                chosen = ev
                break

        kind = chosen[0]
        if kind == "inf_internal":
            k = chosen[1]
            S = S_k(k)
            u = rng.randrange(S)
            if u < S_bulk[k]:
                S_bulk[k] -= 1; I_bulk[k] += 1
            else:
                u -= S_bulk[k]
                if has_L[k] and L_state[k] == 0:
                    if u == 0:
                        L_state[k] = 1
                    else:
                        u -= 1
                        if has_R[k] and R_state[k] == 0 and u == 0:
                            R_state[k] = 1
                else:
                    if has_R[k] and R_state[k] == 0 and u == 0:
                        R_state[k] = 1
            if first[k] is None:
                first[k] = (t, I_k(k), it)
            if has_R[k] and k < K - 1 and R_state[k] == 1 and bridge[k] is None:
                bridge[k] = (t, I_k(k), it)

        elif kind == "recover":
            k = chosen[1]
            I = I_k(k)
            u = rng.randrange(I)
            if u < I_bulk[k]:
                I_bulk[k] -= 1
                if model == 1: S_bulk[k] += 1
                else: R_bulk[k] += 1
            else:
                u -= I_bulk[k]
                if has_L[k] and L_state[k] == 1:
                    if u == 0: L_state[k] = 0 if model == 1 else 2
                    else:
                        u -= 1
                        if has_R[k] and R_state[k] == 1 and u == 0:
                            R_state[k] = 0 if model == 1 else 2
                else:
                    if has_R[k] and R_state[k] == 1 and u == 0:
                        R_state[k] = 0 if model == 1 else 2

        elif kind == "bridge_k_to_k1":
            k = chosen[1]
            L_state[k + 1] = 1
            if first[k + 1] is None:
                first[k + 1] = (t, I_k(k + 1), it)

        elif kind == "bridge_k1_to_k":
            k = chosen[1]
            R_state[k] = 1
            if first[k] is None:
                first[k] = (t, I_k(k), it)
            if bridge[k] is None:
                bridge[k] = (t, I_k(k), it)

        times.append(t)
        for k in range(K):
            series[k].append(I_k(k))

    return np.array(times), [np.array(s) for s in series], first, bridge

def simulate_meta_total(num_cliques, clique_size, edge_weight,
                        initial_infected, infection_rate, recovery_rate, model,
                        max_events, seed):
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
    L_state = np.array([-1 if not has_L[k] else 0 for k in range(K)], dtype=int)
    R_state = np.array([-1 if not has_R[k] else 0 for k in range(K)], dtype=int)

    def I_k(k): return int(I_bulk[k] + (L_state[k] == 1) + (R_state[k] == 1))
    def S_k(k):
        s = int(S_bulk[k])
        if has_L[k] and L_state[k] == 0: s += 1
        if has_R[k] and R_state[k] == 0: s += 1
        return s
    def I_total():
        return int(np.sum(I_bulk) + np.sum(L_state == 1) + np.sum(R_state == 1))

    k0 = initial_infected // N
    start, end = k0 * N, (k0 + 1) * N - 1
    if has_L[k0] and initial_infected == start:
        L_state[k0] = 1
    elif has_R[k0] and initial_infected == end:
        R_state[k0] = 1
    else:
        S_bulk[k0] -= 1
        I_bulk[k0] += 1

    times = [0.0]
    totals = [I_total()]
    t = 0.0

    for it in range(1, max_events + 1):
        rates, events = [], []

        for k in range(K):
            I, S = I_k(k), S_k(k)
            if I > 0 and S > 0:
                rates.append(beta * I * S); events.append(("inf_internal", k))
            if model != 0 and I > 0 and gamma > 0:
                rates.append(gamma * I); events.append(("recover", k))

        for k in range(K - 1):
            if R_state[k] == 1 and L_state[k + 1] == 0:
                rates.append(beta * w_bridge); events.append(("bridge_k_to_k1", k))
            if L_state[k + 1] == 1 and R_state[k] == 0:
                rates.append(beta * w_bridge); events.append(("bridge_k1_to_k", k))

        total_rate = float(np.sum(rates))
        if total_rate <= 1e-12:
            break

        dt = rng.expovariate(total_rate)
        t += dt

        r = rng.uniform(0.0, total_rate)
        cum = 0.0
        chosen = events[-1]
        for rate, ev in zip(rates, events):
            cum += rate
            if cum > r:
                chosen = ev
                break

        kind = chosen[0]
        if kind == "inf_internal":
            k = chosen[1]
            S = S_k(k)
            u = rng.randrange(S)
            if u < S_bulk[k]:
                S_bulk[k] -= 1; I_bulk[k] += 1
            else:
                u -= S_bulk[k]
                if has_L[k] and L_state[k] == 0:
                    if u == 0: L_state[k] = 1
                    else:
                        u -= 1
                        if has_R[k] and R_state[k] == 0 and u == 0:
                            R_state[k] = 1
                else:
                    if has_R[k] and R_state[k] == 0 and u == 0:
                        R_state[k] = 1

        elif kind == "recover":
            k = chosen[1]
            I = I_k(k)
            u = rng.randrange(I)
            if u < I_bulk[k]:
                I_bulk[k] -= 1
                if model == 1: S_bulk[k] += 1
                else: R_bulk[k] += 1
            else:
                u -= I_bulk[k]
                if has_L[k] and L_state[k] == 1:
                    if u == 0: L_state[k] = 0 if model == 1 else 2
                    else:
                        u -= 1
                        if has_R[k] and R_state[k] == 1 and u == 0:
                            R_state[k] = 0 if model == 1 else 2
                else:
                    if has_R[k] and R_state[k] == 1 and u == 0:
                        R_state[k] = 0 if model == 1 else 2

        elif kind == "bridge_k_to_k1":
            k = chosen[1]
            L_state[k + 1] = 1

        elif kind == "bridge_k1_to_k":
            k = chosen[1]
            R_state[k] = 1

        times.append(t)
        totals.append(I_total())

    return np.array(times), np.array(totals)

# ------------------ Run + plot + save ------------------

params = dict(
    num_cliques=10,
    clique_size=100,
    edge_weight=1.0,
    initial_infected=0,
    infection_rate=0.2,
    recovery_rate=0.0,
    model=0,          # SI
    max_events=200000
)

baseG = build_chain_of_cliques(params["num_cliques"], params["clique_size"])

# Keep current plots (one representative run)
t_micro, I_micro, first_micro, bridge_micro = simulate_micro_chain_single(
    baseG, seed=1, **params
)
t_meta, I_meta, first_meta, bridge_meta = simulate_meta_chain_single(
    seed=2, **params
)

# Many runs for statistics
n_runs = 25
micro_times, micro_totals = [], []
meta_times, meta_totals = [], []

for i in range(n_runs):
    tm, Im = simulate_micro_total(baseG, seed=1000 + i, **{k: params[k] for k in params if k != "num_cliques" and k != "clique_size"})
    micro_times.append(tm); micro_totals.append(Im)

    tM, IM = simulate_meta_total(seed=2000 + i, **params)
    meta_times.append(tM); meta_totals.append(IM)

T_max = max(max(t[-1] for t in micro_times), max(t[-1] for t in meta_times))
grid = np.linspace(0.0, T_max, 500)

micro_q = quantile_bands(micro_times, micro_totals, grid, qs=(0.25, 0.5, 0.75))
meta_q  = quantile_bands(meta_times,  meta_totals,  grid, qs=(0.25, 0.5, 0.75))

# Plot: 3 panels (micro per clique, macro per clique, stats total infected)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 12))

for k in range(params["num_cliques"]):
    ax1.plot(t_micro, I_micro[k], label=f"Clique {k+1}")
    if first_micro[k] is not None:
        t0, y0, _ = first_micro[k]
        ax1.plot([t0], [y0], marker="X", linestyle="None")
    if bridge_micro[k] is not None:
        tb, yb, _ = bridge_micro[k]
        ax1.plot([tb], [yb], marker="^", linestyle="None")
ax1.set_title("Micro (node-level Gillespie on full network)")
ax1.set_xlabel("Simulated time")
ax1.set_ylabel("Infected in clique")
ax1.legend(ncol=5, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.02))

for k in range(params["num_cliques"]):
    ax2.plot(t_meta, I_meta[k], label=f"Clique {k+1}")
    if first_meta[k] is not None:
        t0, y0, _ = first_meta[k]
        ax2.plot([t0], [y0], marker="X", linestyle="None")
    if bridge_meta[k] is not None:
        tb, yb, _ = bridge_meta[k]
        ax2.plot([tb], [yb], marker="^", linestyle="None")
ax2.set_title("Macro/metapopulation (well-mixed cliques + explicit bridge nodes)")
ax2.set_xlabel("Simulated time")
ax2.set_ylabel("Infected in clique")

line_m, = ax3.plot(grid, micro_q[1], label="Micro median (total infected)")
ax3.fill_between(grid, micro_q[0], micro_q[2], alpha=0.2, color=line_m.get_color(),
                 label="Micro 25–75% quantiles")
line_M, = ax3.plot(grid, meta_q[1], label="Macro median (total infected)")
ax3.fill_between(grid, meta_q[0], meta_q[2], alpha=0.2, color=line_M.get_color(),
                 label="Macro 25–75% quantiles")
ax3.set_title(f"Total infected (all nodes): median + 25–75% quantile band over {n_runs} runs")
ax3.set_xlabel("Simulated time")
ax3.set_ylabel("Total infected")
ax3.legend(fontsize=9, ncol=2, loc="lower right")

fig.text(0.01, 0.005, "Markers: X = first infection in clique, ^ = right bridge-node infected (to next clique)", fontsize=10)
fig.tight_layout(rect=[0, 0.02, 1, 0.98])

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
outpath = f"/mnt/data/chain_cliques_micro_macro_with_stats_{timestamp}.png"
fig.savefig(outpath, dpi=200, bbox_inches="tight")
plt.close(fig)

(outpath, T_max, n_runs)
