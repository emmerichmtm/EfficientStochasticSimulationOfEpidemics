import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# -----------------------------
# Implement B) bridge nodes at END of clique
# Implement C) per-directed-meta-edge bridge infection markers
# Recompute plots and save PNG
# -----------------------------

def build_graph_of_cliques(metaG: nx.Graph, clique_size: int, internal_weight=1.0, bridge_weight=1.0):
    meta_nodes = list(metaG.nodes())
    meta_to_idx = {u: i for i, u in enumerate(meta_nodes)}
    K = len(meta_nodes)
    max_deg = max(dict(metaG.degree()).values()) if K else 0
    if clique_size <= max_deg:
        raise ValueError(f"clique_size ({clique_size}) must exceed max meta-degree ({max_deg}).")

    fullG = nx.Graph()
    clique_nodes: Dict[int, List[int]] = {}
    for u in meta_nodes:
        idx = meta_to_idx[u]
        nodes = list(range(idx * clique_size, (idx + 1) * clique_size))
        clique_nodes[idx] = nodes
        fullG.add_nodes_from(nodes)
        # clique edges
        for i in range(clique_size):
            a = nodes[i]
            for j in range(i+1, clique_size):
                b = nodes[j]
                fullG.add_edge(a, b, weight=float(internal_weight))

    # B) assign bridge nodes from END of clique
    bridge_node_meta: Dict[Any, Dict[Any, int]] = {u: {} for u in meta_nodes}
    bridge_nodes_list: Dict[int, List[int]] = {}

    for u in meta_nodes:
        idx = meta_to_idx[u]
        nodes = clique_nodes[idx]
        nbrs = sorted(list(metaG.neighbors(u)))
        bridge_nodes_list[idx] = []
        for k, v in enumerate(nbrs):
            bn = nodes[-1 - k]
            bridge_node_meta[u][v] = bn
            bridge_nodes_list[idx].append(bn)

    for u, v in metaG.edges():
        fullG.add_edge(bridge_node_meta[u][v], bridge_node_meta[v][u], weight=float(bridge_weight))

    # Also return a per-clique-index bridge map: bridge_node_idx[k][neighbor_meta] = node_id
    bridge_node_idx: List[Dict[Any, int]] = [dict() for _ in range(K)]
    for u in meta_nodes:
        u_idx = meta_to_idx[u]
        for v, nid in bridge_node_meta[u].items():
            bridge_node_idx[u_idx][v] = nid

    return fullG, meta_to_idx, bridge_node_idx, bridge_nodes_list


class EpidemicGraphFixed:
    def __init__(self, infection_rate=0.2, recovery_rate=0.0, model=0, rng=None):
        self.G = nx.Graph()
        self.model = model
        self.infection_rate = float(infection_rate)
        self.recovery_rate = float(recovery_rate)
        self.infected_nodes = []
        self.total_infection_rate = 0.0  # UNscaled SI weight sum
        self.total_recovery_rate = 0.0
        self.on_infect = None
        self.on_recover = None
        self.rng = rng or random.Random()

    def add_node(self, n):
        self.G.add_node(n, infected=False, recovered=False, sum_of_weights_i=0.0)

    def add_edge(self, u, v, w):
        self.G.add_edge(u, v, weight=float(w))

    def infect_node(self, node, event_time=0.0, event_iter=0):
        if self.G.nodes[node]["infected"] or self.G.nodes[node]["recovered"]:
            return
        self.G.nodes[node]["infected"] = True
        self.infected_nodes.append(node)
        self.total_recovery_rate += self.recovery_rate
        if self.on_infect:
            self.on_infect(node, event_time, event_iter)

        for nbr in self.G.neighbors(node):
            w = self.G[node][nbr]["weight"]
            if not self.G.nodes[nbr]["infected"]:
                self.G.nodes[node]["sum_of_weights_i"] += w
                self.total_infection_rate += w
            else:
                self.G.nodes[nbr]["sum_of_weights_i"] -= w
                self.total_infection_rate -= w

    def recover_node(self, node, event_time, event_iter):
        self.infected_nodes.remove(node)
        self.total_recovery_rate -= self.recovery_rate
        if self.model == 1:  # SIS
            for nbr in self.G.neighbors(node):
                self.G.nodes[nbr]["sum_of_weights_i"] += self.G[node][nbr]["weight"]
            self.total_infection_rate += self.G.nodes[node]["sum_of_weights_i"]
            self.G.nodes[node]["infected"] = False
            self.G.nodes[node]["sum_of_weights_i"] = 0.0
            if self.on_recover:
                self.on_recover(node, event_time, event_iter)
        elif self.model == 2:  # SIR
            self.G.nodes[node]["infected"] = False
            self.G.nodes[node]["recovered"] = True
            self.G.nodes[node]["sum_of_weights_i"] = 0.0
            if self.on_recover:
                self.on_recover(node, event_time, event_iter)

    def infect_neighbor(self, node, event_time, event_iter):
        nbrs = [n for n in self.G.neighbors(node) if (not self.G.nodes[n]["infected"]) and (not self.G.nodes[n]["recovered"])]
        if not nbrs:
            return
        weights = np.array([self.G[node][n]["weight"] for n in nbrs], dtype=float)
        tot = float(np.sum(weights))
        if tot <= 0:
            return
        target = self.rng.uniform(0.0, tot)
        cum = 0.0
        for n, w in zip(nbrs, weights):
            cum += float(w)
            if cum > target:
                self.infect_node(n, event_time, event_iter)
                break

    def simulate_step(self, current_time, event_iter):
        if not self.infected_nodes:
            return float("inf")
        scaled_infection = self.infection_rate * self.total_infection_rate
        scaled_recovery = 0.0 if self.model == 0 else self.total_recovery_rate
        total_rate = scaled_infection + scaled_recovery
        if total_rate <= 1e-12:
            return float("inf")

        dt = self.rng.expovariate(total_rate)
        t = current_time + dt
        r = self.rng.uniform(0.0, total_rate)

        if r < scaled_infection:
            target = self.rng.uniform(0.0, self.total_infection_rate)
            cum = 0.0
            for node in self.infected_nodes:
                cum += self.G.nodes[node]["sum_of_weights_i"]
                if cum > target:
                    self.infect_neighbor(node, t, event_iter)
                    break
        else:
            target = self.rng.uniform(0.0, self.total_recovery_rate)
            cum = 0.0
            for node in self.infected_nodes:
                cum += self.recovery_rate
                if cum > target:
                    self.recover_node(node, t, event_iter)
                    break

        return dt


def micro_single_run(fullG, clique_size, bridge_node_idx: List[Dict[Any, int]],
                    infection_rate=0.2, recovery_rate=0.0, model=0,
                    max_events=200000, seed=1, initial_clique_idx=0, initial_node_offset=0):
    rng = random.Random(seed)
    epi = EpidemicGraphFixed(infection_rate=infection_rate, recovery_rate=recovery_rate, model=model, rng=rng)

    for n in fullG.nodes:
        epi.add_node(n)
    for u, v, data in fullG.edges(data=True):
        epi.add_edge(u, v, data.get("weight", 1.0))

    K = len(bridge_node_idx)
    current = [0] * K
    first = [None] * K

    # C) per directed bridge infection times: list of dicts per clique index
    bridge_times: List[Dict[Any, Tuple[float, int, int]]] = [dict() for _ in range(K)]

    def clique_of(node_id: int) -> int:
        return node_id // clique_size

    # reverse lookup: node_id -> (clique_idx, neighbor_meta)
    node_to_bridge: Dict[int, List[Tuple[int, Any]]] = {}
    for k in range(K):
        for v, nid in bridge_node_idx[k].items():
            node_to_bridge.setdefault(nid, []).append((k, v))

    def on_inf(node_id, t, it):
        k = clique_of(node_id)
        current[k] += 1
        if first[k] is None:
            first[k] = (t, current[k], it)

        for (kk, v) in node_to_bridge.get(node_id, []):
            if v not in bridge_times[kk]:
                bridge_times[kk][v] = (t, current[kk], it)

    def on_rec(node_id, t, it):
        k = clique_of(node_id)
        current[k] -= 1

    epi.on_infect = on_inf
    epi.on_recover = on_rec

    # Seed in a bulk node (bridges are at the end now, so offset 0 is bulk)
    initial_node = initial_clique_idx * clique_size + initial_node_offset
    epi.infect_node(initial_node, event_time=0.0, event_iter=0)

    times = [0.0]
    series = [[ci] for ci in current]
    t = 0.0
    for it in range(1, max_events + 1):
        dt = epi.simulate_step(t, it)
        if not np.isfinite(dt):
            break
        t += dt
        times.append(t)
        for k in range(K):
            series[k].append(current[k])

    return np.array(times), [np.array(s) for s in series], first, bridge_times


def micro_total_run(fullG, infection_rate=0.2, recovery_rate=0.0, model=0,
                    max_events=200000, seed=1, initial_node=0):
    rng = random.Random(seed)
    epi = EpidemicGraphFixed(infection_rate=infection_rate, recovery_rate=recovery_rate, model=model, rng=rng)
    for n in fullG.nodes:
        epi.add_node(n)
    for u, v, data in fullG.edges(data=True):
        epi.add_edge(u, v, data.get("weight", 1.0))

    total_I = 0
    def on_inf(node_id, t, it):
        nonlocal total_I
        total_I += 1
    def on_rec(node_id, t, it):
        nonlocal total_I
        total_I -= 1

    epi.on_infect = on_inf
    epi.on_recover = on_rec

    epi.infect_node(initial_node, event_time=0.0, event_iter=0)
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


def macro_single_run(metaG, clique_size, infection_rate=0.2, recovery_rate=0.0, model=0,
                     bridge_weight=1.0, max_events=200000, seed=2, initial_meta_node=None):
    rng = random.Random(seed)
    beta = float(infection_rate)
    gamma = float(recovery_rate)

    meta_nodes = list(metaG.nodes())
    K = len(meta_nodes)
    deg = dict(metaG.degree())
    if clique_size <= max(deg.values()):
        raise ValueError("clique_size must exceed max meta-degree")

    S_bulk = {u: clique_size - deg[u] for u in meta_nodes}
    I_bulk = {u: 0 for u in meta_nodes}
    R_bulk = {u: 0 for u in meta_nodes}
    bridge_state = {u: {v: 0 for v in metaG.neighbors(u)} for u in meta_nodes}  # 0=S,1=I,2=R

    def I_u(u):
        return I_bulk[u] + sum(1 for v in bridge_state[u] if bridge_state[u][v] == 1)
    def S_u(u):
        return S_bulk[u] + sum(1 for v in bridge_state[u] if bridge_state[u][v] == 0)
    def infected_bridges(u):
        return [v for v in bridge_state[u] if bridge_state[u][v] == 1]
    def susceptible_bridges(u):
        return [v for v in bridge_state[u] if bridge_state[u][v] == 0]

    first = {u: None for u in meta_nodes}
    # C) per directed bridge infection times
    bridge_times = {u: {} for u in meta_nodes}

    if initial_meta_node is None:
        initial_meta_node = meta_nodes[0]
    I_bulk[initial_meta_node] += 1
    S_bulk[initial_meta_node] -= 1
    first[initial_meta_node] = (0.0, I_u(initial_meta_node), 0)

    times = [0.0]
    series = {u: [I_u(u)] for u in meta_nodes}
    t = 0.0

    for it in range(1, max_events + 1):
        rates, events = [], []

        for u in meta_nodes:
            I = I_u(u); S = S_u(u)
            if I > 0 and S > 0:
                rates.append(beta * I * S); events.append(("inf_internal", u))
            if model != 0 and I > 0 and gamma > 0:
                rates.append(gamma * I); events.append(("recover", u))

        for u, v in metaG.edges():
            if bridge_state[u][v] == 1 and bridge_state[v][u] == 0:
                rates.append(beta * bridge_weight); events.append(("bridge", u, v))
            if bridge_state[v][u] == 1 and bridge_state[u][v] == 0:
                rates.append(beta * bridge_weight); events.append(("bridge", v, u))

        total_rate = float(np.sum(rates))
        if total_rate <= 1e-12:
            break

        dt = rng.expovariate(total_rate); t += dt
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
            u = chosen[1]
            S = S_u(u)
            pick = rng.randrange(S)
            if pick < S_bulk[u]:
                S_bulk[u] -= 1; I_bulk[u] += 1
            else:
                pick -= S_bulk[u]
                sb = susceptible_bridges(u)
                if sb:
                    v = sb[pick]
                    bridge_state[u][v] = 1
                    bridge_times[u].setdefault(v, (t, I_u(u), it))
            if first[u] is None:
                first[u] = (t, I_u(u), it)

        elif kind == "recover":
            u = chosen[1]
            I = I_u(u)
            pick = rng.randrange(I)
            if pick < I_bulk[u]:
                I_bulk[u] -= 1
                if model == 1: S_bulk[u] += 1
                else: R_bulk[u] += 1
            else:
                pick -= I_bulk[u]
                ib = infected_bridges(u)
                v = ib[pick]
                bridge_state[u][v] = 0 if model == 1 else 2

        elif kind == "bridge":
            u, v = chosen[1], chosen[2]
            bridge_state[v][u] = 1
            bridge_times[v].setdefault(u, (t, I_u(v), it))
            if first[v] is None:
                first[v] = (t, I_u(v), it)

        times.append(t)
        for u in meta_nodes:
            series[u].append(I_u(u))

    series_list = [np.array(series[u]) for u in meta_nodes]
    first_list = [first[u] for u in meta_nodes]
    bridge_list = [bridge_times[u] for u in meta_nodes]
    return np.array(times), series_list, first_list, bridge_list


def macro_total_run(metaG, clique_size, infection_rate=0.2, recovery_rate=0.0, model=0,
                    bridge_weight=1.0, max_events=200000, seed=2, initial_meta_node=None):
    rng = random.Random(seed)
    beta = float(infection_rate)
    gamma = float(recovery_rate)

    meta_nodes = list(metaG.nodes())
    deg = dict(metaG.degree())
    if clique_size <= max(deg.values()):
        raise ValueError("clique_size must exceed max meta-degree")

    S_bulk = {u: clique_size - deg[u] for u in meta_nodes}
    I_bulk = {u: 0 for u in meta_nodes}
    R_bulk = {u: 0 for u in meta_nodes}
    bridge_state = {u: {v: 0 for v in metaG.neighbors(u)} for u in meta_nodes}

    def I_u(u):
        return I_bulk[u] + sum(1 for v in bridge_state[u] if bridge_state[u][v] == 1)
    def S_u(u):
        return S_bulk[u] + sum(1 for v in bridge_state[u] if bridge_state[u][v] == 0)

    if initial_meta_node is None:
        initial_meta_node = meta_nodes[0]
    I_bulk[initial_meta_node] += 1
    S_bulk[initial_meta_node] -= 1

    times = [0.0]
    totals = [sum(I_u(u) for u in meta_nodes)]
    t = 0.0

    for it in range(1, max_events + 1):
        rates, events = [], []

        for u in meta_nodes:
            I = I_u(u); S = S_u(u)
            if I > 0 and S > 0:
                rates.append(beta * I * S); events.append(("inf_internal", u))
            if model != 0 and I > 0 and gamma > 0:
                rates.append(gamma * I); events.append(("recover", u))

        for u, v in metaG.edges():
            if bridge_state[u][v] == 1 and bridge_state[v][u] == 0:
                rates.append(beta * bridge_weight); events.append(("bridge", u, v))
            if bridge_state[v][u] == 1 and bridge_state[u][v] == 0:
                rates.append(beta * bridge_weight); events.append(("bridge", v, u))

        total_rate = float(np.sum(rates))
        if total_rate <= 1e-12:
            break

        dt = rng.expovariate(total_rate); t += dt
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
            u = chosen[1]
            S = S_u(u)
            pick = rng.randrange(S)
            if pick < S_bulk[u]:
                S_bulk[u] -= 1; I_bulk[u] += 1
            else:
                pick -= S_bulk[u]
                sb = [v for v in bridge_state[u] if bridge_state[u][v] == 0]
                if sb:
                    v = sb[pick]
                    bridge_state[u][v] = 1

        elif kind == "recover":
            u = chosen[1]
            I = I_u(u)
            pick = rng.randrange(I)
            if pick < I_bulk[u]:
                I_bulk[u] -= 1
                if model == 1: S_bulk[u] += 1
                else: R_bulk[u] += 1
            else:
                pick -= I_bulk[u]
                ib = [v for v in bridge_state[u] if bridge_state[u][v] == 1]
                v = ib[pick]
                bridge_state[u][v] = 0 if model == 1 else 2

        elif kind == "bridge":
            u, v = chosen[1], chosen[2]
            bridge_state[v][u] = 1

        times.append(t)
        totals.append(sum(I_u(u) for u in meta_nodes))

    return np.array(times), np.array(totals)


def step_sample(times, values, grid):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    idx = np.searchsorted(times, grid, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]

def quantile_bands(runs_times, runs_vals, grid, qs=(0.25, 0.5, 0.75)):
    M = np.vstack([step_sample(t, v, grid) for t, v in zip(runs_times, runs_vals)])
    return np.quantile(M, qs, axis=0)


# ---- Run BA demo & save PNG ----
meta_n, meta_m = 10, 2
clique_size = 50
infection_rate, recovery_rate, model = 0.2, 0.0, 0
bridge_weight, internal_weight = 1.0, 1.0
n_runs, max_events, seed = 25, 200000, 1

metaG = nx.barabasi_albert_graph(meta_n, meta_m, seed=seed)
fullG, meta_to_idx, bridge_node_idx, bridge_nodes_list = build_graph_of_cliques(metaG, clique_size, internal_weight, bridge_weight)

t_micro, I_micro, first_micro, bridge_micro = micro_single_run(
    fullG, clique_size, bridge_node_idx,
    infection_rate, recovery_rate, model, max_events, seed, 0, 0
)

meta_nodes = list(metaG.nodes())
t_macro, I_macro, first_macro, bridge_macro = macro_single_run(
    metaG, clique_size, infection_rate, recovery_rate, model, bridge_weight, max_events, seed+1, meta_nodes[0]
)

# stats
micro_times, micro_totals, macro_times, macro_totals = [], [], [], []
for i in range(n_runs):
    tm, Im = micro_total_run(fullG, infection_rate, recovery_rate, model, max_events, 1000+seed+i, 0)
    micro_times.append(tm); micro_totals.append(Im)
    tM, IM = macro_total_run(metaG, clique_size, infection_rate, recovery_rate, model, bridge_weight, max_events, 2000+seed+i, meta_nodes[0])
    macro_times.append(tM); macro_totals.append(IM)

T_max = max(max(t[-1] for t in micro_times), max(t[-1] for t in macro_times))
grid = np.linspace(0.0, T_max, 500)
micro_q = quantile_bands(micro_times, micro_totals, grid)
macro_q = quantile_bands(macro_times, macro_totals, grid)

# plot
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(13, 12))
degs = [d for _, d in metaG.degree()]
ax0.set_title(f"Micro (node-level) — BA meta-graph: n={meta_n}, m={meta_m}, degrees={degs}")

for k in range(meta_n):
    ax0.plot(t_micro, I_micro[k], label=f"Clique {k}")
    if first_micro[k] is not None:
        t0, y0, _ = first_micro[k]
        ax0.plot([t0], [y0], marker="X", linestyle="None")
    for v, (tb, yb, _) in bridge_micro[k].items():
        ax0.plot([tb], [yb], marker="^", linestyle="None", markersize=5)

ax0.set_xlabel("Simulated time")
ax0.set_ylabel("Infected in clique")
ax0.legend(ncol=5, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.02))

ax1.set_title("Macro/metapopulation (well-mixed cliques + explicit bridge nodes)")
for k in range(meta_n):
    ax1.plot(t_macro, I_macro[k], label=f"Clique {k}")
    if first_macro[k] is not None:
        t0, y0, _ = first_macro[k]
        ax1.plot([t0], [y0], marker="X", linestyle="None")
    for v, (tb, yb, _) in bridge_macro[k].items():
        ax1.plot([tb], [yb], marker="^", linestyle="None", markersize=5)

ax1.set_xlabel("Simulated time")
ax1.set_ylabel("Infected in clique")

line_m, = ax2.plot(grid, micro_q[1], label="Micro median (total infected)")
ax2.fill_between(grid, micro_q[0], micro_q[2], alpha=0.2, color=line_m.get_color(), label="Micro 25–75% quantiles")
line_M, = ax2.plot(grid, macro_q[1], label="Macro median (total infected)")
ax2.fill_between(grid, macro_q[0], macro_q[2], alpha=0.2, color=line_M.get_color(), label="Macro 25–75% quantiles")

ax2.set_title(f"Total infected: median + 25–75% band over {n_runs} runs")
ax2.set_xlabel("Simulated time")
ax2.set_ylabel("Total infected")
ax2.legend(ncol=2, fontsize=9, loc="lower right")

fig.text(0.01, 0.006, "Markers: X = first infection in clique, ^ = infection of a specific bridge node (one per incident meta-edge)", fontsize=10)
fig.tight_layout(rect=[0, 0.02, 1, 0.98])

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
outpath = Path(f"/mnt/data/ba_metapop_cliques_micro_macro_BC_{ts}.png")
fig.savefig(outpath, dpi=200, bbox_inches="tight")
plt.close(fig)

# Also write an updated script file to /mnt/data for you
script_text = """
# main_chain_of_cliques_generalized.py (B+C version)
# - Bridge nodes assigned at end of each clique (avoid seeding on a bridge node by default)
# - Per-directed-edge bridge infection markers (one per incident meta-edge)
#
# Generated by ChatGPT in this session.
"""
# For brevity, we don't embed the whole notebook code here; you can copy it from the chat if needed.
Path("/mnt/data/main_chain_of_cliques_generalized_BC_notes.txt").write_text(script_text, encoding="utf-8")

str(outpath)

