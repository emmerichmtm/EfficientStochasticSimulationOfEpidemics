import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

class EpidemicGraph:
    def __init__(self, infection_rate=0.1, recovery_rate=0, model=1):
        # model: 0: SI, 1: SIS, 2: SIR
        self.G = nx.Graph()
        self.model = model
        self.infection_rate = float(infection_rate)   # beta
        self.recovery_rate  = float(recovery_rate)    # gamma

        self.infected_nodes = []
        # NOTE: total_infection_rate = sum of SI edge weights (UNSCALED)
        self.total_infection_rate = 0.0
        # total_recovery_rate = gamma * |I|  (already scaled by gamma)
        self.total_recovery_rate = 0.0

    def add_node(self, node_id):
        self.G.add_node(node_id, infected=False,
                        recovered=False, sum_of_weights_i=0.0)

    def add_edge(self, node1, node2, weight):
        self.G.add_edge(node1, node2, weight=float(weight))

    def simulate_step(self):
        if not self.infected_nodes:
            return float('inf')

        # --- FIX: apply beta to infection hazard ---
        scaled_infection = self.infection_rate * self.total_infection_rate
        scaled_recovery  = 0.0 if self.model == 0 else self.total_recovery_rate
        total_rate = scaled_infection + scaled_recovery

        if total_rate < 1e-12:
            return float('inf')

        wait_time = random.expovariate(total_rate)

        r = random.uniform(0.0, total_rate)

        if r < scaled_infection:  # Infection event
            # Choose infected node proportional to its UNscaled SI-weight sum;
            # beta cancels out in conditional selection.
            target = random.uniform(0.0, self.total_infection_rate)
            cumulative = 0.0
            for node in self.infected_nodes:
                cumulative += self.G.nodes[node]['sum_of_weights_i']
                if cumulative > target:
                    self.infect_neighbor(node)
                    break

        else:  # Recovery event (SIS/SIR)
            if self.total_recovery_rate <= 1e-12:
                return wait_time

            target = random.uniform(0.0, self.total_recovery_rate)
            cumulative = 0.0  # <-- consistent recovery selection
            for node in self.infected_nodes:
                cumulative += self.recovery_rate
                if cumulative > target:
                    self.recover_node(node)
                    break

        return wait_time

    def recover_node(self, node):
        self.infected_nodes.remove(node)
        self.total_recovery_rate -= self.recovery_rate

        if self.model == 1:  # SIS
            for neighbor in self.G.neighbors(node):
                self.G.nodes[neighbor]['sum_of_weights_i'] += self.G[node][neighbor]['weight']
            self.total_infection_rate += self.G.nodes[node]['sum_of_weights_i']
            self.G.nodes[node]['infected'] = False
            self.G.nodes[node]['sum_of_weights_i'] = 0.0

        elif self.model == 2:  # SIR
            self.G.nodes[node]['recovered'] = True
            self.G.nodes[node]['infected'] = False
            self.G.nodes[node]['sum_of_weights_i'] = 0.0

    def infect_neighbor(self, node):
        neighbors = [
            n for n in self.G.neighbors(node)
            if (not self.G.nodes[n]['infected']) and (not self.G.nodes[n]['recovered'])
        ]
        if neighbors:
            weights = np.array([self.G[node][n]['weight'] for n in neighbors], dtype=float)
            total_w = float(np.sum(weights))
            if total_w <= 0.0:
                return
            cumulative, target = 0.0, random.uniform(0.0, total_w)
            for i, weight in enumerate(weights):
                cumulative += float(weight)
                if cumulative > target:
                    self.infect_node(neighbors[i])
                    break

    def infect_node(self, node):
        self.G.nodes[node]['infected'] = True
        self.infected_nodes.append(node)
        self.total_recovery_rate += self.recovery_rate

        for neighbor in self.G.neighbors(node):
            if not self.G.nodes[neighbor]['infected']:
                w = self.G[node][neighbor]['weight']
                self.G.nodes[node]['sum_of_weights_i'] += w
                self.total_infection_rate += w
            elif neighbor != node:
                # neighbor is infected => remove edge contribution from neighbor’s SI-sum
                w = self.G[node][neighbor]['weight']
                self.G.nodes[neighbor]['sum_of_weights_i'] -= w
                self.total_infection_rate -= w

    def plot_graph(self, title="Graph", scale=300):
        pos = nx.spring_layout(self.G)
        colors = ['red' if self.G.nodes[n]['infected'] else 'green'
                  for n in self.G.nodes()]

        sizes = [
            (self.G.nodes[n].get('sum_of_weights_i', 0.0) + 1.0) * scale
            for n in self.G.nodes()
        ]

        nx.draw(self.G, pos,
                node_color=colors,
                with_labels=True,
                node_size=sizes)

        nx.draw_networkx_edges(self.G, pos, width=1.0, alpha=0.5)

        # show both unscaled and scaled hazards (optional)
        plt.title(
            f"{title} — SI-weight sum: {self.total_infection_rate:.2f}, "
            f"β·SI: {self.infection_rate*self.total_infection_rate:.2f}"
        )
        plt.show()
