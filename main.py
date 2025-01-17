# Copyright (c) 2025 Michael T.M. Emmerich
# Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
# You are free to share and adapt this work with attribution.
# Provided "as is", without warranty of any kind.

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


class EpidemicGraph:
    def __init__(self, infection_rate=0.1, recovery_rate=0, model=2):
        self.G = nx.Graph()
        self.model = model  # 1: SIS, 2: SIR
        self.infection_rate, self.recovery_rate = infection_rate, recovery_rate
        self.infected_nodes = []
        self.total_infection_rate, self.total_recovery_rate = 0, 0

    def add_node(self, node_id):
        self.G.add_node(node_id, infected=False, recovered=False, sum_of_weights_i=0.0)

    def add_edge(self, node1, node2, weight):
        self.G.add_edge(node1, node2, weight=weight)

    def simulate_step(self):
        if not self.infected_nodes: return float('inf')
        total_rate = self.total_infection_rate + self.total_recovery_rate
        wait_time = random.expovariate(total_rate)
        r = random.uniform(0, total_rate)
        if r < self.total_infection_rate:  # Infection event
            target = random.uniform(0, self.total_infection_rate)
            cumulative = 0
            for node in self.infected_nodes:
                cumulative += self.G.nodes[node]['sum_of_weights_i']
                if cumulative > target: self.infect_neighbor(node); break
        else:  # Recovery event
            target = random.uniform(0, self.total_recovery_rate)
            cumulative = 0
            for node in self.infected_nodes:
                cumulative += self.recovery_rate
                if cumulative > target: self.recover_node(node); break
        return wait_time

    def recover_node(self, node):
        self.infected_nodes.remove(node)
        self.total_recovery_rate -= self.recovery_rate
        if self.model == 1:  # SIS
            for neighbor in self.G.neighbors(node):
                self.G.nodes[neighbor]['sum_of_weights_i'] += self.G[node][neighbor]['weight']
            self.total_infection_rate += self.G.nodes[node]['sum_of_weights_i']
            self.G.nodes[node]['infected'] = False
        elif self.model == 2:  # SIR
            self.G.nodes[node]['recovered'], self.G.nodes[node]['infected'] = True, False

    def infect_neighbor(self, node):
        neighbors = [n for n in self.G.neighbors(node) if
                     not self.G.nodes[n]['infected'] and not self.G.nodes[n]['recovered']]
        if neighbors:
            weights = np.array([self.G[node][n]['weight'] for n in neighbors])
            cumulative, target = 0, random.uniform(0, np.sum(weights))
            for i, weight in enumerate(weights):
                cumulative += weight
                if cumulative > target: self.infect_node(neighbors[i]); break

    def infect_node(self, node):
        self.G.nodes[node]['infected'] = True
        self.infected_nodes.append(node)
        self.total_recovery_rate += self.recovery_rate
        for neighbor in self.G.neighbors(node):
            if not self.G.nodes[neighbor]['infected']:
                self.G.nodes[node]['sum_of_weights_i'] += self.G[node][neighbor]['weight']
                self.total_infection_rate += self.G[node][neighbor]['weight']

    def plot_graph(self, title="Graph"):
        pos = nx.spring_layout(self.G)
        colors = ['red' if self.G.nodes[node]['infected'] else 'green' for node in self.G.nodes()]
        nx.draw(self.G, pos, node_color=colors, with_labels=True, node_size=700)
        nx.draw_networkx_edges(self.G, pos, width=1.0, alpha=0.5)
        plt.title(title), plt.show()


# Test small graph
def test_small_network():
    graph = EpidemicGraph(infection_rate=0.1)
    for i in range(1, 6): graph.add_node(i)
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (2, 5)]
    for edge in edges: graph.add_edge(edge[0], edge[1], 1.0)
    graph.plot_graph("Before Infection")
    graph.infect_node(1)
    for _ in range(3): graph.simulate_step()
    graph.plot_graph("After Infection")


# Test large networks
def test_large_network(model="barabasi_albert"):
    num_nodes = 1000
    if model == "barabasi_albert":
        graph_instance = nx.barabasi_albert_graph(num_nodes, 2)
    elif model == "erdos_renyi":
        graph_instance = nx.erdos_renyi_graph(num_nodes, 0.01)
    elif model == "watts_strogatz":
        graph_instance = nx.watts_strogatz_graph(num_nodes, 4, 0.1)
    else:
        raise ValueError("Invalid model.")

    graph = EpidemicGraph(0.2, 0.8, model = 2)
    for node in graph_instance.nodes: graph.add_node(node)
    for edge in graph_instance.edges: graph.add_edge(edge[0], edge[1], 1.0)
    graph.infect_node(0)

    infections_over_time, simulated_time, total_time = [], [], 0.0
    for _ in range(10000):
        infections_over_time.append(len([n for n in graph.G.nodes if graph.G.nodes[n]['infected']]))
        wait_time = graph.simulate_step()
        total_time += wait_time
        simulated_time.append(total_time)

    plt.scatter(simulated_time, infections_over_time)
    plt.xlabel("Simulated Time"), plt.ylabel("Infected Nodes")
    plt.title(f"Infection Spread in {model.replace('_', ' ').title()} Network"), plt.show()


# Run tests
if __name__ == "__main__":
    test_small_network()
    test_large_network("barabasi_albert")
    test_large_network("erdos_renyi")
    test_large_network("watts_strogatz")
