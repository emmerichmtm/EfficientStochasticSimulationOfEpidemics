import numpy as np
from numba import njit
import networkx as nx
import matplotlib.pyplot as plt
import random
from sortedcontainers import SortedList
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class EpidemicGraph:
    def __init__(self, infection_rate):
        self.G = nx.Graph()
        self.infection_rate = infection_rate
        self.infected_nodes = SortedList(key=lambda x: -self.G.nodes[x]['sum_of_weights_i'])
        self.total_rate = 0

    def add_node(self, node_id):
        self.G.add_node(node_id, infected=False, sum_of_weights_i=0.0)

    def add_edge(self, node1, node2, weight):
        self.G.add_edge(node1, node2, weight=weight)

    def simulate_step(self):
        if not self.infected_nodes or self.total_rate <= 0:
            return float('inf')
        wait_time = random.expovariate(self.infection_rate * self.total_rate)
        r = random.uniform(0, self.total_rate)
        logging.info(f'Random threshold: {r}, Total weight: {self.total_rate}, Infection rate: {self.infection_rate}')
        cumulative = 0
        for node in self.infected_nodes:
            cumulative += self.G.nodes[node]['sum_of_weights_i']
            logging.info(f'Node: {node}, Cumulative: {cumulative}')
            if cumulative > r:
                self.infect_neighbor(node)
                break
        logging.debug(f'Total weight: {self.total_rate}, Rate: {self.infection_rate}, Random threshold: {r}')
        return wait_time

    def infect_neighbor(self, infected_node):
        neighbors = [n for n in self.G.neighbors(infected_node) if not self.G.nodes[n]['infected']]
        if neighbors:
            weights = np.array([self.G[infected_node][n]['weight'] for n in neighbors])
            neighbor_to_infect = choose_neighbor_to_infect(weights)
            self.infect_node(neighbors[neighbor_to_infect])

    def infect_node(self, node):
        self.G.nodes[node]['infected'] = True
        self.infected_nodes.add(node)
        for neighbor in self.G.neighbors(node):
            if not self.G.nodes[neighbor]['infected']:
                self.G.nodes[node]['sum_of_weights_i'] += self.G[node][neighbor]['weight']
                self.total_rate += self.G[node][neighbor]['weight']
            else:
                self.G.nodes[neighbor]['sum_of_weights_i'] -= self.G[node][neighbor]['weight']
                self.total_rate -= self.G[node][neighbor]['weight']

    def plot_graph(self, title="Graph"):
        plt.figure(figsize=(8, 5))
        pos = nx.spring_layout(self.G)
        colors = ['red' if self.G.nodes[node]['infected'] else 'green' for node in self.G.nodes()]
        nx.draw(self.G, pos, node_color=colors, with_labels=True, node_size=700)
        nx.draw_networkx_edges(self.G, pos, width=1.0, alpha=0.5)
        plt.title(title)
        plt.show()

# Function optimized with Numba for choosing the neighbor to infect
@njit
def choose_neighbor_to_infect(weights):
    total_weight = np.sum(weights)
    r = random.uniform(0, total_weight)
    cumulative = 0
    for i in range(len(weights)):
        cumulative += weights[i]
        if cumulative > r:
            return i

# Example Usage
graph = EpidemicGraph(infection_rate=0.1)
for i in range(1, 6):
    graph.add_node(i)
graph.add_edge(1, 2, 1.0)
graph.add_edge(2, 3, 1.0)
graph.add_edge(3, 4, 1.0)
graph.add_edge(4, 5, 1.0)
graph.add_edge(2, 5, 1.0)

# Plot before infection
graph.plot_graph("Graph Before Infection")
logging.info("Infecting node 1...")
graph.infect_node(1)  # Initial infection

steps = 3
for i in range(steps):
    logging.info(f"Simulating step {i + 1}...")
    wait_time = graph.simulate_step()  # Simulate one step
    logging.info(f"Wait time: {wait_time}")

# Plot after initial infection
graph.plot_graph("Graph After Infection")
