import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sortedcontainers import SortedList
import random
class EpidemicGraph:
    def __init__(self, infection_rate):
        self.G = nx.Graph()
        self.infection_rate = infection_rate
        self.infected_nodes = SortedList(key=lambda x: -self.G.nodes[x]['sum_of_weights_i'])
        self.total_weight = 0

    def add_node(self, node_id):
        self.G.add_node(node_id, infected=False, sum_of_weights_i=0.0)

    def add_edge(self, node1, node2, weight):
        self.G.add_edge(node1, node2, weight=weight)

    # Simulate the next step of the epidemic
    def simulate_step(self):
        if not self.infected_nodes or self.total_weight <= 0:
            return float('inf')
        rate = self.infection_rate
        wait_time = random.expovariate(rate*self.total_weight)
        cumulative = 0
        print('total weight:', self.total_weight)
        print('rate:', rate)
        r = random.uniform(0, self.total_weight)
        print('r:', r)
        for node in self.infected_nodes:
            print('sum of weights:', self.G.nodes[node]['sum_of_weights_i'])

        for node in self.infected_nodes:
            print('sum of weights:', self.G.nodes[node]['sum_of_weights_i'])
            cumulative += self.G.nodes[node]['sum_of_weights_i']
            if cumulative > r:
                infecting_node = node
                break
        self.infect_neighbor(infecting_node)
        return wait_time

    def infect_neighbor(self, infected_node):
        neighbors = [n for n in self.G.neighbors(infected_node) if not self.G.nodes[n]['infected']]
        weights = [self.G[infected_node][n]['weight'] for n in neighbors]
        total_neighbor_weight = sum(weights)
        if total_neighbor_weight > 0:
            r = random.uniform(0, total_neighbor_weight)
            cumulative = 0
            for neighbor, weight in zip(neighbors, weights):
                cumulative += weight
                if cumulative > r:
                    self.infect_node(neighbor)
                    print('infecting neighbor:', neighbor)
                    break

    def infect_node(self, node):
        self.G.nodes[node]['infected'] = True
        self.infected_nodes.add(node)

        # Update the sum of weights of the neighbors
        for neighbor in self.G.neighbors(node):
            if not self.G.nodes[neighbor]['infected']:
                self.G.nodes[node]['sum_of_weights_i'] += self.G[node][neighbor]['weight']
                self.total_weight += self.G.nodes[node]['sum_of_weights_i']
            else:
                self.G.nodes[neighbor]['sum_of_weights_i'] -= self.G[node][neighbor]['weight']
                self.total_weight -= self.G[node][neighbor]['weight']
    def plot_graph(self, title="Graph"):
        plt.figure(figsize=(8, 5))
        pos = nx.spring_layout(self.G)
        colors = ['red' if self.G.nodes[node]['infected'] else 'green' for node in self.G.nodes()]
        nx.draw(self.G, pos, node_color=colors, with_labels=True, node_size=700)
        nx.draw_networkx_edges(self.G, pos, width=1.0, alpha=0.5)
        plt.title(title)
        plt.show()  #


# Example Usage
graph = EpidemicGraph(infection_rate=0.1)
for i in range(1, 6):
    graph.add_node(i)
graph.add_edge(1, 2, 1.0)
graph.add_edge(2, 3, 1.0)
graph.add_edge(3, 4, 1.0)
graph.add_edge(4, 5, 1.0)

# Plot before infection
graph.plot_graph("Graph Before Infection")

graph.infect_node(1)  # Initial infection
waittime = graph.simulate_step()  # Simulate one step
print("Wait time:", waittime)
waittime = graph.simulate_step()  # Simulate one step
print("Wait time:", waittime)
waittime = graph.simulate_step()  # Simulate one step

# Plot after initial infection

graph.plot_graph("Graph After Infection")
