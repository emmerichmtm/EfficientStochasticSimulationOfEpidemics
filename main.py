import numpy as np
#from numba import njit
import networkx as nx
import matplotlib.pyplot as plt
import random
from sortedcontainers import SortedList
import logging

# Set up logging
# Put level to INFO to see details
logging.basicConfig(level=logging.CRITICAL)

class EpidemicGraph:
    def __init__(self, infection_rate=0.1, recovery_rate=0.1):
        self.G = nx.Graph()
        self.infection_rate = infection_rate
        self.infected_nodes = SortedList(key=lambda x: -self.G.nodes[x]['sum_of_weights_i'])
        self.total_infection_rate = 0
        self.total_recovery_rate = 0

    def add_node(self, node_id):
        self.G.add_node(node_id, infected=False, sum_of_weights_i=0.0)

    def add_edge(self, node1, node2, weight):
        self.G.add_edge(node1, node2, weight=weight)

    def simulate_step(self):
        if not self.infected_nodes or self.total_infection_rate <= 0:
            return float('inf')
        wait_time = random.expovariate(self.total_infection_rate+self.total_recovery_rate)
        # recovery event or infection event
        # choose a random number between 0 and total rate
        r_or_i = random.uniform(0, self.total_infection_rate+self.total_recovery_rate)
        if r_or_i < self.total_infection_rate: # infection event
            r = random.uniform(0, self.total_infection_rate)
            logging.info(f'Random threshold: {r}, Total weight: {self.total_infection_rate}, Infection rate: {self.infection_rate}')
            cumulative = 0
            for node in self.infected_nodes:
                cumulative += self.G.nodes[node]['sum_of_weights_i']
                logging.info(f'Node: {node}, Cumulative: {cumulative}')
                if cumulative > r:
                    self.infect_neighbor(node)
                    break
            logging.debug(f'Total weight: {self.total_infection_rate}, Rate: {self.infection_rate}, Random threshold: {r}')
        else: # recovery event
            r = random.uniform(0, self.total_recovery_rate)
            cumulative = 0
            node_to_recover = None
            for node in self.infected_nodes:
                cumulative += self.recovery_rate
                if cumulative > r and node in self.infected_nodes:
                    node_to_recover = node
                    break
            if (node_to_recover):
                if node_to_recover not in self.infected_nodes:
                    logging.critical(f"Node {node_to_recover} not in infected nodes list (sim level).")
                    logging.critical(f"Infected nodes: {self.infected_nodes}")
                else:
                    self.recover_node(node_to_recover)
        return wait_time

    def recover_node(self, node):
        self.G.nodes[node]['infected'] = False
        if node in self.infected_nodes:
           self.infected_nodes.remove(node)
        else:
            logging.critical(f"Node {node} not in infected nodes list.")
        self.total_recovery_rate -= self.recovery_rate
        # REDUCE THE TOTAL INFECTION RATE
        self.total_infection_rate -= self.G.nodes[node]['sum_of_weights_i']*self.infection_rate
        # Increase the weight_i of neighbors, because they can now reinfect node
        for neighbor in self.G.neighbors(node):
            if self.G.nodes[neighbor]['infected']:
                self.G.nodes[neighbor]['sum_of_weights_i'] += self.G[node][neighbor]['weight']
                self.total_infection_rate += self.G[node][neighbor]['weight']*self.infection_rate

    def infect_neighbor(self, infected_node):
        neighbors = [n for n in self.G.neighbors(infected_node) if not self.G.nodes[n]['infected']]
        if neighbors:
            weights = np.array([self.G[infected_node][n]['weight'] for n in neighbors])
            neighbor_to_infect = choose_neighbor_to_infect(weights)
            self.infect_node(neighbors[neighbor_to_infect])

    def infect_node(self, node):
        self.G.nodes[node]['infected'] = True
        self.infected_nodes.add(node)
        self.total_recovery_rate += self.recovery_rate
        for neighbor in self.G.neighbors(node):
            if not self.G.nodes[neighbor]['infected']:
                self.G.nodes[node]['sum_of_weights_i'] += self.G[node][neighbor]['weight']
                self.total_infection_rate += self.G[node][neighbor]['weight']
            else:
                self.G.nodes[neighbor]['sum_of_weights_i'] -= self.G[node][neighbor]['weight']
                self.total_infection_rate -= self.G[node][neighbor]['weight']

    def plot_graph(self, title="Graph"):
        plt.figure(figsize=(8, 5))
        pos = nx.spring_layout(self.G)
        colors = ['red' if self.G.nodes[node]['infected'] else 'green' for node in self.G.nodes()]
        nx.draw(self.G, pos, node_color=colors, with_labels=True, node_size=700)
        nx.draw_networkx_edges(self.G, pos, width=1.0, alpha=0.5)
        plt.title(title)
        plt.show()

# Function optimized with Numba for choosing the neighbor to infect
#@njit
def choose_neighbor_to_infect(weights):
    total_weight = np.sum(weights)
    r = random.uniform(0, total_weight)
    cumulative = 0
    for i in range(len(weights)):
        cumulative += weights[i]
        if cumulative > r:
            return i

# Test Small Network
def test_small_network():
    logging.info("Starting small network test...")
    graph = EpidemicGraph(infection_rate=0.1)

    # Create a small graph
    for i in range(1, 6):
        graph.add_node(i)
    graph.add_edge(1, 2, 1.0)
    graph.add_edge(2, 3, 1.0)
    graph.add_edge(3, 4, 1.0)
    graph.add_edge(4, 5, 1.0)
    graph.add_edge(2, 5, 1.0)

    # Plot before infection
    graph.plot_graph("Small Graph Before Infection")
    graph.infect_node(1)  # Initial infection

    steps = 3
    for i in range(steps):
        graph.simulate_step()  # Simulate one step

    # Plot after initial infection
    graph.plot_graph("Small Graph After Infection")


# Test Large Barabasi-Albert Network
def test_large_network(model="barabasi_albert"):
    logging.info(f"Starting large network test with {model} model...")

    infection_rate = 0.1
    num_nodes = 1000

    # Create the graph based on the selected model
    if model == "barabasi_albert":
        num_edges = 2  # Default number of edges to attach in Barabási-Albert
        ba_graph = nx.barabasi_albert_graph(num_nodes, num_edges)
    elif model == "erdos_renyi":
        p = 0.01  # Default probability for Erdos-Renyi graph
        ba_graph = nx.erdos_renyi_graph(num_nodes, p)
    elif model == "watts_strogatz":
        k = 4  # Each node is connected to k nearest neighbors in ring topology
        p = 0.1  # The probability of rewiring each edge
        ba_graph = nx.watts_strogatz_graph(num_nodes, k, p)
    else:
        raise ValueError("Invalid model type. Choose from 'barabasi_albert', 'erdos_renyi', or 'watts_strogatz'.")

    # Initialize the epidemic graph
    graph = EpidemicGraph(infection_rate)

    # Add nodes and edges from the generated network to our epidemic graph
    for node in ba_graph.nodes:
        graph.add_node(node)
    for edge in ba_graph.edges:
        graph.add_edge(edge[0], edge[1], 1.0)  # Add edges with weight 1.0

    # Infect an initial node
    graph.infect_node(0)

    infections_over_time = []
    simulated_time = []
    total_time = 0.0
    time_steps = 10000

    # Simulate infection spread over time
    for _ in range(time_steps):
        infections_over_time.append(len([n for n in graph.G.nodes if graph.G.nodes[n]['infected']]))
        wait_time = graph.simulate_step()  # Get the wait time for the next infection
        total_time += wait_time
        simulated_time.append(total_time)  # Store the cumulative simulated time

    # Plot the number of infected nodes over simulated time
    plt.figure(figsize=(10, 6))
    plt.scatter(simulated_time, infections_over_time)
    plt.xlabel("Simulated Time")
    plt.ylabel("Number of Infected Nodes")
    plt.title(f"Infection Spread Over Simulated Time in {model.replace('_', ' ').title()} Network")
    plt.show()

# Example usage:
if __name__ == "__main__":
    # You can switch between models by passing 'barabasi_albert', 'erdos_renyi', or 'watts_strogatz'
    test_small_network()
    test_large_network("barabasi_albert")  # Barabási-Albert model
    test_large_network("erdos_renyi")  # Erdős-Rényi model
    test_large_network("watts_strogatz")  # Watts-Strogatz model
