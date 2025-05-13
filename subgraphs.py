import networkx as nx
import pickle


with open("graph.pkl", "rb") as f:
    monthly_graphs = pickle.load(f)

monthly_nodes = {month: set(G.nodes()) for month, G in enumerate(monthly_graphs)}
common_nodes = set.intersection(*monthly_nodes.values())

filtered_monthly_graphs = []

for month, G in enumerate(monthly_graphs):
    subgraph = G.subgraph(common_nodes).copy()
    filtered_monthly_graphs.append(subgraph)

with open("subgraphs.pkl", "wb") as f:
    pickle.dump(filtered_monthly_graphs, f)
