import matplotlib.pyplot as plt
import networkx as nx
import pickle

with open("graph.pkl", "rb") as f:
    G = pickle.load(f)

plt.figure(figsize=(20,20))
pos = nx.spring_layout(G, k=.1)
nx.draw_networkx(G, pos, node_size=25, node_color='red', with_labels=True, edge_color='blue')
plt.show()