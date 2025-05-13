import matplotlib.pyplot as plt
import networkx as nx
import pickle

with open("subgraphs.pkl", "rb") as f:
    subgraphs = pickle.load(f)


for i, G in enumerate(subgraphs):
    plt.figure(figsize=(20,20))
    pos = nx.spring_layout(G, k=.1)
    nx.draw_networkx(G, pos, node_size=25, node_color='red', with_labels=True, edge_color='blue')
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    plt.savefig("plots/subGmonth" + str(i) + ".png")