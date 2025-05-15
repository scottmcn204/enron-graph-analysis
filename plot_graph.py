import matplotlib.pyplot as plt
import networkx as nx
import pickle

with open("subgraphs.pkl", "rb") as f:
    subgraphs = pickle.load(f)

with open("predicted_edges.pkl", "rb") as f:
    predicted_edges = pickle.load(f)

with open("actual_edges.pkl", "rb") as f:
    actual_edges = pickle.load(f)

def visualize_subgraphs():
    for i, G in enumerate(subgraphs):
        plt.figure(figsize=(20,20))
        pos = nx.spring_layout(G, k=.1)
        nx.draw_networkx(G, pos, node_size=25, node_color='red', with_labels=True, edge_color='blue')
        labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        plt.savefig("plots/subGmonth" + str(i) + ".png")

def visualize_predictions(G_true, actual_edges, predicted_edges, save_path="plots/prediction_visual.png"):
    pos = nx.spring_layout(G_true, seed=42, k=1.5)
    predicted_edges_set = set(tuple(sorted(clean_edge(e))) for e in predicted_edges)
    actual_edges_set = set(tuple(sorted(clean_edge(e))) for e in actual_edges)
    true_positives = predicted_edges_set & actual_edges_set
    false_positives = predicted_edges_set - actual_edges_set
    false_negatives = actual_edges_set - predicted_edges_set
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G_true, pos, node_size=30, node_color='darkblue')
    nx.draw_networkx_edges(G_true, pos, edgelist=true_positives, edge_color='green', width=1, label="True Positives")
    nx.draw_networkx_edges(G_true, pos, edgelist=false_positives, edge_color='red', style='dashed', width=1, label="False Positives")
    nx.draw_networkx_edges(G_true, pos, edgelist=false_negatives, edge_color='black', style='dotted', width=0.8, label="False Negatives")
    plt.legend()
    plt.title("Link Prediction Visualisation")
    plt.axis('off')
    plt.savefig(save_path)

def clean_edge(edge):
    if isinstance(edge, str):
        return eval(edge)  # turn "('alice', 'bob')" into ('alice', 'bob')
    return edge

G_eval = subgraphs[17].to_undirected()
visualize_predictions(G_eval, actual_edges, predicted_edges)