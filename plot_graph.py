import matplotlib.pyplot as plt
import networkx as nx
import pickle

with open("graph.pkl", "rb") as f:
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

def visualize_predictions(G_true, actual_edges, predicted_edges, save_path="plots/prediction_visual_top100.png"):
    pagerank = nx.pagerank(G_true)
    top_nodes = set(sorted(pagerank, key=pagerank.get, reverse=True)[:200])
    predicted_edges_set = set(tuple(sorted(clean_edge(e))) for e in predicted_edges if clean_edge(e)[0] in top_nodes and clean_edge(e)[1] in top_nodes)
    actual_edges_set = set(tuple(sorted(clean_edge(e))) for e in actual_edges if clean_edge(e)[0] in top_nodes and clean_edge(e)[1] in top_nodes)
    G_sub = G_true.subgraph(top_nodes).copy()
    pos = nx.spring_layout(G_sub, seed=42, k=1.5)
    true_positives = predicted_edges_set & actual_edges_set
    false_positives = predicted_edges_set - actual_edges_set
    false_negatives = actual_edges_set - predicted_edges_set
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G_sub, pos, node_size=30, node_color='darkblue')
    nx.draw_networkx_edges(G_sub, pos, edgelist=true_positives, edge_color='green', width=1, label="True Positives")
    nx.draw_networkx_edges(G_sub, pos, edgelist=false_positives, edge_color='red', style='dashed', width=1, label="False Positives")
    nx.draw_networkx_edges(G_sub, pos, edgelist=false_negatives, edge_color='black', style='dotted', width=0.8, label="False Negatives")
    plt.legend()
    plt.title("Link Prediction Visualisation, Top 100")
    plt.axis('off')
    plt.savefig(save_path)


def clean_edge(edge):
    if isinstance(edge, str):
        return eval(edge)  # turn "('alice', 'bob')" into ('alice', 'bob')
    return edge

G_eval = subgraphs[12].to_undirected()
visualize_predictions(G_eval, actual_edges, predicted_edges)