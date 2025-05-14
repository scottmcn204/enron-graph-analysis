import matplotlib.pyplot as plt
import networkx as nx
import pickle

with open("subgraphs.pkl", "rb") as f:
    subgraphs = pickle.load(f)

with open("predicted_edges.pkl", "rb") as f:
    predicted_edges = pickle.load(f)


def visualize_subgraphs():
    for i, G in enumerate(subgraphs):
        plt.figure(figsize=(20,20))
        pos = nx.spring_layout(G, k=.1)
        nx.draw_networkx(G, pos, node_size=25, node_color='red', with_labels=True, edge_color='blue')
        labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        plt.savefig("plots/subGmonth" + str(i) + ".png")

def visualize_predictions(G_true, predicted_edges, threshold=0.5, score_dict=None):
    pos = nx.spring_layout(G_true, seed=42)

    # Draw base graph (existing edges)
    nx.draw_networkx_nodes(G_true, pos, node_size=50, node_color='lightblue')
    nx.draw_networkx_edges(G_true, pos, edge_color='gray', alpha=0.3)

    # Predicted edges (colored)
    if score_dict:
        high_confidence = [e for e, score in score_dict.items() if score > threshold]
    else:
        high_confidence = predicted_edges

    nx.draw_networkx_edges(G_true, pos, edgelist=high_confidence,
                           edge_color='green', style='dashed', width=2, label='Predicted Edges')

    plt.title("Graph with Predicted Edges")
    plt.legend()
    plt.axis('off')
    plt.savefig("plots/predictions.png")


visualize_predictions(subgraphs[0], predicted_edges)