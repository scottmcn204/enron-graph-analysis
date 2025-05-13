import pandas as pd
import pickle
import networkx as nx
from networkx.algorithms.link_prediction import adamic_adar_index, jaccard_coefficient, preferential_attachment
from itertools import product
import random

def directed_to_undirected_with_weights(G_directed):
    G_undirected = nx.Graph()
    for u, v, data in G_directed.edges(data=True):
        weight = data.get("weight", 1)
        if G_undirected.has_edge(u, v):
            G_undirected[u][v]["weight"] += weight
        else:
            G_undirected.add_edge(u, v, weight=weight)
    return G_undirected

with open("subgraphs.pkl", "rb") as f:
    monthly_graphs = pickle.load(f)

for i in range(len(monthly_graphs) - 1):
    Gi = directed_to_undirected_with_weights(monthly_graphs[i])
    Gi1 = directed_to_undirected_with_weights(monthly_graphs[i+1])
    all_nodes = set(Gi.nodes())
    candidate_edges = [(u, v) for u, v in product(all_nodes, all_nodes) if u != v and not Gi.has_edge(u, v)]

    positive_edges = [(u, v) for u, v in candidate_edges if Gi1.has_edge(u, v)]
    negative_edges = [(u, v) for u, v in candidate_edges if not Gi1.has_edge(u, v)]

    sampled_neg = random.sample(negative_edges, min(len(negative_edges), len(positive_edges) * 2))
    balanced_edges = positive_edges + sampled_neg
    labels = [1] * len(positive_edges) + [0] * len(sampled_neg)

    aa = dict(((u,v), p) for u, v, p in adamic_adar_index(Gi, balanced_edges))
    jc = dict(((u,v), p) for u, v, p in jaccard_coefficient(Gi, balanced_edges))
    pa = dict(((u,v), p) for u, v, p in preferential_attachment(Gi, balanced_edges))

    features = [] 
    for edge in balanced_edges:
        row = [
            edge,
            aa.get(edge, 0),
            jc.get(edge, 0),
            pa.get(edge, 0),
        ]
        features.append(row)

    df = pd.DataFrame(features, columns=["edge", "adamic_adar", "jaccard", "preferential_attachment"])
    df["label"] = labels
    print(df)
    df.to_csv(("features" + str(i) + ".csv"), index=False)

