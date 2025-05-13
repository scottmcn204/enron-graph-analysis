import pandas as pd
import pickle
import networkx as nx
from networkx.algorithms.link_prediction import adamic_adar_index, jaccard_coefficient, preferential_attachment
from itertools import product


with open("graph.pkl", "rb") as f:
    monthly_graphs = pickle.load(f)

for i in range(len(monthly_graphs) - 1):
    Gi = monthly_graphs[i]
    Gi1 = monthly_graphs[i+1]
    all_nodes = set(Gi.nodes())
    candidate_edges = list(product(all_nodes, all_nodes))
    candidate_edges = [(u, v) for u, v in product(all_nodes, all_nodes) if u != v and not Gi.has_edge(u, v)]
    # aa = dict(((u,v), p) for u, v, p in adamic_adar_index(Gi.to_undirected(), candidate_edges))
    jc = dict(((u,v), p) for u, v, p in jaccard_coefficient(Gi.to_undirected(), candidate_edges))
    pa = dict(((u,v), p) for u, v, p in preferential_attachment(Gi.to_undirected(), candidate_edges))
    labels = []
    for u, v in candidate_edges:
        label = 1 if Gi1.has_edge(u, v) else 0
        labels.append(label)
    features = []
    for edge in candidate_edges:
        row = [
            # aa.get(edge, 0),
            jc.get(edge, 0),
            pa.get(edge, 0),
        ]
        features.append(row)

    df = pd.DataFrame(features, columns=["jaccard", "preferential_attachment"])
    df["label"] = labels
    print(df)

