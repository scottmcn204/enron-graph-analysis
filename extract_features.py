import pandas as pd
import pickle
import networkx as nx
from networkx.algorithms.link_prediction import adamic_adar_index, jaccard_coefficient, preferential_attachment, common_neighbor_centrality
from itertools import product

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
    candidate_edges = list(product(all_nodes, all_nodes))
    candidate_edges = [(u, v) for u, v in product(all_nodes, all_nodes) if u != v and not Gi.has_edge(u, v)]
    aa = dict(((u,v), p) for u, v, p in adamic_adar_index(Gi, candidate_edges))
    jc = dict(((u,v), p) for u, v, p in jaccard_coefficient(Gi, candidate_edges))
    pa = dict(((u,v), p) for u, v, p in preferential_attachment(Gi, candidate_edges))
    # cn = {(u, v): len(list(common_neighbor_centrality(Gi, u, v)))
        #   for u, v in candidate_edges if u in Gi and v in Gi}
    labels = []
    for u, v in candidate_edges:
        label = 1 if Gi1.has_edge(u, v) else 0
        labels.append(label)
    features = [] 
    for edge in candidate_edges:
        row = [
            edge,
            aa.get(edge, 0),
            jc.get(edge, 0),
            pa.get(edge, 0),
            # cn.get(edge, 0),
        ]
        features.append(row)

    df = pd.DataFrame(features, columns=["edge", "adamic_adar", "jaccard", "preferential_attachment"])
    df["label"] = labels
    print(df)
    df.to_csv(("features" + str(i) + ".csv"), index=False)

