import pandas as pd
import pickle
import networkx as nx
from networkx.algorithms.link_prediction import adamic_adar_index, jaccard_coefficient, preferential_attachment, resource_allocation_index
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
    ra = dict(((u, v), p) for u, v, p in resource_allocation_index(Gi, candidate_edges))
    features = [] 
    for u, v in balanced_edges:
        try:
            cn_count = len(list(nx.common_neighbors(Gi, u, v)))
        except:
            cn_count = 0
        try:
            spl = nx.shortest_path_length(Gi, u, v)
        except nx.NetworkXNoPath:
            spl = -1
        deg_u = Gi.degree(u)
        deg_v = Gi.degree(v)
        wdeg_u = Gi.degree(u, weight="weight")
        wdeg_v = Gi.degree(v, weight="weight")
        cc_u = nx.clustering(Gi, u)
        cc_v = nx.clustering(Gi, v)
        row = {
            "edge": (u, v),
            "adamic_adar": aa.get((u, v), 0),
            "jaccard": jc.get((u, v), 0),
            "preferential_attachment": pa.get((u, v), 0),
            "resource_allocation": ra.get((u, v), 0),
            "common_neighbors": cn_count,
            "shortest_path": spl,
            "deg_u": deg_u,
            "deg_v": deg_v,
            "wdeg_u": wdeg_u,
            "wdeg_v": wdeg_v,
            "clustering_u": cc_u,
            "clustering_v": cc_v,
        }
        features.append(row)

    df = pd.DataFrame(features)
    df["label"] = labels
    print(df)
    df.to_csv(("features" + str(i) + ".csv"), index=False)

