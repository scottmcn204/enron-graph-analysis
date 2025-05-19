# import glob
# import pandas as pd

# features = sorted(glob.glob("features/features*.csv"))[:17]
# feature_frames = [pd.read_csv(f) for f in features]

# window_size = 3
# lookahead = 1 
# data_sequences = []
# X = []
# y = []
# edges = []
# target_frame = feature_frames[-1]
# feature_frames = feature_frames[:-1]

# # Loop through edges in the target month
# for idx, row in target_frame.iterrows():
#     u, v = row['u'], row['v']
#     label = row['label']
#     edge_features = []

#     for frame in feature_frames:
#         match = frame[(frame['u'] == u) & (frame['v'] == v)]
#         if not match.empty:
#             edge_features.extend([
#                 match['adamic_adar'].values[0],
#                 match['jaccard'].values[0],
#                 match['common_neighbors'].values[0]
#             ])
#         else:
#             # Edge not present in this month
#             edge_features.extend(["err", "err", "err"])

#     X.append(edge_features)
#     y.append(label)
#     edges.append((u, v))  # Keep track of the edge

# # Convert to DataFrame if needed
# X_df = pd.DataFrame(X)
# y_series = pd.Series(y)

# print(X_df)

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

with open("graph.pkl", "rb") as f:
    monthly_graphs = pickle.load(f)

history = 3

for i in range(14, len(monthly_graphs) - 1):
    print("month" + str(i))
    Gi = directed_to_undirected_with_weights(monthly_graphs[i])
    Gi1 = directed_to_undirected_with_weights(monthly_graphs[i+1])
    history_graphs = [directed_to_undirected_with_weights(monthly_graphs[j]) 
                  for j in range(i - history + 1, i + 1) if j >= 0]
    all_nodes = set(Gi.nodes())
    candidate_edges = [(u, v) for u, v in product(all_nodes, all_nodes) if u != v and not Gi.has_edge(u, v)]

    positive_edges = [(u, v) for u, v in candidate_edges if Gi1.has_edge(u, v)]
    negative_edges = [(u, v) for u, v in candidate_edges if not Gi1.has_edge(u, v)]

    sampled_neg = random.sample(negative_edges, min(len(negative_edges), len(positive_edges)))
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
        # deg_u_series = [G.degree(u) for G in history_graphs]
        # deg_v_series = [G.degree(v) for G in history_graphs]
        # deg_u_delta = int(deg_u_series[-1]) - int(deg_u_series[0]) if len(deg_u_series) > 1 else 0
        # deg_v_delta = deg_v_series[-1] - deg_v_series[0] if len(deg_v_series) > 1 else 0
        cc_u_series = [nx.clustering(G, u) if u in G else 0 for G in history_graphs]
        cc_v_series = [nx.clustering(G, v) if v in G else 0 for G in history_graphs]
        cc_u_delta = cc_u_series[-1] - cc_u_series[0] if len(cc_u_series) > 1 else 0
        cc_v_delta = cc_v_series[-1] - cc_v_series[0] if len(cc_v_series) > 1 else 0
        link_presence = [1 if G.has_edge(u, v) else 0 for G in history_graphs]
        edge_persistence = sum(link_presence)
        row = {
            "edge" : (u,v),
            "adamic_adar": aa.get((u, v), 0),
            "jaccard": jc.get((u, v), 0),
            "preferential_attachment": pa.get((u, v), 0),
            "resource_allocation": ra.get((u, v), 0),
            "common_neighbors": cn_count,
            "shortest_path": spl,
            "edge_persistence": edge_persistence,
            "cc_u_delta": cc_u_delta,
            "cc_v_delta": cc_v_delta
        }
        features.append(row)

    df = pd.DataFrame(features)
    df["label"] = labels
    df.to_csv(("features/features" + str(i) + ".csv"), index=False)

