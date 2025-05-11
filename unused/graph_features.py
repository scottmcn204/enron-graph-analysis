import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pickle

with open("graph.pkl", "rb") as f:
    G = pickle.load(f)

features = pd.DataFrame(index=G.nodes())
features['degree'] = dict(G.degree( ))
features['weighted_degree'] = dict(G.degree(weight='weight'))
features['closeness'] = nx.closeness_centrality(G)
features['betweeness'] = nx.betweenness_centrality(G)
features['pagerank'] = nx.pagerank(G)

with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(features)

with open("graph_features.csv", "wb") as f:
    features.to_csv(f)