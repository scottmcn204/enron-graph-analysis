import networkx as nx
import pandas as pd

G = nx.read_edgelist("enronedgelist.txt", create_using=nx.DiGraph())

features =[]
for node in G.nodes():
    features.append({
        'node': node,
        'degree': G.degree(node),
        'in_degree': G.in_degree(node),
        'out_degree' : G.out_degree(node),
        'page_rank' : nx.pagerank(G)[node],


    })
featuresdf = pd.DataFrame(features)
print(featuresdf)