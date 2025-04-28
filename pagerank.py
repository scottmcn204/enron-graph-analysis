import networkx as nx
import matplotlib.pyplot as plt
import pickle

with open("graph.pkl", "rb") as f:
    G = pickle.load(f)

pagerank_scores = nx.pagerank(G, weight='weight')
sorted_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
for person, score in sorted_pagerank[:10]:
    print(f"{person}: {score:.5f}") 
top_nodes = [person for person, _ in sorted_pagerank[:10]]
subG = G.subgraph(top_nodes)
pos = nx.kamada_kawai_layout(subG)
node_size = [pagerank_scores[node]*50000 for node in subG.nodes()]
nx.draw_networkx_nodes(subG, pos, node_size=node_size, node_color='skyblue')
nx.draw_networkx_edges(subG, pos, arrowstyle='->', arrowsize=10)
nx.draw_networkx_labels(subG, pos, font_size=8)
plt.title("Top 10 Nodes by PageRank (Enron Email Network)")
plt.axis('off')
plt.savefig("pagerank_top10.png", dpi=300)  