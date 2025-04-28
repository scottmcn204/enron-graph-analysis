import networkx as nx
import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt

emails = pd.read_csv("emails.csv")
G = nx.DiGraph()
edge_weights = defaultdict(int)

def extract_sender(message):
    match = re.search(r'From:\s(.+)', str(message))
    if match:
        senders = match.group(1)
        return [r.strip() for r in senders.split(',')]
    else:
        return []

def extract_recipients(message):
    match = re.search(r'To:\s(.+)', str(message))
    if match:
        recipients = match.group(1)
        return [r.strip() for r in recipients.split(',')]
    else:
        return []

emails['sender'] = emails['message'].apply(extract_sender)
emails['recipients'] = emails['message'].apply(extract_recipients)

print(emails)
for index, row in emails.iterrows():
    sender = row['sender'][0]
    recipients = row['recipients']
    for recipient in recipients:
        if sender and recipient:
            edge_weights[(sender,recipient)] += 1
for (sender, recipient), weigth in edge_weights.items():
    G.add_edge(sender, recipient, weigth=weigth)
            

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

pagerank_scores = nx.pagerank(G, weight='weight')
sorted_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
for person, score in sorted_pagerank[:10]:
    print(f"{person}: {score:.5f}") 
top_nodes = [person for person, _ in sorted_pagerank[:20]]
subG = G.subgraph(top_nodes)
pos = nx.spring_layout(subG, seed=42)
node_size = [pagerank_scores[node]*50000 for node in subG.nodes()]
nx.draw_networkx_nodes(subG, pos, node_size=node_size, node_color='skyblue')
nx.draw_networkx_edges(subG, pos, arrowstyle='->', arrowsize=10)
nx.draw_networkx_labels(subG, pos, font_size=8)
plt.title("Top 20 Nodes by PageRank (Enron Email Network)")
plt.axis('off')
plt.savefig("pagerank_top20.png", dpi=300)