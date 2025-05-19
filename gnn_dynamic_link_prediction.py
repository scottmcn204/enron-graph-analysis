import torch
import torch.nn.functional as F
import torch.nn as nn
import networkx as nx
import random
import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx, negative_sampling
from sklearn.metrics import roc_auc_score, f1_score
import pickle
import pandas as pd
import ast

def nx_to_pyg(G):
    G = G.to_undirected()
    for node in G.nodes():
        G.nodes[node]['feat'] = [G.degree(node)]
    data = from_networkx(G)
    data.x = data.feat.float()
    return data

class GraphSAGE_LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.link_pred = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_pairs):
        h = F.relu(self.sage1(x, edge_index))
        h = self.sage2(h, edge_index)
        h_src = h[edge_pairs[0]]
        h_dst = h[edge_pairs[1]]
        h_cat = torch.cat([h_src, h_dst], dim=1)
        return torch.sigmoid(self.link_pred(h_cat)).squeeze()

with open("graph.pkl", "rb") as f:
    monthly_graphs = pickle.load(f)

G_t = monthly_graphs[0]
G_t1 = monthly_graphs[1]

data = nx_to_pyg(G_t)

# Positive edges = new links in t+1 not in t
existing_edges = set(G_t.edges())
all_nodes = list(G_t.nodes())
candidate_edges = [(u, v) for u in all_nodes for v in all_nodes if u < v and (u, v) not in existing_edges]
positive_edges = [(u, v) for (u, v) in candidate_edges if G_t1.has_edge(u, v)]
negative_edges = random.sample([e for e in candidate_edges if e not in positive_edges], len(positive_edges))

def clean_edge(edge):
    if isinstance(edge, str):
        return eval(edge)  # careful: only use eval if you trust the source
    return edge
pos_edges = [clean_edge(e) for e in positive_edges]
neg_edges = [clean_edge(e) for e in negative_edges]

pos_edge_index = torch.tensor(pos_edges, dtype=torch.long).T
neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).T


model = GraphSAGE_LinkPredictor(in_channels=data.x.shape[1], hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    edge_pairs = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    labels = torch.cat([
        torch.ones(pos_edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ])

    preds = model(data.x, data.edge_index, edge_pairs)
    loss = F.binary_cross_entropy(preds, labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# === 5. Evaluate ===

model.eval()
with torch.no_grad():
    preds = model(data.x, data.edge_index, edge_pairs)
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    auc = roc_auc_score(labels_np, preds_np)
    f1 = f1_score(labels_np, preds_np > 0.5)
    print(f"\nAUC: {auc:.4f} | F1: {f1:.4f}")
