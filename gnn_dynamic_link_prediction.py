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

def train_on_graph_pair(G_t, G_t1, model, optimizer):
    print("month")
    data = nx_to_pyg(G_t)
    all_nodes = list(G_t.nodes())
    name_to_id = {name: idx for idx, name in enumerate(sorted(all_nodes))}

    existing_edges = set(G_t.edges())
    candidate_edges = [(u, v) for u in all_nodes for v in all_nodes if u < v and (u, v) not in existing_edges]
    positive_edges = [(u, v) for (u, v) in candidate_edges if G_t1.has_edge(u, v)]

    if len(positive_edges) == 0:
        return 0.0  # skip if no new edges

    negative_edges = random.sample([e for e in candidate_edges if e not in positive_edges], len(positive_edges))
    pos_edges = [(name_to_id[u], name_to_id[v]) for u, v in positive_edges]
    neg_edges = [(name_to_id[u], name_to_id[v]) for u, v in negative_edges]

    pos_edge_index = torch.tensor(pos_edges, dtype=torch.long).T
    neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).T

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

    return loss.item()

# Load your monthly graphs
with open("graph.pkl", "rb") as f:
    monthly_graphs = pickle.load(f)

# Initialize model
model = GraphSAGE_LinkPredictor(in_channels=1, hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train across time steps and epochs
for epoch in range(20):  # number of passes over the data
    total_loss = 0
    for t in range(len(monthly_graphs) - 1):
        G_t = monthly_graphs[t]
        G_t1 = monthly_graphs[t + 1]
        loss = train_on_graph_pair(G_t, G_t1, model, optimizer)
        total_loss += loss
    print(f"Epoch {epoch} | Total Loss: {total_loss:.4f}")

# Evaluate on last pair
def evaluate_on_last_pair(G_t, G_t1, model):
    data = nx_to_pyg(G_t)
    all_nodes = list(G_t.nodes())
    name_to_id = {name: idx for idx, name in enumerate(sorted(all_nodes))}

    existing_edges = set(G_t.edges())
    candidate_edges = [(u, v) for u in all_nodes for v in all_nodes if u < v and (u, v) not in existing_edges]
    positive_edges = [(u, v) for (u, v) in candidate_edges if G_t1.has_edge(u, v)]
    negative_edges = random.sample([e for e in candidate_edges if e not in positive_edges], len(positive_edges))

    pos_edges = [(name_to_id[u], name_to_id[v]) for u, v in positive_edges]
    neg_edges = [(name_to_id[u], name_to_id[v]) for u, v in negative_edges]

    pos_edge_index = torch.tensor(pos_edges, dtype=torch.long).T
    neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).T
    edge_pairs = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    labels = torch.cat([
        torch.ones(pos_edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ])

    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index, edge_pairs)
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        auc = roc_auc_score(labels_np, preds_np)
        f1 = f1_score(labels_np, preds_np > 0.5)
        print(f"\nFinal Eval — AUC: {auc:.4f} | F1: {f1:.4f}")

# Evaluate on final timestep
evaluate_on_last_pair(monthly_graphs[-2], monthly_graphs[-1], model)
