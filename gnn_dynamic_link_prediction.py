import torch
import torch.nn.functional as F
import torch.nn as nn
import networkx as nx
import random
import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx
from sklearn.metrics import roc_auc_score, f1_score
import pickle

# ===== Simplified GNN model =====
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
        return torch.sigmoid(self.link_pred(torch.cat([h_src, h_dst], dim=1))).squeeze()

# ===== Graph conversion =====
def nx_to_pyg(G):
    G = G.to_undirected()
    feat = torch.tensor([[G.degree(n)] for n in G.nodes()], dtype=torch.float)
    data = from_networkx(G)
    data.x = feat
    return data

# ===== Faster training function =====
def train_on_graph_pair(G_t, G_t1, model, optimizer, num_samples=500):
    data = nx_to_pyg(G_t)
    all_nodes = list(G_t.nodes())
    name_to_id = {n: i for i, n in enumerate(all_nodes)}
    G_t_edges = set(G_t.edges())
    G_t1_edges = set(G_t1.edges())

    # Sample a subset of candidate pairs
    candidates = [(u, v) for u in all_nodes for v in all_nodes if u < v and (u, v) not in G_t_edges]
    random.shuffle(candidates)
    candidates = candidates[:num_samples]

    positives = [(u, v) for (u, v) in candidates if (u, v) in G_t1_edges or (v, u) in G_t1_edges]
    negatives = [(u, v) for (u, v) in candidates if (u, v) not in G_t1_edges and (v, u) not in G_t1_edges]
    negatives = negatives[:len(positives)]  # balance

    if not positives:
        return 0.0

    pos_idx = torch.tensor([[name_to_id[u], name_to_id[v]] for u, v in positives], dtype=torch.long).T
    neg_idx = torch.tensor([[name_to_id[u], name_to_id[v]] for u, v in negatives], dtype=torch.long).T
    edge_pairs = torch.cat([pos_idx, neg_idx], dim=1)
    labels = torch.cat([torch.ones(pos_idx.shape[1]), torch.zeros(neg_idx.shape[1])])

    model.train()
    optimizer.zero_grad()
    preds = model(data.x, data.edge_index, edge_pairs)
    loss = F.binary_cross_entropy(preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

# ===== Evaluation function =====
def evaluate_on_pair(G_t, G_t1, model):
    print("evalutaion")
    data = nx_to_pyg(G_t)
    all_nodes = list(G_t.nodes())
    name_to_id = {n: i for i, n in enumerate(all_nodes)}
    G_t_edges = set(G_t.edges())
    G_t1_edges = set(G_t1.edges())

    candidates = [(u, v) for u in all_nodes for v in all_nodes if u < v and (u, v) not in G_t_edges]
    candidates = random.sample(candidates, 1000)

    pos = [(u, v) for (u, v) in candidates if (u, v) in G_t1_edges or (v, u) in G_t1_edges]
    neg = [(u, v) for (u, v) in candidates if (u, v) not in G_t1_edges and (v, u) not in G_t1_edges]
    neg = neg[:len(pos)]
    if not pos:
        return

    pos_idx = torch.tensor([[name_to_id[u], name_to_id[v]] for u, v in pos], dtype=torch.long).T
    neg_idx = torch.tensor([[name_to_id[u], name_to_id[v]] for u, v in neg], dtype=torch.long).T
    edge_pairs = torch.cat([pos_idx, neg_idx], dim=1)
    labels = torch.cat([torch.ones(pos_idx.shape[1]), torch.zeros(neg_idx.shape[1])])

    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index, edge_pairs)
    auc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, preds > 0.5)
    print(f"\n🔍 Eval — AUC: {auc:.3f} | F1: {f1:.3f}")

# ===== Main training loop =====
with open("graph.pkl", "rb") as f:
    monthly_graphs = pickle.load(f)

model = GraphSAGE_LinkPredictor(in_channels=1, hidden_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(5):  # fewer epochs for demo
    total_loss = 0
    for t in range(len(monthly_graphs) - 2):  # only use a few time steps
        print("month")
        G_t = monthly_graphs[t]
        G_t1 = monthly_graphs[t + 1]
        loss = train_on_graph_pair(G_t, G_t1, model, optimizer)
        total_loss += loss
    print(f"Epoch {epoch+1} | Total Loss: {total_loss:.4f}")

# Final evaluation  
evaluate_on_pair(monthly_graphs[-2], monthly_graphs[-1], model)
