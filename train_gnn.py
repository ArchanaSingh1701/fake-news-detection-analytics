import pickle
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

DATA_DIR = "data"

with open(f"{DATA_DIR}/graph.gpickle", "rb") as f:
    G = pickle.load(f)

nodes = list(G.nodes())
idx = {n: i for i, n in enumerate(nodes)}

edges = []
for u, v in G.edges():
    edges.append([idx[u], idx[v]])
    edges.append([idx[v], idx[u]])

edge_index = torch.tensor(edges).t()

x = []
y = []

for n in nodes:
    deg = G.degree(n)
    is_news = 1 if G.nodes[n]["type"] == "news" else 0
    x.append([deg, is_news])
    if is_news:
        y.append(1 if G.nodes[n]["label"] == "fake" else 0)
    else:
        y.append(-1)

data = Data(
    x=torch.tensor(x, dtype=torch.float),
    edge_index=edge_index,
    y=torch.tensor(y)
)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = GCNConv(2, 16)
        self.c2 = GCNConv(16, 2)

    def forward(self, x, e):
        x = F.relu(self.c1(x, e))
        return self.c2(x, e)

model = GCN()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

for _ in range(100):
    opt.zero_grad()
    out = model(data.x, data.edge_index)
    mask = data.y >= 0
    loss = F.cross_entropy(out[mask], data.y[mask])
    loss.backward()
    opt.step()

torch.save(model.state_dict(), f"{DATA_DIR}/gnn_model.pth")
print("✅ GNN trained")
