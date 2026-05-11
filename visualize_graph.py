import pickle
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch 

with open("data/graph.gpickle", "rb") as f:
    G = pickle.load(f)

pos = nx.spring_layout(G, seed=42)

colors = []
for n, d in G.nodes(data=True):
    if d["type"] == "user":
        colors.append("skyblue")
    elif d["label"] == "fake":
        colors.append("red")
    else:
        colors.append("green")

plt.figure(figsize=(8, 6))
nx.draw(G, pos, node_color=colors, node_size=200, with_labels=False)
plt.title("Fake News Graph")
plt.show()


with open("data/graph.gpickle", "rb") as f:
    G = pickle.load(f)

pos = nx.spring_layout(G, seed=42)

colors = []
for n, d in G.nodes(data=True):
    if d["type"] == "user":
        colors.append("skyblue")
    elif d["label"] == "fake":
        colors.append("red")
    else:
        colors.append("green")

plt.figure(figsize=(8, 6))
nx.draw(G, pos, node_color=colors, node_size=200, with_labels=False)
plt.title("Fake News Graph")


legend_elements = [
    Patch(facecolor='red', edgecolor='black', label='Fake News'),
    Patch(facecolor='green', edgecolor='black', label='Real News')
]

plt.legend(
    handles=legend_elements,
    loc='upper left',
    fontsize=9,
    frameon=True
)

plt.show()


