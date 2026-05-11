import pickle
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"

with open(f"{DATA_DIR}/graph.gpickle", "rb") as f:
    G = pickle.load(f)

news_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "news"]
texts = [G.nodes[n]["text"] for n in news_nodes]
labels = [1 if G.nodes[n]["label"] == "fake" else 0 for n in news_nodes]

tfidf = TfidfVectorizer(max_features=100)
X_text = tfidf.fit_transform(texts).toarray()

deg = dict(G.degree())
X_graph = np.array([[deg[n]] for n in news_nodes])

X = np.hstack([X_text, X_graph])
X = StandardScaler().fit_transform(X)

np.save(f"{DATA_DIR}/X.npy", X)
np.save(f"{DATA_DIR}/y.npy", np.array(labels))

print("✅ Features extracted")

