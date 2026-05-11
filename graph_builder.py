import networkx as nx
import pandas as pd
import pickle
from pathlib import Path

DATA_DIR = Path("data")

def main():
    news = pd.read_csv(DATA_DIR / "news.csv")
    edges = pd.read_csv(DATA_DIR / "user_news_edges.csv")

    G = nx.Graph()

    for _, r in news.iterrows():
        G.add_node(
            r["news_id"],
            type="news",
            label=r["label"],
            text=r["text"]
        )

    for _, r in edges.iterrows():
        u, n = r["user_id"], r["news_id"]
        if u not in G:
            G.add_node(u, type="user")
        if n in G:
            G.add_edge(u, n)

    with open(DATA_DIR / "graph.gpickle", "wb") as f:
        pickle.dump(G, f)

    print("✅ Graph built")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

if __name__ == "__main__":
    main()
