import pickle
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

DATA_DIR = "data"

def main():
    # Load graph
    with open(f"{DATA_DIR}/graph.gpickle", "rb") as f:
        G = pickle.load(f)

    # Degree of each node
    degrees = [deg for _, deg in G.degree()]

    # Count frequency of degrees
    degree_count = Counter(degrees)

    # Sort for linear plot
    x = sorted(degree_count.keys())
    y = [degree_count[d] for d in x]

    # Plot LINE GRAPH
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker='o')
    plt.xlabel("Degree of Node")
    plt.ylabel("Number of Nodes")
    plt.title("Linear Graph: Degree Distribution of Fake News Network")
    plt.grid(True)
    plt.show()

    print("✅ Linear graph displayed successfully")

if __name__ == "__main__":
    main()
