import os
import sys

steps = [
   # "m.py",

    
    "modified_data_prep.py",
    "graph_builder.py",
    "feature_extractor.py",
    "train_ml.py",
   # "visualize_graph.py"
    "linear_graph_analysis.py"
]


for s in steps:
    print(f"\n▶ Running {s}")
    os.system(f"{sys.executable} {s}")
