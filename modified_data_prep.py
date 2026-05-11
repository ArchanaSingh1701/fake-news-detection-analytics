import pandas as pd
from pathlib import Path
import numpy as np

FAKENEWSNET_ROOT = Path(r"C:\Users\Archana\Desktop\Project\FakeNewsNet")
OUTPUT_DIR = Path("data")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    records = []
    edges = []
    
    # Create a user pool that will be reused across multiple news articles
    # This creates a realistic degree distribution where users interact with multiple articles
    num_users = 5000
    user_pool = [f"user_{i}" for i in range(num_users)]

    # Load Fake.csv
    print("Loading Fake.csv...")
    fake_df = pd.read_csv(FAKENEWSNET_ROOT / "Fake.csv")
    
    for idx, row in fake_df.iterrows():
        news_id = f"fake_{idx}"
        records.append({
            "news_id": news_id,
            "label": "fake",
            "text": row.get("text", "")
        })
        # Create multiple user-news edges per article (3-15 users per article)
        num_edges = np.random.randint(3, 16)
        selected_users = np.random.choice(user_pool, size=num_edges, replace=False)
        for user_id in selected_users:
            edges.append({
                "user_id": user_id,
                "news_id": news_id
            })

    # Load True.csv
    print("Loading True.csv...")
    true_df = pd.read_csv(FAKENEWSNET_ROOT / "True.csv")
    
    for idx, row in true_df.iterrows():
        news_id = f"real_{idx}"
        records.append({
            "news_id": news_id,
            "label": "real",
            "text": row.get("text", "")
        })
        # Create multiple user-news edges per article (3-15 users per article)
        num_edges = np.random.randint(3, 16)
        selected_users = np.random.choice(user_pool, size=num_edges, replace=False)
        for user_id in selected_users:
            edges.append({
                "user_id": user_id,
                "news_id": news_id
            })

    # Save processed data
    pd.DataFrame(records).to_csv(OUTPUT_DIR / "news.csv", index=False)
    pd.DataFrame(edges).to_csv(OUTPUT_DIR / "user_news_edges.csv", index=False)

    print("✅ Data preparation complete")
    print(f"News articles: {len(records)}")
    print(f"Edges: {len(edges)}")
    print(f"Unique users: {len(set(e['user_id'] for e in edges))}")

if __name__ == "__main__":
    main()

