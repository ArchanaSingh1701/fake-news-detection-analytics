import pandas as pd
from pathlib import Path

# 1. Setup paths (make sure these files are in your current folder or provide the full path)
FAKE_CSV_PATH = r"C:\Users\Archana\Desktop\Fake.csv"
TRUE_CSV_PATH = r"C:\Users\Archana\Desktop\True.csv"
OUTPUT_DIR = Path("data")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 2. Load the CSVs
    # Note: Check if your columns are named 'text' or 'content'. 
    # Usually, these datasets have 'title' and 'text'.
    try:
        df_fake = pd.read_csv(FAKE_CSV_PATH)
        df_true = pd.read_csv(TRUE_CSV_PATH)
    except FileNotFoundError:
        print("Error: Ensure fake.csv and true.csv are in the correct directory.")
        return

    # 3. Add labels and unique IDs
    df_fake['label'] = 'fake'
    df_true['label'] = 'real'
    
    # Combine them
    df_combined = pd.concat([df_fake, df_true], ignore_index=True)

    # Create a news_id (e.g., news_0, news_1...)
    df_combined['news_id'] = [f"news_{i}" for i in range(len(df_combined))]

    # 4. Prepare the final "news.csv"
    # We only keep what your original code wanted: news_id, label, and text
    if 'text' not in df_combined.columns:
        print(f"Warning: 'text' column not found. Available columns: {list(df_combined.columns)}")
        # If your column is named differently, rename it here:
        # df_combined = df_combined.rename(columns={'body': 'text'})

    news_df = df_combined[['news_id', 'label', 'text']]
    news_df.to_csv(OUTPUT_DIR / "news.csv", index=False)

    # 5. Handle "edges" (User-News relationships)
    # IMPORTANT: standard CSV datasets (like ISOT) do NOT have user/tweet info.
    # We create an empty file or demo data to keep your downstream code from breaking.
    edges = [] 
    # Example dummy edge if needed: 
    # edges.append({"user_id": "system_gen", "news_id": "news_0"})
    
    pd.DataFrame(edges, columns=["user_id", "news_id"]).to_csv(
        OUTPUT_DIR / "user_news_edges.csv", index=False
    )

    print("✅ Data preparation complete")
    print(f"News articles processed: {len(news_df)}")
    print(f"Edges created: {len(edges)} (CSVs usually don't contain social data)")

if __name__ == "__main__":
    main()