import pandas as pd

edges = pd.read_csv(r'C:\Users\Archana\Desktop\Project\data\user_news_edges.csv')
print(f"Total edges: {len(edges)}")
print(f"Unique users: {edges['user_id'].nunique()}")
print(f"Unique news: {edges['news_id'].nunique()}")
print(f"\nUser degree stats:")
user_degrees = edges.groupby('user_id').size()
print(f"  Min: {user_degrees.min()}, Max: {user_degrees.max()}, Mean: {user_degrees.mean():.2f}")
print(f"\nNews degree stats:")
news_degrees = edges.groupby('news_id').size()
print(f"  Min: {news_degrees.min()}, Max: {news_degrees.max()}, Mean: {news_degrees.mean():.2f}")
