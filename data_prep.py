# data_prep.py

import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


DATA_DIR = 'data'

SYN_USERS = 200
SYN_NEWS = 100


np.random.seed(42)


def synthesize(output_dir=DATA_DIR, n_users=SYN_USERS, n_news=SYN_NEWS):
    """Generate synthetic dataset for fake news detection"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate users
    users = []
    for i in range(n_users):
        users.append({
            'user_id': f'user_{i}', 
            'followers': int(np.random.exponential(50)), 
            'is_bot_prob': np.random.rand()
        })
    users_df = pd.DataFrame(users)
    users_df.to_csv(os.path.join(output_dir, 'users.csv'), index=False)

    # Generate news articles
    news = []
    labels = ['real', 'fake']
    for j in range(n_news):
        label = np.random.choice(labels, p=[0.6, 0.4])
        text = f"This is a sample news article {j} about topic {np.random.randint(0, 20)}."
        news.append({
            'news_id': f'news_{j}', 
            'title': f'Title {j}', 
            'text': text, 
            'label': label
        })
    news_df = pd.DataFrame(news)
    news_df.to_csv(os.path.join(output_dir, 'news.csv'), index=False)

    # Generate user-news interactions
    edges = []
    for u in users_df['user_id']:
        # Each user shares 1-7 news articles
        k = np.random.randint(1, 8)
        chosen = np.random.choice(news_df['news_id'], size=k, replace=False)
        for n in chosen:
            edges.append({'user_id': u, 'news_id': n, 'relation': 'shared'})
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(os.path.join(output_dir, 'user_news_edges.csv'), index=False)

    # Generate user-user follow relationships
    follows = []
    for u in users_df['user_id']:
        k = np.random.poisson(2)
        if k <= 0:
            continue
        available_users = users_df[users_df['user_id'] != u]['user_id'].tolist()
        if not available_users:
            continue
        targets = np.random.choice(available_users, size=min(k, len(available_users)), replace=False)
        for t in targets:
            follows.append({'follower': u, 'followee': t})
    follows_df = pd.DataFrame(follows)
    follows_df.to_csv(os.path.join(output_dir, 'user_follows.csv'), index=False)

    print(f'Synthesized data in {output_dir}')
    print(f'Generated {len(users)} users, {len(news)} news articles, {len(edges)} interactions, {len(follows)} follows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic fake news dataset')
    parser.add_argument('--synthesize', action='store_true', help='Generate synthetic dataset')
    parser.add_argument('--output_dir', default=DATA_DIR, help='Output directory for generated data')
    parser.add_argument('--n_users', type=int, default=SYN_USERS, help='Number of users to generate')
    parser.add_argument('--n_news', type=int, default=SYN_NEWS, help='Number of news articles to generate')
    
    args = parser.parse_args()
    
    if args.synthesize:
        synthesize(args.output_dir, args.n_users, args.n_news)
    else:
        print('Run with --synthesize to create a toy dataset')