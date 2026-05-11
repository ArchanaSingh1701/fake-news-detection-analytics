# Fake News Detection System

A comprehensive fake news detection system using graph neural networks (GNNs) and traditional machine learning approaches. The system analyzes both content features and social network propagation patterns to identify fake news articles.

## Features

- **Dual Approach**: Combines traditional ML (Random Forest) with Graph Neural Networks
- **Graph-based Analysis**: Models user-news interactions and social relationships
- **Feature Engineering**: TF-IDF content features + graph-based features (degree, betweenness centrality, clustering)
- **Flexible Data Sources**: Supports synthetic data generation and real FakeNewsNet dataset
- **Comprehensive Evaluation**: Detailed performance metrics and error analysis

## Project Structure

```
├── data_prep.py              # Synthetic data generation
├── modified_data_prep.py     # FakeNewsNet data preprocessing
├── graph_builder.py          # Heterogeneous graph construction
├── feature_extractor.py      # Feature extraction (TF-IDF + graph features)
├── train_ml.py              # Random Forest training
├── train_gnn.py             # Graph Neural Network training
├── evaluate.py              # Model evaluation and analysis
├── run_pipeline.py          # Complete pipeline orchestration
├── TextAnalyzer.java        # Java utility for text analysis
└── requirements.txt         # Python dependencies
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fake-news-detection
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional dependencies for NLP**
   ```bash
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
# Run with synthetic data (fastest)
python run_pipeline.py --data-source synthetic

# Run with FakeNewsNet data (requires dataset)
python run_pipeline.py --data-source fakenewsnet
```

### Option 2: Step-by-Step Execution

1. **Generate synthetic data**
   ```bash
   python data_prep.py --synthesize
   ```

2. **Build graph**
   ```bash
   python graph_builder.py
   ```

3. **Extract features**
   ```bash
   python feature_extractor.py
   ```

4. **Train models**
   ```bash
   python train_ml.py    # Random Forest
   python train_gnn.py   # Graph Neural Network
   ```

5. **Evaluate results**
   ```bash
   python evaluate.py
   ```

## Data Sources

### Synthetic Data (Default)
- Generates 200 users and 100 news articles
- Random user-news interactions and social relationships
- Balanced fake/real news distribution (40%/60%)
- Perfect for testing and development

### FakeNewsNet Dataset
- Real-world dataset with news articles and social media data
- Requires downloading the FakeNewsNet dataset
- Set `DATASET_ROOT` in `modified_data_prep.py` to your dataset path

## Model Architecture

### Traditional ML Pipeline
1. **Content Features**: TF-IDF vectorization (500 features)
2. **Graph Features**: Node degree, betweenness centrality, clustering coefficient
3. **Classifier**: Random Forest with class balancing

### Graph Neural Network Pipeline
1. **Node Features**: Degree + node type (user/news)
2. **Architecture**: 2-layer GCN with ReLU activation and dropout
3. **Training**: Cross-entropy loss with Adam optimizer

## Performance Metrics

The system evaluates models using:
- **Classification Report**: Precision, recall, F1-score for each class
- **ROC AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed breakdown of predictions
- **Feature Importance**: Top contributing features (Random Forest)

## Configuration

### Key Parameters
- `SYN_USERS`: Number of synthetic users (default: 200)
- `SYN_NEWS`: Number of synthetic news articles (default: 100)
- `DATA_DIR`: Data directory path (default: 'data')
- TF-IDF max features: 500
- GNN hidden dimensions: 64

### Customization
Modify parameters in individual scripts or use command-line arguments:

```bash
python data_prep.py --synthesize --n_users 500 --n_news 200
python run_pipeline.py --skip-gnn  # Skip GNN training
```

## Output Files

After running the pipeline, the following files are generated in the `data/` directory:

- `users.csv`: User profiles and metadata
- `news.csv`: News articles with labels
- `user_news_edges.csv`: User-news interaction edges
- `user_follows.csv`: User-user follow relationships
- `graph.gpickle`: NetworkX graph object
- `X.npy`, `y.npy`: Feature matrix and labels
- `rf_model.joblib`: Trained Random Forest model
- `gnn_model.pth`: Trained GNN model weights
- `nodes.csv`: Node information with predictions

## Troubleshooting

### Common Issues

1. **Missing dependencies**
   ```bash
   pip install torch torch-geometric --extra-index-url https://download.pytorch.org/whl/cpu
   ```

2. **Graph file not found**
   - Ensure `graph_builder.py` runs successfully before feature extraction

3. **Empty dataset**
   - Check data generation parameters
   - Verify CSV files are created in the data directory

4. **Memory issues with large graphs**
   - Reduce the number of users/news in synthetic data
   - Use sampling in betweenness centrality calculation

### Performance Tips

- Use synthetic data for development and testing
- Enable multiprocessing in Random Forest (`n_jobs=-1`)
- Sample large graphs for centrality calculations
- Use GPU for GNN training if available

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{fake-news-detection,
  title={Fake News Detection using Graph Neural Networks},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/fake-news-detection}
}
```

## Acknowledgments

- FakeNewsNet dataset creators
- PyTorch Geometric team
- NetworkX developers
- Scikit-learn contributors