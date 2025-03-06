import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, sentences, query_idx=0):
    """
    Visualize the relationships between embeddings using t-SNE.
    
    Args:
        embeddings: Array of embedding vectors
        sentences: List of original text sentences
        query_idx: Index of the query in the sentences list
    """
    # Configure t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(len(sentences) - 1, 30))
    
    # Transform embeddings to 2D
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Plot document points
    plt.scatter(embeddings_2d[1:, 0], embeddings_2d[1:, 1], 
                color='blue', label='Documents', s=100, alpha=0.7)
    
    # Plot query point
    plt.scatter(embeddings_2d[query_idx, 0], embeddings_2d[query_idx, 1], 
                color='red', label='Query', s=150, marker='*')
    
    # Annotate points
    for i, sentence in enumerate(sentences):
        plt.annotate(sentence, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                     xytext=(10, 10), textcoords='offset points', 
                     fontsize=10, fontweight='bold' if i == query_idx else 'normal')
    
    # Draw lines from query to other points
    for i in range(len(sentences)):
        if i != query_idx:
            plt.plot([embeddings_2d[query_idx, 0], embeddings_2d[i, 0]], 
                     [embeddings_2d[query_idx, 1], embeddings_2d[i, 1]], 
                     'k--', alpha=0.3)
    
    # Add plot details
    plt.title('Semantic Relationships in Vector Space (t-SNE Visualization)', fontsize=16)
    plt.xlabel('Semantic Direction A', fontsize=14)
    plt.ylabel('Semantic Direction B', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('data/embedding_visualization.png')
    plt.show()
