from FlagEmbedding import BGEM3FlagModel
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils.visualization import visualize_embeddings

def main():
    # Initialize the BGE M3 embedding model
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    # Define example queries and potential documents
    query = "What are machine learning embeddings used for?"
    sentences = [
        query,
        "What is BGE M3?",
        "Definition of BM25",
        "How do embedding models work?",
        "Explanation of vector search",
        "Machine learning embeddings"
    ]

    # Encode sentences into embeddings
    embeddings = model.encode(sentences, batch_size=12, max_length=8192)['dense_vecs']
    
    # Save embeddings
    np.save('data/embeddings.npy', embeddings)
    
    # Visualize the embeddings
    visualize_embeddings(embeddings, sentences, query_idx=0)

if __name__ == "__main__":
    main()
