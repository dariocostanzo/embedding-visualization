from FlagEmbedding import BGEM3FlagModel  # Specialized library for generating embeddings
import numpy as np  # Numerical computing library for array operations
from sklearn.manifold import TSNE  # Dimensionality reduction technique for visualization
import matplotlib.pyplot as plt  # Plotting library to create visual representations

"""
==================================================================================
COMPREHENSIVE GUIDE: EMBEDDINGS, RAG SYSTEMS, AND LLMs
==================================================================================

What are Embeddings?
-------------------
- Embeddings are numerical vector representations of text (or other data types)
- They capture semantic meaning by positioning similar content closer in vector space
- Modern embedding dimensions typically range from 384 to 4096 numbers per text snippet
- They allow machines to understand relationships between concepts

What is BGE M3?
--------------
- BGE M3 is a powerful embedding model developed by BAAI (Beijing Academy of Artificial Intelligence)
- It's specifically designed for retrieval-based applications
- Can encode text into high-quality vector representations
- Supports multiple languages and long contexts (up to 8192 tokens)

LLMs and RAG Systems
-------------------
Large Language Models (LLMs):
- Models like GPT-4, Claude, Llama, etc. that generate human-like text
- They have internal knowledge from training but can't access external information directly
- Limited by knowledge cutoff dates and potential for hallucination

Retrieval Augmented Generation (RAG):
- A technique that enhances LLMs with up-to-date or domain-specific knowledge
- The RAG process works in five key steps:
  1. INDEXING: Convert documents into embeddings and store them
  2. QUERY ENCODING: Convert user's question into an embedding 
  3. RETRIEVAL: Find documents with embeddings most similar to the query embedding
  4. AUGMENTATION: Add retrieved context to the original prompt
  5. GENERATION: LLM generates a response using both the query and retrieved context

Why This Visualization Matters:
------------------------------
- Shows how different questions/documents relate to each other semantically
- Demonstrates how a RAG system would find relevant documents
- Illustrates the "nearest neighbour" concept central to vector search
- Helps debug and optimize retrieval quality by visualizing relationships

Use Cases for Embeddings & RAG:
------------------------------
1. Knowledge bases and documentation search
2. Customer support automation
3. Legal document analysis
4. Research assistance
5. Personal knowledge management
6. E-commerce product search
"""

# Initialize the BGE M3 embedding model
# use_fp16=True enables faster computation with minimal performance loss
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Define example queries and potential documents for a RAG system
# The first sentence is our primary query of interest
query = "What are machine learning embeddings used for?"

# These sentences represent potential documents in a RAG knowledge base
sentences_1 = [
    query,  # Our primary query about embeddings
    "What is BGE M3?",  # Information about an embedding model
    "Definition of BM25",  # Classic information retrieval algorithm document
    "How do embedding models work?",  # Technical explanation document
    "Explanation of vector search",  # Document about finding similar vectors
    "Machine learning embeddings"  # General overview document
]

# Encode sentences into dense vector representations (embeddings)
# In a real RAG system:
# - Documents would be split into chunks and encoded during indexing
# - The query would be encoded at retrieval time
embeddings_1 = model.encode(sentences_1, 
                            batch_size=12,  # Process in batches for efficiency
                            max_length=8192)['dense_vecs']  # Support long text

# Save embeddings to a file (in a real system, might use a vector database like Pinecone, Weaviate, etc.)
np.save('embeddings_1.npy', embeddings_1)

# Load the saved embeddings
loaded_embeddings_1 = np.load('embeddings_1.npy')

# Configure t-SNE for dimensionality reduction
# t-SNE helps visualize high-dimensional embeddings in 2D space
# In a real RAG system, this visualization step wouldn't be needed for operation
tsne = TSNE(n_components=2,  # Reduce to 2 dimensions for visualization
            random_state=42,  # Ensure reproducibility
            perplexity=min(len(sentences_1) - 1, 30))  # Parameter affecting local vs. global structure

# Transform high-dimensional embeddings to 2D space for visualization
embeddings_2d = tsne.fit_transform(loaded_embeddings_1)

# Create a visualization of the semantic relationships
plt.figure(figsize=(14, 10))  # Larger figure for better readability

# Plot document points in blue
plt.scatter(embeddings_2d[1:, 0], embeddings_2d[1:, 1], 
            color='blue', label='Documents', s=100, alpha=0.7)

# Plot query point in red to highlight
plt.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1], 
            color='red', label='Query', s=150, marker='*')

# Annotate each point with its corresponding text
for i, sentence in enumerate(sentences_1):
    plt.annotate(sentence, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                 xytext=(10, 10), textcoords='offset points', 
                 fontsize=10, fontweight='bold' if i == 0 else 'normal')

# Draw lines from query to all other points (illustrating distances)
for i in range(1, len(sentences_1)):
    plt.plot([embeddings_2d[0, 0], embeddings_2d[i, 0]], 
             [embeddings_2d[0, 1], embeddings_2d[i, 1]], 
             'k--', alpha=0.3)

# Add plot details
plt.title('Semantic Relationships in Vector Space (t-SNE Visualization)', fontsize=16)
plt.xlabel('Semantic Direction A', fontsize=14)  # More intuitive axis label
plt.ylabel('Semantic Direction B', fontsize=14)  # More intuitive axis label
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Print the 2D embeddings for detailed inspection
print("2D Embeddings (t-SNE projection):")
print(embeddings_2d)

"""
==================================================================================
INTERPRETING THE VISUALIZATION
==================================================================================

What You're Seeing:
------------------
- Each point represents a text snippet in semantic space
- The red star (*) is our query: "What are machine learning embeddings used for?"
- Blue dots are potential documents in our knowledge base
- Dotted lines show connections between the query and each document
- Closer points are more semantically similar

Understanding the Axes:
---------------------
- "Semantic Direction A" and "Semantic Direction B" are abstract dimensions
- They don't represent specific concepts but rather directions in semantic space
- t-SNE creates these dimensions to preserve similarity relationships
- The actual meaning of these directions is not defined

How This Relates to RAG:
----------------------
- In a RAG system, we would retrieve the documents closest to our query
- The closest documents (shortest dotted lines) would be added to the LLM prompt
- Distance approximates relevance - closer points are more likely to contain helpful information
- This visualization lets us see at a glance which documents are most relevant

Practical Applications:
---------------------
- Data exploration: Understand your document relationships
- Query analysis: See how different queries relate to your knowledge base
- System tuning: Identify potential retrieval issues
- Document organization: Detect clusters and gaps in coverage

Important Limitations:
--------------------
- t-SNE is a non-linear projection and distorts some distance relationships
- The original embeddings exist in much higher dimensions (typically 768-1536)
- Some semantic nuances are lost in the 2D visualization
- Real vector search uses distance metrics in the original high-dimensional space
"""