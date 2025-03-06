# Embedding Visualization

A tool for visualizing machine learning embeddings and demonstrating RAG (Retrieval Augmented Generation) principles.

## Overview

This project demonstrates:
- How to generate embeddings using the BGE M3 model
- Visualization of semantic relationships between text using t-SNE
- Core concepts behind Retrieval Augmented Generation (RAG) systems

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Installation

```bash
pip install -r requirements.txt
python src/main.py
```

## Visualization Explanation

Our visualization demonstrates the key retrieval mechanism in RAG:

- The query point (red star) represents the user's question
- Document points (blue dots) represent potential knowledge sources
- The distance between points indicates semantic similarity
- In a real RAG system, the closest documents would be retrieved and used to augment the prompt
- This visual representation helps understand why certain documents are retrieved for specific queries.

![Figure_1](https://github.com/user-attachments/assets/1de7415c-b41a-4306-81b3-33479fce477d)




