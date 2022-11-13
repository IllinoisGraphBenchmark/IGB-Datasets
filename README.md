# IGB260M-Datasets

## Abstract

Graphs are powerful data structures that solve complex problems like recommender systems, fraud detection, and influence prediction. 
However, graphs alone have limited information, and mining this information to get additional insights is often challenging. 
Graph Neural Nets (GNNs), a class of deep neural networks, have become a popular method to process graph data and to solve a wide variety of emerging downstream tasks like classification, clustering, molecular and protein structure prediction, recommendation, and predicting user behavior in e-commerce and social networks and extracting meaningful insights from unstructured data. 
GNNs have proliferated these emerging applications due to their unique capabilities, like incorporating node, edge, and graph-level information into the output prediction. 
However, dataset sizes have plagued the development of GNNs due to the proprietary nature of industry data, limited size, and the non-availability of synthetic datasets.

In this work, we introduce the **Illinois Graph Benchmark (IGB)**, a collection of enormous graph datasets for node classification tasks. IGB incorporates the most extensive real-world homogeneous graph with 260 million nodes and more than three billion edges, including 220 million labeled nodes for node classification tasks. 
Compared to the largest graph dataset publicly available, IGB provides over 162x more labeled data for deep learning practitioners and developers to create and evaluate the model with higher accuracy. 
IGB captures relational information in the Microsoft Academic Graph for the edges and nodes and the Semantic Scholar database for the node labels. 
IGB also comprises synthetic and real graph datasets where the synthetic dataset has randomly initialized node embeddings while the real graph dataset has variable dimension node embeddings generated using Sentence-BERT models. 
IGB provides a comprehensive study on the impact of embedding generation and large labeled nodes on various GNN models. 
IGB is compatible with popular GNN frameworks like DGL and PyTorch Geometric and comes with predefined popular models like graph Convolutional Neural Networks (GCN), GraphSAGE, and Graph Attention Network (GAT) for easy model development.

## Homogeneous Dataset Metrics


