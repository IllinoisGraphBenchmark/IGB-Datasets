# IGB260M-Datasets

<p align='center'>
  <img width='40%' src='https://github.com/IllinoisGraphBenchmark/IGB260M-Datasets/blob/main/logo.png' />
</p>

## Abstract

In this work, we introduce the **Illinois Graph Benchmark (IGB)**, a collection of enormous graph datasets for node classification tasks. IGB incorporates the most extensive real-world homogeneous graph with 260 million nodes and more than three billion edges, including 220 million labeled nodes for node classification tasks. 
Compared to the largest graph dataset publicly available, IGB provides over 162x more labeled data for deep learning practitioners and developers to create and evaluate the model with higher accuracy. 
IGB captures relational information in the Microsoft Academic Graph for the edges and nodes and the Semantic Scholar database for the node labels. 
IGB also comprises synthetic and real graph datasets where the synthetic dataset has randomly initialized node embeddings while the real graph dataset has variable dimension node embeddings generated using Sentence-BERT models. 
IGB provides a comprehensive study on the impact of embedding generation and large labeled nodes on various GNN models. 
IGB is compatible with popular GNN frameworks like DGL and PyTorch Geometric and comes with predefined popular models like graph Convolutional Neural Networks (GCN), GraphSAGE, and Graph Attention Network (GAT) for easy model development.

## Homogeneous Dataset Metrics


<p align='left'>
  <img width='80%' src='https://github.com/IllinoisGraphBenchmark/IGB260M-Datasets/blob/main/igbmetrics.png' />
</p>
