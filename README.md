# IGB260M-Datasets

<p align='center'>
  <img width='50%' src='https://github.com/IllinoisGraphBenchmark/IGB260M-Datasets/blob/main/logo.png' />
</p>

## Abstract

In this work, we introduce the **Illinois Graph Benchmark (IGB)**, a collection of enormous graph datasets for node classification tasks. IGB incorporates the most extensive real-world homogeneous graph with 260 million nodes and more than three billion edges, including 220 million labeled nodes for node classification tasks. 
Compared to the largest graph dataset publicly available, IGB provides over 162x more labeled data for deep learning practitioners and developers to create and evaluate the model with higher accuracy. 
IGB captures relational information in the [Microsoft Academic Graph](https://www.microsoft.com/en-us/research/project/microsoft-academic-graph/) for the edges and nodes and the [Semantic Scholar](https://www.semanticscholar.org) database for the node labels. 
IGB also comprises synthetic and real graph datasets where the synthetic dataset has randomly initialized node embeddings while the real graph dataset has variable dimension node embeddings generated using [Sentence-BERT](https://www.sbert.net) models. 
IGB provides a comprehensive study on the impact of embedding generation and large labeled nodes on various GNN models. 
IGB is compatible with popular GNN frameworks like DGL and PyTorch Geometric and comes with predefined popular models like graph Convolutional Neural Networks (GCN), GraphSAGE, and Graph Attention Network (GAT) for easy model development.

## Homogeneous Dataset Metrics

<p align='left'>
  <img width='80%' src='https://github.com/IllinoisGraphBenchmark/IGB260M-Datasets/blob/main/igbmetrics.png' />
</p>

## Usage

#### (1) Data loaders 

We have easy to use DGLDataset dataloader and we will soon add a PyTorch Geometric dataloader. 

```python
import dgl
from dataloader import IGB260MDGLDataset

dataset = IGB260MDGLDataset(args)
graph = dataset[0]
```
#### (2) Popular GNN Models

We have implmented Graph Convolutional Neural Net (GCN), GraphSAGE and Graph Attention Network (GAT).

```python
import torch
from models import *

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'

if args.model_type == 'gcn':
    model = GCN(in_feats, args.hidden_channels, args.num_classes, args.num_layers).to(device)
if args.model_type == 'sage':
    model = SAGE(in_feats, args.hidden_channels, args.num_classes, args.num_layers).to(device)
if args.model_type == 'gat':
    model = GAT(in_feats, args.hidden_channels, args.num_classes, args.num_layers, args.num_heads).to(device)
```

#### (3) Baseline

We ran each of these models on the IGB dataset collections to get a baseline.

<p align='left'>
  <img width='80%' src='https://github.com/IllinoisGraphBenchmark/IGB260M-Datasets/blob/main/igbbaseline.png' />
</p>

We aim to improve these baselines by testing out more hyperparameters.

#### (4) Multi-GPU Runs

We provide scripts to run the above models on mulitple GPUs using DGL and PyTorch methods. To test it out by running GCN on IGB-tiny with the default hyperparameters you can test it out using

```python
python train_multi_gpu.py --model_type gcn --dataset_size experimental --num_classes 19
```

To learn more about the hyperparameters please take a look at `train/train_multi_gpu.py`.






