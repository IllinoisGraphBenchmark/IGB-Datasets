# IGB260M-Datasets

<p align='center'>
  <img width='60%' src='https://github.com/IllinoisGraphBenchmark/IGB260M-Datasets/blob/main/logo.png' />
</p>

### [Senior Thesis](https://github.com/IllinoisGraphBenchmark/IGB260M-Datasets/blob/main/Arpandeep_Khatua__UG_Senior_Thesis___Illinois_Graph_Benchmark_Dataset.pdf)

## Abstract

In this work, we introduce the **Illinois Graph Benchmark (IGB)**, a collection of enormous graph datasets for node classification tasks. IGB incorporates the most extensive real-world homogeneous graph with 260 million nodes and more than three billion edges, including 220 million labeled nodes for node classification tasks. 
Compared to the largest graph dataset publicly available, IGB provides over 162x more labeled data that MAG240M for deep learning practitioners and developers to create and evaluate the model with higher accuracy. 
IGB captures relational information in the [Microsoft Academic Graph](https://www.microsoft.com/en-us/research/project/microsoft-academic-graph/) for the edges and nodes and the [Semantic Scholar](https://www.semanticscholar.org) database for the node labels. 
IGB also comprises synthetic and real graph datasets where the synthetic dataset has randomly initialized node embeddings while the real graph dataset has variable dimension node embeddings generated using [Sentence-BERT](https://www.sbert.net) models. 
IGB provides a comprehensive study on the impact of embedding generation and large labeled nodes on various GNN models. 
IGB is compatible with popular GNN frameworks like DGL and PyTorch Geometric and comes with predefined popular models like graph Convolutional Neural Networks (GCN), GraphSAGE, and Graph Attention Network (GAT) for easy model development.

## Homogeneous Dataset Metrics

<p align='left'>
  <img width='80%' src='https://github.com/IllinoisGraphBenchmark/IGB260M-Datasets/blob/main/igbmetrics.png' />
</p>

## IGB-full Heterogeneous Dataset Metrics

<p align='left'>
  <img width='80%' src='https://github.com/IllinoisGraphBenchmark/IGB260M-Datasets/blob/main/igb_hetero_schema.png' />
</p>


## Downloading dataset

Hosted on AWS. To get early access scroll to the bottom. Coming soon...

## Usage

### (1) Data loaders 

We have easy to use DGLDataset dataloader and we will soon add a PyTorch Geometric dataloader. The dataloader takes in arguments for the path of the dataset, the number of classes $\in$ [19, 2983]. You can also mention whether to read the data into the memory or in `mmap_mode='r'` incase the dataset doesn't fit in your RAM. (Training becomes significantly slower when reading from disk). We can also choose to get a synthetic node embeddings for testing systems. 

```python
import argparse, dgl
from dataloader import IGB260MDGLDataset

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M/', 
    help='path containing the datasets')
parser.add_argument('--dataset_size', type=str, default='experimental',
    choices=['experimental', 'small', 'medium', 'large', 'full'], 
    help='size of the datasets')
parser.add_argument('--num_classes', type=int, default=19, 
    choices=[19, 2983], help='number of classes')
parser.add_argument('--in_memory', type=int, default=0, 
    choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
parser.add_argument('--synthetic', type=int, default=0,
    choices=[0, 1], help='0:nlp-node embeddings, 1:random')
args = parser.parse_args()

dataset = IGB260MDGLDataset(args)
graph = dataset[0]
print(graph)
# Graph(num_nodes=100000, num_edges=547416,
#      ndata_schemes={'feat': Scheme(shape=(1024,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64), 'train_mask': Scheme(shape=(), #dtype=torch.bool), 'val_mask': Scheme(shape=(), dtype=torch.bool), 'test_mask': Scheme(shape=(), dtype=torch.bool)}
#      edata_schemes={})
```
### (2) Popular GNN Models

We have implmented [Graph Convolutional Neural Net (GCN)](https://arxiv.org/abs/1609.02907), [GraphSAGE](https://arxiv.org/abs/1706.02216) and [Graph Attention Network (GAT)](https://arxiv.org/abs/1710.10903). These models take in the dimension of the input, hidden dimensions and the expected output dimension (which would be your \# classes) along with the number of layers, dropout and in case of the GAT model, num of attention heads.

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

### (3) Baseline

We ran each of these models on the IGB dataset collections to get a baseline. Our goal is to enable GNN researchers to develop and test novel models using this dataset. We expect more robust models due to the presence of massive labeled data. We will released detailed analysis of the runs and the hyperparameters along with other relevant experiments in our upcoming paper. 

<p align='left'>
  <img width='80%' src='https://github.com/IllinoisGraphBenchmark/IGB260M-Datasets/blob/main/igbbaseline.png' />
</p>

We aim to improve these baselines by testing out more hyperparameters. *Models have been trained for 3 epochs with suboptimal hyperparameters on these datasets.

### (4) Multi-GPU Runs

We provide scripts to run the above models on mulitple GPUs using DGL and PyTorch methods. To test it out by running GCN on IGB-tiny with the default hyperparameters you can test it out using:

```python
python train_multi_gpu.py --model_type gcn --dataset_size experimental --num_classes 19 --gpu_devices 0,1,2,3
```
To try single GPU run use:
```python
python train_multi_gpu.py --model_type gcn --dataset_size experimental --num_classes 19 --gpu_devices 0
```
```python
python train_single_gpu.py --model_type gcn --dataset_size experimental --num_classes 19
```

To learn more about the hyperparameters please take a look at `train/train_multi_gpu.py`.

## Inquiries/Getting Involved
Please reach out to [Arpandeep Khatua](mailto:arpandeepk@gmail.com?subject=[IGB260M]%20Inquiry%20regarding%20dataset%20from%20github)




