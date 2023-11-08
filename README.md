# IGB-Datasets

<p align='center'>
  <img width='60%' src='https://github.com/IllinoisGraphBenchmark/IGB260M-Datasets/blob/main/doc/figures/logo.png' />
</p>


## Official IGB Leadboard is now online!! 🎉
Head over to the [leaderboard](https://github.com/IllinoisGraphBenchmark/IGB-Datasets/blob/leaderboard-creation/results/README.md#leaderboard-wip) and make your submission. 

## Installation Guide

```python
# Clone the repo
>>> git clone https://github.com/IllinoisGraphBenchmark/IGB-Datasets.git
# Go to the folder root
>>> cd IGB-Datasets
# Install the igb package
>>> pip install .
```
Now in order to get the dataloader you can: `from igb import dataloader`

## Get access to dataset

After you install the igb package in order to download `igb(h)-tiny`, `igb(h)-small`, `igb(h)-medium` please follow this code example.

```python
>>> from igb import download
>>> download.download_dataset(path='/root/igb_datasets', dataset_type='homogeneous', dataset_size'tiny')
Downloaded 0.36 GB: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:03<00:00, 98.94it/s]
Downloaded igb_homogeneous_tiny -> md5sum verified.
Final dataset size 0.39 GB.
```
The script downloads the zipped files from aws, does a md5sum check and extracts the folder in your specified path. Change the `dataset_type` to `"heterogeneous"` and `the dataset_size` to either `"small"` or `"medium"` in order to get the other datasets.

In the current version if you want the download the `igb(h)-large` and `igb260m/igbh600m` please use the bash download scripts provided. Please note these two large datasets require disk space over >500GB. 

### Important update: 
> Note: We have updated the paper embedding file of the full dataset. If you have downloaded the dataset prior to 7th November 2023 you will need to update to get the embeddings for the last ~5M paper nodes. To make the process easier so users don't have to re-download the 1TB paper embedding file please follow these steps to update the embedding in place.

First download the embeddings using 
```bash
wget --recursive --no-parent https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper/node_feat_5M_tail.npy
```

Then run this python script:
```python
import numpy as np
from tqdm import tqdm
# Open the paper embedding file in r+ mode (read/write)
num_paper_nodes = 269346174
paper_node_features = np.memmap('/mnt/raid0/full/processed/paper/node_feat.npy', dtype='float32', 
                                mode='r+',  shape=(num_paper_nodes,1024))

# Open the extra embedding file in read more    
num_tail = 4957567
node_feat_5M_tail = np.memmap('/mnt/raid0/full/processed/paper/node_feat_5M_tail.npy', dtype='float32', 
                                mode='r',  shape=(4957567,1024))

# Here we do it sequencially to log the progress. 
# You can do it in parallel by paper_node_features[offset:] = node_feat_5M_tail
offset = num_paper_nodes-num_tail
for i in tqdm(range(num_tail)):
    paper_node_features[i + offset] = node_feat_5M_tail[i]

# flush to save to disk
paper_node_features.flush()
```

## Abstract
Graph neural networks (GNNs) have shown high potential for a variety of real-world, challenging applications, but one of the major obstacles in GNN research is the lack of large-scale flexible datasets. Most existing public datasets for GNNs are relatively small, which limits the ability of GNNs to generalize to unseen data. The few existing large-scale graph datasets provide very limited labeled data. This makes it difficult to determine if the GNN model's low accuracy for unseen data is inherently due to insufficient training data or if the model failed to generalize. Additionally, datasets used to train GNNs need to offer flexibility to enable a thorough study of the impact of various factors while training GNN models.

In this work, we introduce the Illinois Graph Benchmark (IGB), a research dataset tool that the developers can use to train, scrutinize and systematically evaluate GNN models with high fidelity. **IGB includes both homogeneous and heterogeneous real-world citation graphs of enormous sizes, with more than 40% of their nodes labeled**. Compared to the largest graph datasets publicly available, the IGB provides over 162X more labeled data for deep learning practitioners and developers to create and evaluate models with higher accuracy. The IGB dataset is designed to be flexible, enabling the study of various GNN architectures, embedding generation techniques, and analyzing system performance issues. IGB is open-sourced, supports DGL and PyG frameworks, and comes with releases of the raw text that we believe foster emerging language models and GNN research projects.  


## IGB Homogeneous Dataset Metrics

<p align='left'>
  <img width='80%' src='https://github.com/IllinoisGraphBenchmark/IGB260M-Datasets/blob/main/doc/figures/igbmetrics.png' />
</p>

## IGB Heterogeneous Dataset Metrics

<p align='left'>
  <img width='80%' src='https://github.com/IllinoisGraphBenchmark/IGB260M-Datasets/blob/main/doc/figures/igb_hetero_schema.png' />
</p>


## Downloading dataset

Hosted on AWS. Early access description is provided at the top of this readme.

## Usage

### (1) Data loaders 

We have easy to use DGLDataset dataloader and we will soon add a PyTorch Geometric dataloader. The dataloader takes in arguments for the path of the dataset, the number of classes $\in$ [19, 2983]. You can also mention whether to read the data into the memory or in `mmap_mode='r'` incase the dataset doesn't fit in your RAM. (Training becomes significantly slower when reading from disk). We can also choose to get a synthetic node embeddings for testing systems. 

```python
import argparse, dgl
from dataloader import IGB260MDGLDataset

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M/', 
    help='path containing the datasets')
parser.add_argument('--dataset_size', type=str, default='tiny',
    choices=['tiny', 'small', 'medium', 'large', 'full'], 
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
  <img width='80%' src='https://github.com/IllinoisGraphBenchmark/IGB260M-Datasets/blob/main/doc/figures/igbbaseline.png' />
</p>

We aim to improve these baselines by testing out more hyperparameters. *Models have been trained for 3 epochs with suboptimal hyperparameters on these datasets.

### (4) Multi-GPU Runs

We provide scripts to run the above models on mulitple GPUs using DGL and PyTorch methods. To test it out by running GCN on IGB-tiny with the default hyperparameters you can test it out using:

```
python train_multi_gpu.py --model_type gcn --dataset_size tiny --num_classes 19 --gpu_devices 0,1,2,3  #For homogenous
python train_multi_hetero.py --model_type rgcn --dataset_size tiny --num_classes 19 --gpu_devices 0,1,2,3  #For heterogenous
```
To try single GPU run use:
``` 
python train_multi_gpu.py --model_type gcn --dataset_size tiny --num_classes 19 --gpu_devices 0  #For homogenous
python train_multi_hetero.py --model_type rgcn --dataset_size tiny --num_classes 19 --gpu_devices 0  #For heterogenous
```
or
``` 
python train_single_gpu.py --model_type gcn --dataset_size tiny --num_classes 19
```

To learn more about the hyperparameters please take a look at `train/train_multi_gpu.py` or `train/train_multi_hetero.py`.

## IGB Documentation

Please read our paper in Arxiv. 

## Contributions

Please check the [Contributions.md](Contributions.md) file for more details.

## Questions
1. Please reach out to [Arpandeep Khatua](mailto:arpandeepk@gmail.com?subject=[IGB260M]%20Inquiry%20regarding%20dataset%20from%20github) and [Vikram Sharma Mailthody](mailto:vsm2@illinois.edu?subject=[IGB260M]%20Inquiry%20regarding%20dataset%20from%20github)
2. Please feel free to join our [Slack Channel](TBH).


## Future updates

1. We will be releasing raw text data for enabling NLP+GNN tasks. 
2. Temporal graph datasets.

If you have additional requests, please add them in the issues. 


## Citations
The work is done using the funds from IBM-Illinois Discovery Accelerator Institute and Amazon Research Awards and in collaboration with NVIDIA Research. 
If you use datasets, please cite the below article.

```
@inproceedings{igbdatasets,
  doi = {10.48550/ARXIV.2302.13522},
  url = {https://arxiv.org/abs/2302.13522},
  author = {Khatua, Arpandeep and Mailthody, Vikram Sharma and Taleka, Bhagyashree and Ma, Tengfei and Song, Xiang and Hwu, Wen-mei},  
  title = {IGB: Addressing The Gaps In Labeling, Features, Heterogeneity, and Size of Public Graph Datasets for Deep Learning Research},
  year = {2023},
  booktitle = {In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23)},
  series = {KDD '23}
  copyright = {Creative Commons Attribution 4.0 International}
}
```

