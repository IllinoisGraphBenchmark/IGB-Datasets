import dgl
from dgl.data import DGLDataset
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv
import os.path as osp
from sys import getsizeof
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# SIZE = 'large'
# NODES = 100000000

# file_path = '/mnt/nvme1/'
file_path = '/mnt/nvme6/IGB260M/'
# file_path = '/mnt/nvme15/IGB260M_part_2'

class IGL260MDataset(object):
    def __init__(self, root: str, size: str, in_memory: int, classes: int):
        self.dir = root
        self.size = size
        self.in_memory = in_memory
        self.num_classes = classes
    

    def num_features(self) -> int:
        return 1024
    

    def num_classes(self, type_of_class: str) -> int:
        if type_of_class == 'small':
            return 19
        else:
            return 2983

    @property
    def paper_feat(self) -> np.ndarray:
        path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_feat.npy')
        if self.in_memory:
            return np.load(path)
        else:
            return np.load(path, mmap_mode='r')

    @property
    def paper_label(self) -> np.ndarray:
        if self.num_classes == 19:
            path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_19_test.npy')
        else:
            path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_2K.npy')
        if self.in_memory:
            return np.load(path)
        else:
            return np.load(path, mmap_mode='r')

    @property
    def paper_edge(self) -> np.ndarray:
        path = osp.join(self.dir, self.size, 'processed', 'paper__cites__paper', 'edge_index.npy')
        if self.in_memory:
            return np.load(path)
        else:
            return np.load(path, mmap_mode='r')

class IGL260M(DGLDataset):
    def __init__(self, args):
        super().__init__(name='IGB260M')

    def process(self):
        dataset = IGL260MDataset(root='/mnt/nvme14/IGB260M/', size=args.dataset_size, in_memory=args.in_memory, classes=args.num_classes)
        node_features = torch.from_numpy(dataset.paper_feat)
        node_edges = torch.from_numpy(dataset.paper_edge)
        node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)

        self.graph = dgl.graph((node_edges[:, 0],node_edges[:, 1]), num_nodes=node_features.shape[0])

        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        
        self.graph = dgl.remove_self_loop(self.graph)
        self.graph = dgl.add_self_loop(self.graph)
        
        n_nodes = node_features.shape[0]

        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            if l != len(self.layers) - 1:
                # h = self.activation(h)
                h = self.dropout(h)
            h = layer(block, h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.number_of_nodes(), self.n_hidden if l !=
                         len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.number_of_nodes()),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=4)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y
class GAT(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, num_heads, activation
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.GATConv(
                (in_feats, in_feats),
                n_hidden,
                num_heads=num_heads,
                activation=activation,
            )
        )
        for i in range(1, n_layers - 1):
            self.layers.append(
                dglnn.GATConv(
                    (n_hidden * num_heads, n_hidden * num_heads),
                    n_hidden,
                    num_heads=num_heads,
                    activation=activation,
                )
            )
        self.layers.append(
            dglnn.GATConv(
                (n_hidden * num_heads, n_hidden * num_heads),
                n_classes,
                num_heads=num_heads,
                activation=None,
            )
        )

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[: block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            if l < self.n_layers - 1:
                h = layer(block, (h, h_dst)).flatten(1)
            else:
                h = layer(block, (h, h_dst))
        h = h.mean(1)
        return h.log_softmax(dim=-1)

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GAT model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        # TODO: make thiw into a variable
        num_heads = 2
        for l, layer in enumerate(self.layers):
            if l < self.n_layers - 1:
                y = torch.zeros(
                    g.num_nodes(),
                    self.n_hidden * num_heads
                    if l != len(self.layers) - 1
                    else self.n_classes,
                )
            else:
                y = torch.zeros(
                    g.num_nodes(),
                    self.n_hidden
                    if l != len(self.layers) - 1
                    else self.n_classes,
                )

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                torch.arange(g.num_nodes()),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=4,
            )

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0].int().to(device)

                h = x[input_nodes].to(device)
                h_dst = h[: block.num_dst_nodes()]
                if l < self.n_layers - 1:
                    h = layer(block, (h, h_dst)).flatten(1)
                else:
                    h = layer(block, (h, h_dst))
                    h = h.mean(1)
                    h = h.log_softmax(dim=-1)

                y[output_nodes] = h.cpu()

            x = y
        return y
class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                aggregator_type):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, aggregator_type))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, aggregator_type))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, aggregator_type))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.number_of_nodes(), self.n_hidden if l !=
                         len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.number_of_nodes()),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=4)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, inputs, labels, val_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid])

def load_subtensor(g, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = g.ndata['labels'][seeds].to(device)
    return batch_inputs, batch_labels

def track_acc(g, args, model_type):
    train_accuracy = []
    test_accuracy = []
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    in_feats = g.ndata['features'].shape[1]
    n_classes = args.num_classes

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    g.create_formats_()

    num_epochs = args.epochs
    num_hidden = args.hidden_channels
    num_layers = args.num_layers
    fan_out = args.fan_out
    batch_size = args.batch_size
    lr = args.learning_rate
    dropout = args.dropout
    num_workers = args.num_workers

    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
                [int(fanout) for fanout in fan_out.split(',')])
    
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)

    # Define model and optimizer
    if model_type == 'gcn':
        model = GCN(in_feats, num_hidden, n_classes, 1, F.relu, dropout)
    if model_type == 'sage':
        model = SAGE(in_feats, num_hidden, n_classes, num_layers, F.relu, dropout, 'gcn')
    if model_type == 'gat':
        model = GAT(in_feats, num_hidden, n_classes, num_layers, 2, F.relu)

    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.decay)

     # Training loop
    avg = 0
    best_test_acc = 0
    log_every = 1
    training_start = time.time()
    for epoch in tqdm(range(num_epochs)):
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        epoch_loss = 0
        gpu_mem_alloc = 0
        epoch_start = time.time()
        for step, (input_nodes, seeds, blocks) in (enumerate(dataloader)):
            # Load the input features as well as output labels
            #batch_inputs, batch_labels = load_subtensor(g, seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach()

            gpu_mem_alloc += (
                torch.cuda.max_memory_allocated() / 1000000
                if torch.cuda.is_available()
                else 0
            )

        train_g = g
        train_nid = torch.nonzero(
            train_g.ndata['train_mask'], as_tuple=True)[0]
        train_acc = evaluate(
            model, train_g, train_g.ndata['features'], train_g.ndata['labels'], train_nid, batch_size, device)
        
        test_g = g
        test_nid = torch.nonzero(
            test_g.ndata['test_mask'], as_tuple=True)[0]
        test_acc = evaluate(
            model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, batch_size, device)

        if test_acc.item() > best_test_acc:
            best_test_acc = test_acc.item()
        tqdm.write(
            "Epoch {:05d} | Loss {:.4f} | Train Acc {:.4f} | Test Acc {:.4f} | Time {:.2f}s | GPU {:.1f} MB".format(
                epoch,
                epoch_loss,
                train_acc.item(),
                test_acc.item(),
                time.time() - epoch_start,
                gpu_mem_alloc
            )
        )
        test_accuracy.append(test_acc.item())
        train_accuracy.append(train_acc.item())
        torch.save(model.state_dict(), args.modelpath)
    print()
    print("Total time taken: ", time.time() - training_start)

    return best_test_acc, train_accuracy, test_accuracy


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M/')
    parser.add_argument('--modelpath', type=str, default='gsage_2983.pt')
    parser.add_argument('--dataset_size', type=str, default='medium', choices=['tiny', 'small', 'medium', 'large', 'full'])
    parser.add_argument('--num_classes', type=int, default=2983, choices=[19, 2983])
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--fan_out', type=str, default='5,10')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--learning_rate', type=int, default=0.05)
    parser.add_argument('--decay', type=int, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=2048*16)
    parser.add_argument('--dropout', type=float, default=0.5)


    parser.add_argument('--epochs', type=int, default=5)

    parser.add_argument('--model', type=str, default='gat',
                        choices=['gat', 'sage', 'gcn'])
    parser.add_argument('--in_memory', type=int, default=1)
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    print("Dataset_size: " + args.dataset_size) 
    print("Model       : " + args.model)
    print("Num_classes : " + str(args.num_classes))
    print()
    
    device = f'cuda:2' if torch.cuda.is_available() else 'cpu'

    dataset = IGL260M(args)
    g = dataset[0]

    best_test_acc, train_acc, test_acc = track_acc(g, args, model_type=args.model)

    print(f"Train accuracy: {np.mean(train_acc):.2f} \u00B1 {np.std(train_acc):.2f} \t Best: {np.max(train_acc) * 100:.4f}%")
    print(f"Test accuracy: {np.mean(test_acc):.2f} \u00B1 {np.std(test_acc):.2f} \t Best: {np.max(test_acc) * 100:.4f}%")
    print()
    print(" -------- For debugging --------- ")
    print("Parameters: ", args)
    print(g)
    print("Train accuracy: ", train_acc)
    print("Test accuracy: ", test_acc)


