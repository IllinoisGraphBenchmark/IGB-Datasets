import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv

class SAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers=2, dropout=0.2):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, h_feats, aggregator_type='mean'))
        for _ in range(num_layers-2):
            self.layers.append(SAGEConv(h_feats, h_feats, aggregator_type='mean'))
        self.layers.append(SAGEConv(h_feats, num_classes, aggregator_type='mean'))
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.num_dst_nodes()]
            h = layer(block, (h, h_dst))
            
            if l != len(self.layers) - 1:
                h = self.dropout(h)
                h = F.relu(h)
        return h

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers=2, dropout=0.2):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, h_feats))
        for _ in range(num_layers-2):
            self.layers.append(GraphConv(h_feats, h_feats))
        self.layers.append(GraphConv(h_feats, num_classes))
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.num_dst_nodes()]
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.dropout(h)
                h = F.relu(h)
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_heads, num_layers=2, dropout=0.2):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_feats, h_feats, num_heads))
        for _ in range(num_layers-2):
            self.layers.append(GATConv(h_feats * num_heads, h_feats, num_heads))
        self.layers.append(GATConv(h_feats * num_heads, num_classes, num_heads))
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.num_dst_nodes()]
            if l < len(self.layers) - 1:
                h = layer(block, (h, h_dst)).flatten(1)
                h = F.relu(h)
                h = self.dropout(h)
            else:
                h = layer(block, (h, h_dst)).mean(1)  
        return h