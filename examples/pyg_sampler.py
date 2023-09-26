# Libraries required for PyGSampler
import dgl

# Libraries required for sanity checks
from collections import Counter
from igb.dataloader import IGBHeteroDGLDataset
import torch
import argparse
import tqdm
import math

class PyGSampler(dgl.dataloading.Sampler):
    r"""
    An example DGL sampler implementation that matches PyG/GLT sampler behavior. 
    The following differences need to be addressed: 
    1.  PyG/GLT applies conv_i to edges in layer_i, and all subsequent layers, while DGL only applies conv_i to edges in layer_i. 
        For instance, consider a path a->b->c. At layer 0, 
        DGL updates only node b's embedding with a->b, but 
        PyG/GLT updates both node b and c's embeddings.
        Therefore, if we use h_i(x) to denote the hidden representation of node x at layer i, then the output h_2(c) is: 
            DGL:     h_2(c) = conv_2(h_1(c), h_1(b)) = conv_2(h_0(c), conv_1(h_0(b), h_0(a)))
            PyG/GLT: h_2(c) = conv_2(h_1(c), h_1(b)) = conv_2(conv_1(h_0(c), h_0(b)), conv_1(h_0(b), h_0(a)))
    2.  When creating blocks for layer i-1, DGL not only uses the destination nodes from layer i, 
        but also includes all subsequent i+1 ... n layers' destination nodes as seed nodes.
    More discussions and examples can be found here: https://github.com/alibaba/graphlearn-for-pytorch/issues/79. 
    """
    def __init__(self, fanouts):
        super().__init__()
        self.fanouts = fanouts

    def sample(self, g, seed_nodes):
        output_nodes = seed_nodes
        subgs = []
        previous_edges = {}
        previous_seed_nodes = seed_nodes
        for fanout in reversed(self.fanouts):
            # Sample a fixed number of neighbors of the current seed nodes.
            sg = g.sample_neighbors(seed_nodes, fanout)

            # Before we add the edges, we need to first record the source nodes (of the current seed nodes)
            # so that other edges' source nodes will not be included as next layer's seed nodes. 
            temp = dgl.to_block(sg, previous_seed_nodes, include_dst_in_src=False)
            seed_nodes = temp.srcdata[dgl.NID]

            # We add all previously accumulated edges to this subgraph
            for etype in previous_edges:
                sg.add_edges(*previous_edges[etype], etype=etype)
            
            # This subgraph now contains all its new edges 
            # and previously accumulated edges
            # so we add the 
            previous_edges = {}
            for etype in sg.etypes:
                previous_edges[etype] = sg.edges(etype=etype)

            # Convert this subgraph to a message flow graph.
            # we need to turn on the include_dst_in_src
            # so that we get compatibility with DGL's OOTB GATConv. 
            sg = dgl.to_block(sg, previous_seed_nodes, include_dst_in_src=True)

            # for this layers seed nodes - 
            # they will be our next layers' destination nodes
            # so we add them to the collection of previous seed nodes. 
            previous_seed_nodes = sg.srcdata[dgl.NID]
            
            # we insert the block to our list of blocks
            subgs.insert(0, sg)
            input_nodes = seed_nodes
        return input_nodes, output_nodes, subgs


# Sanity check
def test_correct_layers(blocks, seed_nodes, ntypes, etypes):
    """
    To verify the correctness of this sampler, we need to make sure the following things at each layer i:
    1. All previous edges (layer i+1, ..., layer n) are included in this layer. 
        This ensures that DGL matches GLT/PyG in bullet point 1 of the issue list.  
    2. All incremental new edges <src_i -> dst_i> must have the destination nodes dst_i as last layer's source node src_{i-1}. 
        This ensures that DGL matches GLT/PyG in bullet point 2 of the issue list.

    DGL's native sampler violates both rules, therefore it will not pass the two assertions. 
    GLT sampler **with trim_to_layer** and the above PyGSampler will pass this test.  
    """
    edges = {etype: [] for etype in etypes}
    last_layer_src_nodes = {ntype: nodes.tolist() for ntype, nodes in seed_nodes.items()}
    for block in reversed(blocks):
        src_nodes_copy = {}
        for srctype, etype, dsttype in block.canonical_etypes:
            srcids = block.srcdata[dgl.NID][srctype]
            dstids = block.dstdata[dgl.NID][dsttype]
            current_layer_edge_indices = [(srcids[srcid].item(), dstids[dstid].item()) for srcid, dstid in zip(*block.edges(etype=etype))]
            current_layer_edge_counter = Counter(current_layer_edge_indices)
            last_layer_edge_counter = Counter(edges[(srctype, etype, dsttype)])
            
            # make sure that all previous edges are in the current layer
            assert last_layer_edge_counter - current_layer_edge_counter == {}, f"{dict(last_layer_edge_indices - current_layer_edge_indices)}"

            # make sure that all newly-added edges' dstnodes are in-edges to last layer src nodes (this layer's seed/dst nodes)
            assert all([
                dstnode in last_layer_src_nodes[dsttype]
                for _, dstnode in (current_layer_edge_counter - last_layer_edge_counter).keys()
            ]), f"{current_layer_edge_counter - last_layer_edge_counter}, {last_layer_src_nodes[dsttype]}"
                
            src_nodes_copy[srctype] = srcids.tolist()
            edges[(srctype, etype, dsttype)] = current_layer_edge_indices
        last_layer_src_nodes = src_nodes_copy


# We repeat the experiment several times to avoid edge cases
def correctness_test(n_repeats, n_batches, batch_size, fan_outs, graph):
    for _ in range(n_repeats):
        sampler = PyGSampler(fan_outs)
        # sampler = dgl.dataloading.MultiLayerNeighborSampler(fan_outs) # this will not pass the test
        loader = dgl.dataloading.DataLoader(graph, {"paper": graph.nodes('paper')}, sampler, batch_size=batch_size, shuffle=True)
        counter = 0
        for _, seed_nodes, blocks in tqdm.tqdm(loader, total=min(n_batches, math.ceil(graph.nodes('paper').shape[0] / batch_size))):
            if counter >= n_batches:
                break
            test_correct_layers(blocks, seed_nodes, graph.ntypes, graph.canonical_etypes)
            counter += 1
    print(f"Correctness test passed. Sampled {fan_outs} with random {n_batches * batch_size} nodes as seed nodes for {n_repeats} times.")


if __name__ == "__main__":
    # we perform some sanity checks here. 

    # Prepare graphs
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/data', 
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
    graph = IGBHeteroDGLDataset(args)[0]

    # monotonically increasing fanouts
    correctness_test(10, 64, 128, [10,15], graph)

    # constant fanouts
    correctness_test(10, 64, 128, [10,10], graph)

    # monotonically decreasing fanouts
    correctness_test(10, 64, 128, [15,10], graph)

    # 3-layer, we experiment the same thing
    correctness_test(10, 64, 128, [5,10,15], graph)
    correctness_test(10, 64, 128, [10,10,10], graph)
    correctness_test(10, 64, 128, [15,10,5], graph)
    