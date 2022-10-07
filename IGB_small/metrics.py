from collections import defaultdict, Counter
import numpy as np
import os.path as osp
from tqdm import tqdm
import cugraph
from scipy.sparse import coo_matrix
import cudf
from collections import OrderedDict
from cugraph.experimental.datasets import karate

# class IGL260MDataset(object):
#     def __init__(self, root: str, size: str, in_memory: int, classes: int):
#         self.dir = root
#         self.size = size
#         self.in_memory = in_memory
#         self.num_classes = classes
    
#     def num_features(self) -> int:
#         return 1024
    
#     def num_classes(self, type_of_class: str) -> int:
#         if type_of_class == 'small':
#             return 19
#         else:
#             return 2983

#     @property
#     def paper_feat(self) -> np.ndarray:
#         path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_feat.npy')
#         if self.in_memory:
#             return np.load(path)
#         else:
#             return np.load(path, mmap_mode='r')

#     @property
#     def paper_label(self) -> np.ndarray:
#         if self.num_classes == 19:
#             path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_19.npy')
#         else:
#             path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_2K.npy')
#         if self.in_memory:
#             return np.load(path)
#         else:
#             return np.load(path, mmap_mode='r')

#     @property
#     def paper_edge(self) -> np.ndarray:
#         path = osp.join(self.dir, self.size, 'processed', 'paper__cites__paper', 'edge_index.npy')
#         if self.in_memory:
#             return np.load(path)
#         else:
#             return np.load(path, mmap_mode='r')

#     @property
#     def paper_mapping(self) -> np.ndarray:
#         path = osp.join(self.dir, self.size, 'processed', 'paper', 'paper_id_index_mapping.npy')
#         return np.load(path, allow_pickle=True).tolist()

SIZE = 100000000

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
        path = '/mnt/nvme16/node_feat.npy'
        node_features = np.memmap(path, dtype='float32', mode='r',  shape=(100000000,1024))
        return node_features

    @property
    def paper_label(self) -> np.ndarray:
        path = '/mnt/nvme15/IGB260M_part_2/processed/paper/node_label_19.npy'
        node_features = np.memmap(path, dtype='float32', mode='r',  shape=(100000000)).astype(int)
        return node_features

    @property
    def paper_edge(self) -> np.ndarray:
        path = '/mnt/nvme5/large/processed/paper__cites__paper/edge_index.npy'
        return np.load(path)

if __name__ == '__main__':

    SIZE = 'full'
    NUM_CLASSES = 19
    dataset = IGL260MDataset(root='/mnt/nvme14/IGB260M/', size=SIZE, in_memory=0, classes=NUM_CLASSES)
    edges = dataset.paper_edge
    labels = dataset.paper_label

    print("Graph Dataset Name: IGB260M-"+SIZE)
    print(" Number of nodes: {:,}".format(len(labels)))
    print(" Number of edges: {:,}".format(len(edges)))
    print(" Num of classes:  {:,}".format(NUM_CLASSES))
    print()
    print(" ---- DEGREE ----")
    edge_degree = [0 for _ in range(len(labels))]
    homophily = defaultdict(lambda: defaultdict(int))
    for edge1, edge2 in tqdm(edges):
        edge_degree[edge1] += 1
        if labels[edge1] == labels[edge2]:
            homophily[edge1][0] += 1
            homophily[edge1][1] += 1
        else:
            homophily[edge1][1] += 1

    print(" Max: {:.2f}".format(np.max(edge_degree)))
    print(" Min: {:.2f}".format(np.min(edge_degree)))
    print(" Avg: {:.2f}".format(np.mean(edge_degree)))
    print(" Std: {:.2f}".format(np.std(edge_degree)))
    edge_dist = Counter(edge_degree)
    print()
    print(" ---- HOMOPHILY ----")
    print(" Homopily: {:.2f}%".format(np.mean([val[0]/val[1] for val in homophily.values()])*100))
    del homophily
    del edge_degree
    print()
    print(" ---- OTHER METRICS ----")
    edges = cudf.DataFrame([ (edge1, edge2, 1) for edge1, edge2 in edges], columns=['src', 'dst', 'weight'])
    g = cugraph.Graph()
    g.from_cudf_edgelist(
        edges
        , source='src'
        , destination='dst'
        , edge_attr='weight'
        , renumber=False
    )
    df = cugraph.weakly_connected_components(g)
    label_gby = df.groupby('labels')
    label_count = label_gby.count()
    print(" Total number of weakly connected components found : {:,}".format(len(label_count)))
    print()
    print(" ---- EXTRA STUFF ----")
    print(" Distribution: ", edge_dist)

    # for edge1, edge2 in tqdm(edges):
    #     if labels[edge1] == labels[edge2]:
    #         homophily[edge1][0] += 1
    #         homophily[edge1][1] += 1
    #         homophily[edge2][0] += 1
    #         homophily[edge2][1] += 1
    #     else:
    #         homophily[edge1][1] += 1
    #         homophily[edge2][1] += 1
    # edges = cudf.DataFrame([ (edge1, edge2, 1) for edge1, edge2 in edges], columns=['src', 'dst', 'weight'])
    # g = cugraph.Graph()
    # g.from_cudf_edgelist(
    # edges
    # , source='src'
    # , destination='dst'
    # , edge_attr='weight'
    # , renumber=False
    # )
    # triangles_per_vertex = cugraph.triangle_count(g)
    # cugraph_triangle_results = \
    #             triangles_per_vertex["counts"].sum()
    # print(cugraph_triangle_results)
    # df = cugraph.weakly_connected_components(g)
    # label_gby = df.groupby('labels')
    # label_count = label_gby.count()
    # print("Total number of weakly connected components found : ", len(label_count))

    # df = cugraph.strongly_connected_components(g)
    # label_gby = df.groupby('labels')
    # label_count = label_gby.count()
    # print("Total number of strongly connected components found : ", len(label_count))

    # adj_list = coo_matrix((values, (sources, destinations))).tocsr()
    # g = cugraph.Graph()
    # g.from_cudf_adjlist(cudf.Series(adj_list.indptr), cudf.Series(adj_list.indices))
    # # print(cugraph.betweenness_centrality(g, k=20, normalized=True, weight=None))
    # df = cugraph.weakly_connected_components(g)
    # print(df.head())
    # label_gby = df.groupby('labels')
    # label_count = label_gby.count()

    # print("Total number of components found : ", len(label_count))

    # # print(g)
    # # triangles_per_vertex = cugraph.triangle_count(g)
    # # cugraph_triangle_results = triangles_per_vertex["counts"].sum()
    # # print(cugraph_triangle_results)




    # # # # print(len(labels))

    # # label_distribution = Counter(labels[:])
    # # print(label_distribution)
    # # print(len(label_distribution))

    # # # Homophily
    # # homophily = defaultdict(lambda: defaultdict(int))
    # for edge1, edge2 in tqdm(edges):
    #     if labels[edge1] == labels[edge2]:
    #         homophily[edge1][0] += 1
    #         homophily[edge1][1] += 1
    #         homophily[edge2][0] += 1
    #         homophily[edge2][1] += 1
    #     else:
    #         homophily[edge1][1] += 1
    #         homophily[edge2][1] += 1

    # homophily = np.mean([val[0]/val[1] for val in homophily.values()])
    # print(homophily)

# root@399cf41c7af4:/mnt/nvme6/gnndataset/homogeneous_graph_generation/IGB_small# python metrics.py 
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 447416/447416 [00:02<00:00, 166089.45it/s]
# 0.5678557243257698
# root@399cf41c7af4:/mnt/nvme6/gnndataset/homogeneous_graph_generation/IGB_small# python metrics.py 
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12070502/12070502 [01:29<00:00, 135307.02it/s]
# 0.5662297463990068
# root@399cf41c7af4:/mnt/nvme6/gnndataset/homogeneous_graph_generation/IGB_small# python metrics.py 
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 120077694/120077694 [18:23<00:00, 108830.94it/s]
# 0.5993406947326637




