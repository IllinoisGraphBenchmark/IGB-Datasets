import numpy as np
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import pickle
from collections import defaultdict

if __name__ == '__main__':

    exp_edges = []

    node_degree = np.load('/mnt/nvme12/paper_node_degree.npy', allow_pickle=True).tolist()

    print(len(node_degree))

    degreedist = dict({10: 0, 100: 0, 1000: 0, 10000: 0, 100000: 0, 1000000: 0})
    for paper_id, degree in tqdm(node_degree.items()):
        if degree < 10:
            degreedist[10] += 1
        elif degree < 100:
            degreedist[100] += 1
        elif degree < 1000:
            degreedist[1000] += 1
        elif degree < 10000:
            degreedist[10000] += 1
        elif degree < 100000:
            degreedist[100000] += 1
        elif degree < 1000000:
            degreedist[1000000] += 1

    print(degreedist)

    # # # paper labels
    # with open('/mnt/nvme12/paper_labels_19_classes.npy', 'rb') as f:  
    #     paper_labels = np.load(f, allow_pickle=True).tolist()
    # papers_with_labels = set(list(paper_labels.keys()))


    # # paper cites paper
    # paper_edges = np.load('/mnt/nvme12/final_edges.npy', mmap_mode = 'r')
    # paper_node_degree = defaultdict(int)
    # for edge in tqdm(paper_edges):
    #     if edge[0] in papers_with_labels and edge[1] in papers_with_labels:
    #         paper_node_degree[edge[0]] += 1
    #         paper_node_degree[edge[1]] += 1

    # with open('/mnt/nvme12/paper_node_degree.npy', 'wb') as f:
    #     np.save(f, paper_node_degree) 