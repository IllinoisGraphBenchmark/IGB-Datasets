import numpy as np
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import pickle

NUM_NODES = 100000000
SIZE = 'large'

if __name__ == '__main__':
    path = '/mnt/nvme15/IGB260M_part_2/processed/paper__cites__paper/edge_index.npy'
    edges = np.load(path)

    large_edge = []

    for edge1, edge2 in tqdm(edges):
        if edge1 < NUM_NODES and edge2 < NUM_NODES:
            large_edge.append([edge1, edge2])

    print(len(large_edge))
    output = '/mnt/nvme5/' + SIZE + '/processed/paper__cites__paper/edge_index.npy'
    np.save(output, np.array(large_edge)) 

