import numpy as np
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import pickle

NUM_NODES = 10000000
SIZE = 'medium'

if __name__ == '__main__':

    exp_edges = []

    # # paper labels
    with open('/mnt/nvme12/paper_labels_19_classes.npy', 'rb') as f:  
        paper_labels = np.load(f, allow_pickle=True).tolist()
    with open('/mnt/nvme12/paper_labels_2K_classes.npy', 'rb') as f:  
        paper_labels_large = np.load(f, allow_pickle=True).tolist()
    papers_with_labels = set(list(paper_labels.keys()))

    # paper features
    path = '/mnt/nvme13/paper_emb_part_2/'
    files = [join(path,f) for f in listdir(path) if (isfile(join(path, f)))][::-1]
    path = '/mnt/nvme12/paper_emb_part_1/'
    files.extend([join(path,f) for f in listdir(path) if (isfile(join(path, f)))][::-1])
    print(files)

    exp_feat_with_labels = np.memmap('/mnt/nvme14/IGB260M/' + SIZE + '/processed/paper/node_feat_memmapped.npy', dtype='float32', mode='w+', shape=(NUM_NODES,1024))
    exp_labels_19 = np.memmap('/mnt/nvme14/IGB260M/' + SIZE + '/processed/paper/node_label_19_memmapped.npy', dtype='float32', mode='w+', shape=(NUM_NODES))
    exp_labels_2K = np.memmap('/mnt/nvme14/IGB260M/' + SIZE + '/processed/paper/node_label_2K_memmapped.npy', dtype='float32', mode='w+', shape=(NUM_NODES))
    fp = open("/mnt/nvme6/gnndataset/notebooks/node_degree_map.pickle", "rb")
    node_degree = pickle.load(fp)
    # first take all the paper_ids with labels
    idx = 0
    paper_id_idx_mapping = dict()
    count_of_paper_left = 0
    for filename in tqdm(files):
        paper_feats = np.load(filename, allow_pickle=True)
        for paper_id, feats in paper_feats:
            if paper_id in papers_with_labels:
                if paper_id in node_degree:
                    if node_degree[paper_id] > 12:
                        paper_id_idx_mapping[paper_id] = idx
                        exp_feat_with_labels[idx] = feats
                        exp_labels_19[idx] = paper_labels[paper_id]
                        exp_labels_2K[idx] = paper_labels_large[paper_id]
                        idx += 1
            if idx == NUM_NODES:
                break
        else:
            continue 
        break

    np.save('/mnt/nvme14/IGB260M/' + SIZE + '/processed/paper/node_feat.npy', exp_feat_with_labels)
    np.save('/mnt/nvme14/IGB260M/' + SIZE + '/processed/paper/node_label_19.npy', exp_labels_19)
    np.save('/mnt/nvme14/IGB260M/' + SIZE + '/processed/paper/node_label_2K.npy', exp_labels_2K)
    del paper_labels
    del paper_labels_large

    with open('/mnt/nvme14/IGB260M/' + SIZE + '/processed/paper/paper_id_index_mapping.npy', 'wb') as f:
        np.save(f, paper_id_idx_mapping) 

    # paper cites paper
    paper_edges = np.load('/mnt/nvme12/final_edges.npy', mmap_mode = 'r')
    for edge in tqdm(paper_edges):
        if edge[0] in paper_id_idx_mapping and edge[1] in paper_id_idx_mapping:
            exp_edges.append(np.array([paper_id_idx_mapping[edge[0]], paper_id_idx_mapping[edge[1]]]))
            exp_edges.append(np.array([paper_id_idx_mapping[edge[1]], paper_id_idx_mapping[edge[0]]]))
    print("Finished paper edges")

    # with open('/mnt/nvme14/IGB260M/' + SIZE + '/processed/paper/paper_id_index_mapping.npy', 'wb') as f:
    #     np.save(f, paper_id_idx_mapping) 

    del paper_id_idx_mapping

    print(np.array(exp_edges).shape)
    
    with open('/mnt/nvme14/IGB260M/' + SIZE + '/processed/paper__cites__paper/edge_index.npy', 'wb') as f:
        np.save(f, np.array(exp_edges)) 

