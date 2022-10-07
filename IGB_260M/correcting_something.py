import numpy as np


if __name__ == '__main__':

    
    path = '/mnt/nvme16/node_feat.npy'
    output = '/mnt/nvme15/IGB260M_part_2/processed/paper/node_feat.npy'
    fp = np.memmap(output, dtype='float32', mode='w+',  shape=(157675969, 1024))
    print(fp.shape)
    info = np.load(path, mmap_mode='r')
    print(info.shape)
    fp[:,:] = info[:, :]
    
    
    