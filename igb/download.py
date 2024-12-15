import argparse, tarfile, hashlib, os, requests
from colorama import Fore, Back, Style
from tqdm import tqdm
import urllib.request as ur
import os

GBFACTOR = float(1 << 30)

def decide_download(url):
    d = ur.urlopen(url)
    size = int(d.info()["Content-Length"])/GBFACTOR
    ### confirm if larger than 1GB
    if size > 1:
        return input("This will download %.2fGB. Will you proceed? (y/N) " % (size)).lower() == "y"
    else:
        return True
    

dataset_urls = {
    'homogeneous' : {
        'tiny' : 'https://igb-public-awsopen.s3.amazonaws.com/igb-homogeneous/igb_homogeneous_tiny.tar.gz',
        'small' : 'https://igb-public-awsopen.s3.amazonaws.com/igb-homogeneous/igb_homogeneous_small.tar.gz',
        'medium' : 'https://igb-public-awsopen.s3.amazonaws.com/igb-homogeneous/igb_homogeneous_medium.tar.gz'
    },
    'heterogeneous' : {
        'tiny' : 'https://igb-public-awsopen.s3.amazonaws.com/igb-heterogeneous/igb_heterogeneous_tiny.tar.gz',
        'small' : 'https://igb-public-awsopen.s3.amazonaws.com/igb-heterogeneous/igb_heterogeneous_small.tar.gz',
        'medium' : 'https://igb-public-awsopen.s3.amazonaws.com/igb-heterogeneous/igb_heterogeneous_medium.tar.gz'
    }  
}


md5checksums = {
    'homogeneous' : {
        'tiny' : '34856534da55419b316d620e2d5b21be',
        'small' : '6781c699723529902ace0a95cafe6fe4',
        'medium' : '4640df4ceee46851fd18c0a44ddcc622'
    },
    'heterogeneous' : {
        'tiny' : '83fbc1091497ff92cf20afe82fae0ade',
        'small' : '2f42077be60a074aec24f7c60089e1bd',
        'medium' : '7f0df4296eca36553ff3a6a63abbd347'
    }  
}


def check_md5sum(dataset_type, dataset_size, filename):
    original_md5 = md5checksums[dataset_type][dataset_size]

    with open(filename, 'rb') as file_to_check:
        data = file_to_check.read()    
        md5_returned = hashlib.md5(data).hexdigest()

    if original_md5 == md5_returned:
        print(" md5sum verified.")
        return
    else:
        os.remove(filename)
        raise Exception(" md5sum verification failed!.")
        

def download_dataset(path, dataset_type, dataset_size, confirm_download=False):
    output_directory = path
    url = dataset_urls[dataset_type][dataset_size]
    filename = path + "/igb_" + dataset_type + "_" + dataset_size + ".tar.gz"
    # check if the dataset is already downloaded
    if os.path.exists(filename):
        print("Dataset already downloaded.")
    elif confirm_download or os.environ.get('SKIP_USER_PROMPT', '').lower() in [ "yes", "1", "true" ] or decide_download(url):
        data = ur.urlopen(url)
        size = int(data.info()["Content-Length"])
        chunk_size = 1024*1024
        num_iter = int(size/chunk_size) + 2
        downloaded_size = 0
        with open(filename, 'wb') as f:
            pbar = tqdm(range(num_iter))
            for i in pbar:
                chunk = data.read(chunk_size)
                downloaded_size += len(chunk)
                pbar.set_description("Downloaded {:.2f} GB".format(float(downloaded_size)/GBFACTOR))
                f.write(chunk) 
        print("Downloaded" + Fore.GREEN + " igb_" + dataset_type + "_" + dataset_size + Style.RESET_ALL, end=" ->")
    check_md5sum(dataset_type, dataset_size, filename)
    destination = output_directory
    file = tarfile.open(filename)
    file.extractall(output_directory)
    file.close()
    size = 0
    for path, dirs, files in os.walk(output_directory+"/"+dataset_size):
        for f in files:
            fp = os.path.join(path, f)
            size += os.path.getsize(fp)
    print("Final dataset size {:.2f} GB.".format(size/GBFACTOR))
    os.remove(filename)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/root/heterogeneous_dataset_gen/testing_downloads', 
        help='path containing the datasets')
    parser.add_argument('--dataset_type', type=str, default='homogeneous',
        choices=['homogeneous', 'heterogeneous'], 
        help='dataset type')
    parser.add_argument('--dataset_size', type=str, default='tiny',
        choices=['tiny', 'small', 'medium'], 
        help='size of the datasets')
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        default=False,
        help="To skip the user prompt for confirming the download, which is useful for Docker detached mode.")
    args = parser.parse_args()    
    download_dataset(args.path, args.dataset_type, args.dataset_size, args.confirm_download)
    
