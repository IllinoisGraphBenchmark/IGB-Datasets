import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import torch.multiprocessing as mp
import time, tqdm, numpy as np
from models import *
from dataloader import IGB260MDGLDataset

torch.manual_seed(0)
dgl.seed(0)
import warnings
warnings.filterwarnings("ignore")




def run(proc_id, devices, g, args):

    dev_id = devices[proc_id]
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='12345')
    if torch.cuda.device_count() < 1:
        device = torch.device('cpu')
        torch.distributed.init_process_group(
            backend='gloo', init_method=dist_init_method, world_size=len(devices), rank=proc_id)
    else:
        torch.cuda.set_device(dev_id)
        device = torch.device('cuda:' + str(dev_id))
        torch.distributed.init_process_group(
            backend='nccl', init_method=dist_init_method, world_size=len(devices), rank=proc_id)

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
                [int(fanout) for fanout in args.fan_out.split(',')])

    # sampler = dgl.dataloading.NeighborSampler([4, 4])

    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
    
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g, train_nid, sampler, device='cpu', use_ddp=True,
        batch_size=args.batch_size,
        shuffle=True, drop_last=False,
        num_workers=0)    

    val_dataloader = dgl.dataloading.NodeDataLoader(
        g, val_nid, sampler, device='cpu', use_ddp=True,
        batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        num_workers=0)

    test_dataloader = dgl.dataloading.NodeDataLoader(
        g, test_nid, sampler, device='cpu', use_ddp=True,
        batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        num_workers=0)

    in_feats = g.ndata['feat'].shape[1]

    if args.model_type == 'gcn':
        model = GCN(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers).to(device)
    if args.model_type == 'sage':
        model = SAGE(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers).to(device)
    if args.model_type == 'gat':
        model = GAT(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers, args.num_heads).to(device)

    if device == torch.device('cpu'):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)

    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), 
        lr=args.learning_rate, weight_decay=args.decay)

     # Training loop
    best_accuracy = 0
    training_start = time.time()
    for epoch in range(args.epochs):
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        epoch_loss = 0
        epoch_start = time.time()
        model.train()
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            blocks = [block.to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['feat'].to(device)
            batch_labels = blocks[-1].dstdata['label'].to(device)

            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach()
            train_acc = sklearn.metrics.accuracy_score(batch_labels.cpu().numpy(), 
                batch_pred.argmax(1).detach().cpu().numpy())*100

        model.eval()
        if proc_id == 0:
            if epoch%args.log_every == 0:
                predictions = []
                labels = []
                with torch.no_grad():
                    for _, _, blocks in val_dataloader:
                        blocks = [block.to(device) for block in blocks]
                        inputs = blocks[0].srcdata['feat'].to(device)
                        labels.append(blocks[-1].dstdata['label'].cpu().numpy())
                        predictions.append(model(blocks, inputs).argmax(1).cpu().numpy())
                    predictions = np.concatenate(predictions)
                    labels = np.concatenate(labels)
                    val_acc = sklearn.metrics.accuracy_score(labels, predictions)*100
                    if best_accuracy < val_acc:
                        best_accuracy = val_acc

                print(
                    "Epoch {:03d} | Loss {:.4f} | Train Acc {:.2f} | Val Acc {:.2f} | Time {}".format(
                        epoch,
                        epoch_loss,
                        train_acc,
                        val_acc,
                        str(datetime.timedelta(seconds = int(time.time() - epoch_start)))   
                    )
                )

    model.eval()
    if proc_id == 0:
        predictions = []
        labels = []
        with torch.no_grad():
            for _, _, blocks in test_dataloader:
                blocks = [block.to(device) for block in blocks]
                inputs = blocks[0].srcdata['feat'].to(device)
                labels.append(blocks[-1].dstdata['label'].cpu().numpy())
                predictions.append(model(blocks, inputs).argmax(1).cpu().numpy())
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            test_acc = sklearn.metrics.accuracy_score(labels, predictions)*100
        print("Test Acc {:.2f}%".format(test_acc))
        print("Total time taken " + str(datetime.timedelta(seconds = int(time.time() - training_start))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
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

    # Model
    parser.add_argument('--model_type', type=str, default='gat',
                        choices=['gat', 'sage', 'gcn'])
    parser.add_argument('--modelpath', type=str, default='deletethis.pt')
    parser.add_argument('--model_save', type=int, default=0)

    # Model parameters 
    parser.add_argument('--fan_out', type=str, default='10,15')
    parser.add_argument('--batch_size', type=int, default=10240)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--learning_rate', type=int, default=0.01)
    parser.add_argument('--decay', type=int, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--log_every', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--gpu_devices', type=str, default='0,1,2,3')
    args = parser.parse_args()

    gpu_idx = [int(fanout) for fanout in args.gpu_devices.split(',')]
    num_gpus = len(gpu_idx)
    dataset = IGB260MDGLDataset(args)
    g = dataset[0]
    print(g)

    start = time.time()
    mp.spawn(run, args=(gpu_idx, g, args,), nprocs=num_gpus)