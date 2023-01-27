import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import time, tqdm, numpy as np
from models import *
from dataloader import IGB260MDGLDataset

torch.manual_seed(0)
dgl.seed(0)
import warnings
warnings.filterwarnings("ignore")

def track_acc(g, args, device):

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
                [int(fanout) for fanout in args.fan_out.split(',')])

    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
    
    train_dataloader = dgl.dataloading.DataLoader(
        g, train_nid, sampler,
        batch_size=args.batch_size,
        shuffle=True, drop_last=False,
        num_workers=args.num_workers)    

    val_dataloader = dgl.dataloading.DataLoader(
        g, val_nid, sampler,
        batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        num_workers=args.num_workers)

    test_dataloader = dgl.dataloading.DataLoader(
        g, test_nid, sampler,
        batch_size=args.batch_size,
        shuffle=True, drop_last=False,
        num_workers=args.num_workers)

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

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), 
        lr=args.learning_rate, weight_decay=args.decay)

     # Training loop
    best_accuracy = 0
    training_start = time.time()
    for epoch in tqdm.tqdm(range(args.epochs)):
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        epoch_loss = 0
        gpu_mem_alloc = 0
        epoch_start = time.time()
        model.train()
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['feat']
            batch_labels = blocks[-1].dstdata['label']

            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach()
            train_acc = sklearn.metrics.accuracy_score(batch_labels.cpu().numpy(), 
                batch_pred.argmax(1).detach().cpu().numpy())*100

            gpu_mem_alloc = (
                torch.cuda.max_memory_allocated() / 1000000
                if torch.cuda.is_available()
                else 0
            )
        if epoch%args.log_every == 0:
            model.eval()
            predictions = []
            labels = []
            with torch.no_grad():
                for _, _, blocks in val_dataloader:
                    blocks = [block.to(device) for block in blocks]
                    inputs = blocks[0].srcdata['feat']
                    labels.append(blocks[-1].dstdata['label'].cpu().numpy())
                    predictions.append(model(blocks, inputs).argmax(1).cpu().numpy())
                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)
                val_acc = sklearn.metrics.accuracy_score(labels, predictions)*100
                if best_accuracy < val_acc:
                    best_accuracy = val_acc
                    if args.model_save:
                        torch.save(model.state_dict(), args.modelpath)

            tqdm.tqdm.write(
                "Epoch {:03d} | Loss {:.4f} | Train Acc {:.2f} | Val Acc {:.2f} | Time {} | GPU {:.1f} MB".format(
                    epoch,
                    epoch_loss,
                    train_acc,
                    val_acc,
                    str(datetime.timedelta(seconds = int(time.time() - epoch_start))),
                    gpu_mem_alloc
                )
            )

    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for _, _, blocks in test_dataloader:
            blocks = [block.to(device) for block in blocks]
            inputs = blocks[0].srcdata['feat']
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

    args = parser.parse_args()

    device = f'cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu'

    dataset = IGB260MDGLDataset(args)
    g = dataset[0]
    print(g)

    track_acc(g, args, device)




