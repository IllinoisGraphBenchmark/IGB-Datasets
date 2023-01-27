import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import torchmetrics.functional as MF
import time, tqdm, numpy as np
from models import *
from dataloader import IGBHeteroDGLDataset

torch.manual_seed(0)
dgl.seed(0)
import warnings
warnings.filterwarnings("ignore")


def evaluate(model, dataloader):
    predictions = []
    labels = []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [block.to(device) for block in blocks]
            inputs = blocks[0].srcdata['feat']
            labels.append(blocks[-1].dstdata['label']['paper'].cpu().numpy())
            predictions.append(model(blocks, inputs).argmax(1).cpu().numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        acc = sklearn.metrics.accuracy_score(labels, predictions)
        return acc


def track_acc(g, category, args, device):

    sampler = dgl.dataloading.NeighborSampler([int(fanout) for fanout in args.fan_out.split(',')],
            prefetch_node_feats={k: ['feat'] for k in g.ntypes},
            prefetch_labels={category: ['label']})

    train_nid = torch.nonzero(g.nodes[category].data['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.nodes[category].data['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.nodes[category].data['test_mask'], as_tuple=True)[0]

    train_dataloader = dgl.dataloading.DataLoader(
        g, {category: train_nid}, sampler,
        batch_size=args.batch_size,
        shuffle=True, drop_last=False,
        num_workers=args.num_workers)

    val_dataloader = dgl.dataloading.DataLoader(
        g, {category: val_nid}, sampler,
        batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        num_workers=args.num_workers)

    test_dataloader = dgl.dataloading.DataLoader(
        g, {category: test_nid}, sampler,
        batch_size=args.batch_size,
        shuffle=True, drop_last=False,
        num_workers=args.num_workers)

    in_feats = g.ndata['feat'][category].shape[1]

    if args.model_type == 'rgcn':
        model = RGCN(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers).to(device)
    if args.model_type == 'rsage':
        model = RSAGE(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers).to(device)
    if args.model_type == 'rgat':
        model = RGAT(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers, args.num_heads).to(device)

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
        lr=args.learning_rate)
    sched = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.25)

    best_accuracy = 0
    training_start = time.time()
    for epoch in tqdm.tqdm(range(args.epochs)):
        model.train()
        total_loss = 0
        train_acc = 0
        idx = 0
        gpu_mem_alloc = 0
        epoch_start = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            idx += 1
            blocks = [block.to(device) for block in blocks]
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']['paper']
            y_hat = model(blocks, x)
            loss = loss_fcn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_acc += sklearn.metrics.accuracy_score(y.cpu().numpy(),
                y_hat.argmax(1).detach().cpu().numpy())*100
            gpu_mem_alloc += (
                torch.cuda.max_memory_allocated() / 1000000
                if torch.cuda.is_available()
                else 0
            )
        train_acc /= idx
        gpu_mem_alloc /= idx

        if epoch%args.log_every == 0:
            model.eval()
            val_acc = evaluate(model, val_dataloader).item()*100
            if best_accuracy < val_acc:
                best_accuracy = val_acc
                if args.model_save:
                    torch.save(model.state_dict(), args.modelpath)

            tqdm.tqdm.write(
                "Epoch {:03d} | Loss {:.4f} | Train Acc {:.2f} | Val Acc {:.2f} | Time {} | GPU {:.1f} MB".format(
                    epoch,
                    total_loss,
                    train_acc,
                    val_acc,
                    str(datetime.timedelta(seconds = int(time.time() - epoch_start))),
                    gpu_mem_alloc
                )
            )
        sched.step()

    model.eval()
    test_acc = evaluate(model, test_dataloader).item()*100
    print("Test Acc {:.2f}%".format(test_acc))
    print("Total time taken " + str(datetime.timedelta(seconds = int(time.time() - training_start))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/root/gnndataset',
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
    parser.add_argument('--model_type', type=str, default='rgat',
                        choices=['rgat', 'rsage', 'rgcn'])
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

    dataset = IGBHeteroDGLDataset(args)
    g = dataset[0]
    category = g.predict

    track_acc(g, category, args, device)
