import time
import copy
import random
import numpy as np
import networkx as nx

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import Data
import torch_geometric.utils as utils
import scipy.sparse as sp

from arguments import arg_parse
from datasets import *
from model import TMC
from dataset_process import Dataset
from utils_process import sparse_mx_to_torch_sparse_tensor

torch.cuda.set_device(0)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def edge_index_to_tensor(data):
    adj = utils.to_scipy_sparse_matrix(data.edge_index)
    adj = torch.Tensor(adj.toarray())
    # values = adj.data
    # indices = np.vstack((adj.row, adj.col))
    # i = torch.LongTensor(indices)
    # v = torch.FloatTensor(values)
    # shape = adj.shape
    # adj = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
    return adj

def main():

    args = arg_parse()


    if args.seed is not None:
        set_seed(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    
    if args.dataset == "cora" or args.dataset == "citeseer" or args.dataset == "pubmed":
        data = Dataset(root='./data', name=args.dataset)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        idx_train = idx_train[:int(args.label_rate * adj.shape[0])]
    elif args.dataset == "cs" or args.dataset == "physics":
        dataset = get_coauthor_dataset(args.dataset, args.normalize_features)
    elif args.dataset == "computers" or args.dataset == "photo":
        dataset = get_amazon_dataset(args.dataset, args.normalize_features)
    elif args.dataset=='dblp':
        from torch_geometric.datasets import CitationFull
        dataset = CitationFull('./data','dblp')

    if args.dataset == "dblp" or args.dataset == "cs" or args.dataset == "computers" or args.dataset == "photo":
        adj = utils.to_scipy_sparse_matrix(dataset.data.edge_index)
        features = dataset.data.x.numpy()
        labels = dataset.data.y.numpy()
        idx = np.arange(len(labels))
        np.random.shuffle(idx)
        idx_test = idx[:int(0.8 * len(labels))]
        idx_val = idx[int(0.8 * len(labels)):int(0.9 * len(labels))]
        idx_train = idx[int(0.9 * len(labels)):int((0.9+args.label_rate) * len(labels))]
    print(f"Dataset: {args.dataset}")
    
    from utils_process import noisify_with_P
    ptb = args.ptb_rate
    nclass = labels.max() + 1
    train_labels = labels[idx_train]
    noise_y, P = noisify_with_P(train_labels, nclass, ptb, 10, args.noise) 
    noise_labels = labels.copy()
    noise_labels[idx_train] = noise_y

    # Get indices of noisy labels
    idx_noisy = np.where(labels != noise_labels)[0]
    idx_noisy = np.intersect1d(idx_noisy, idx_train)
    # Get indices of clean labels
    idx_clean = np.where(labels == noise_labels)[0]
    idx_clean = np.intersect1d(idx_clean, idx_train)


    edge_index, _ = utils.from_scipy_sparse_matrix(adj)
    if sp.issparse(features):
        features = sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
    else:
        features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(np.array(labels))

    data = Data(x=features, edge_index=edge_index, y=torch.LongTensor(noise_labels)).to(args.device)
    idx_train, idx_val, idx_test = torch.LongTensor(idx_train).to(args.device), torch.LongTensor(idx_val).to(args.device), torch.LongTensor(idx_test).to(args.device)

    args.num_features = max(features.shape[1], 1)
    args.classes = labels.max().item() + 1

    model = TMC(args).to(args.device)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True) # Adam, RMSprop
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train(args, model, data, idx_train, idx_val, idx_test, optimizer, idx_clean, idx_noisy)


def train(args, model, data, idx_train, idx_val, idx_test, optimizer, idx_clean, idx_noisy):
    loss_values = []
    acc_values = []
    bad_counter = 0
    loss_best = np.inf
    acc_best = 0.0

    loss_mn = np.inf
    acc_mx = 0.0

    data_input = dict()
    for v_num in range(args.views):
        data_input[v_num] = data
    train_mask = idx_train
    target = data.y[train_mask]
    labels = data.y

    adj_list = []
    for v_num in range(len(data_input)):
        adj_list.append(edge_index_to_tensor(data_input[v_num]).to(args.device))

    best_epoch = 0
    targets_u = None
    pseudo_mask = None
    acc_test_list = []
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()

        # data_aug1 = copy.deepcopy(data)
        # data_aug2 = copy.deepcopy(data)
        # data_aug3 = copy.deepcopy(data)
        # data_aug1 = mask_nodes(data_aug1, args.mask_aug_ratio)
        # data_aug2 = permute_edges(data_aug2, args.edge_aug_ratio)
        # data_aug3 = permute_edges(data_aug2, args.edge_aug_ratio)
        # data_input = {0:data_aug1, 1:data_aug2, 2:data_aug3}

        evidences, evidence_a, loss_train, evidence_a_all = model(args, data_input, target, train_mask, epoch, adj_list, targets_u, pseudo_mask
                                                                  , idx_clean, idx_noisy, epoch, labels)

        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        acc_train = accuracy(evidence_a, target)

        pseudo_label = torch.softmax(evidence_a_all.detach()/args.T, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        pseudo_mask = max_probs.ge(args.threshold).float()

        # Val/Test
        model.eval()
        val_mask = idx_val
        val_target = data.y[val_mask]
        _, output_val, loss_val, _ = model(args, data_input, val_target, val_mask, epoch, adj_list, targets_u, pseudo_mask
                                           , idx_clean, idx_noisy, epoch, labels)  
        acc_val = accuracy(output_val, val_target)

        test_mask = idx_test
        test_target = data.y[test_mask]
        _, output_test, _, _ = model(args, data_input, test_target, test_mask, epoch, adj_list, targets_u, pseudo_mask
                                     , idx_clean, idx_noisy, epoch, labels)
        acc_test = accuracy(output_test, test_target)
        acc_test_list.append(acc_test.item())

        print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'loss_val: {:.4f}'.format(loss_val.item()),
        'acc_val: {:.4f}'.format(acc_val.item()),
        'acc_test: {:.4f}'.format(acc_test.item()),
        'time: {:.4f}s'.format(time.time() - t))

        loss_values.append(loss_val.item())
        acc_values.append(acc_val.item())

        print(bad_counter)

        if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:# or epoch < 400:
            # if loss_values[-1] <= loss_best:
            if acc_values[-1] >= acc_best:
            # if loss_values[-1] <= loss_best and acc_values[-1] >= acc_best:
                loss_best = loss_values[-1]
                acc_best = acc_values[-1]
                best_epoch = epoch
                torch.save(model.state_dict(), './save_models/' + args.dataset + '.pkl')

            loss_mn = np.min((loss_values[-1], loss_mn))
            acc_mx = np.max((acc_values[-1], acc_mx))
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            print('Early stop! Min loss: ', loss_mn, ', Max accuracy: ', acc_mx)
            print('Early stop model validation loss: ', loss_best, ', accuracy: ', acc_best)
            print('Early stop last validation loss: ', loss_values[-1], ', accuracy: ', acc_values[-1])
            break

    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('./save_models/' + args.dataset +'.pkl'))
    acc_test = test(args, data_input, model, test_mask, adj_list, targets_u, pseudo_mask
                    , idx_clean, idx_noisy, epoch, labels)



def test(args, data, model, mask, adj_list, targets_u, pseudo_mask
         , idx_clean, idx_noisy, epoch, labels):
    with torch.no_grad():
        model.eval()
        target = data[0].y[mask]
        _, output, _, _ = model(args, data, target, mask, 0, adj_list, targets_u, pseudo_mask
                                , idx_clean, idx_noisy, epoch, labels)
        # tsne(args, data, model, idx_test)
        # output, _ = model(data.x, data.edge_index)
        acc_test = accuracy(output, target)
        print("Test set results:",
            # "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


def tsne(args, data, model, idx_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    clf = LogisticRegression(random_state=0, max_iter=2000)
    with torch.no_grad():
        embeds = model(args, data)
        test_embs = embeds[idx_test]
        test_labels = data.y[idx_test]
    pred_test_labels = embeds.max(1)[1][idx_test].type_as(test_labels)

    tsne_test_input = test_embs.cpu().numpy()
    tsne_test_label = test_labels.cpu().numpy()
    tsne_test_pre_labels = pred_test_labels.cpu().numpy()
    np.save("visualization/tsne_test_embs.npy", tsne_test_input)
    np.save("visualization/tsne_test_tru_label.npy", tsne_test_label)
    np.save("visualization/tsne_test_pre_label.npy", tsne_test_pre_labels)
    return accuracy_score(test_labels.cpu().numpy(), pred_test_labels.cpu().numpy())

if __name__ == '__main__':
    main()


