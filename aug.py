import copy
import random
import numpy as np

import torch
from functional import drop_feature, drop_edge_weighted, degree_drop_weights, evc_drop_weights, pr_drop_weights, \
                        feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense, compute_pr, eigenvector_centrality

from torch_geometric.utils import dropout_adj, degree, to_undirected

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_aug(args, data):
    if args.drop_scheme == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(args.device)
    elif args.drop_scheme == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(args.device)
    elif args.drop_scheme == 'evc':
        drop_weights = evc_drop_weights(data).to(args.device)
    else:
        drop_weights = None

    if args.drop_scheme == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(args.device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(args.device)
    elif args.drop_scheme == 'pr':
        node_pr = compute_pr(data.edge_index)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(args.device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(args.device)
    elif args.drop_scheme == 'evc':
        node_evc = eigenvector_centrality(data)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(args.device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(args.device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(args.device)

    def drop_edge(drop_feature_rate, drop_weights):

        if args.drop_scheme == 'uniform':
            return dropout_adj(data.edge_index, p=drop_feature_rate)[0]
        elif args.drop_scheme in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(data.edge_index, drop_weights, p=drop_feature_rate, threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {args.drop_scheme}')

    edge_index_1 = drop_edge(args.drop_feature_rate_1, drop_weights)
    edge_index_2 = drop_edge(args.drop_feature_rate_2, drop_weights)
    x_1 = drop_feature(data.x, args.drop_feature_rate_1)
    x_2 = drop_feature(data.x, args.drop_feature_rate_2)

    if args.drop_scheme in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, args.drop_feature_rate_1)
        x_2 = drop_feature_weighted_2(data.x, feature_weights, args.drop_feature_rate_2)

    data_aug1 = copy.deepcopy(data)
    data_aug2 = copy.deepcopy(data)
    data_aug1.x = x_1
    data_aug2.x = x_2
    data_aug1.edge_index = edge_index_1
    data_aug2.edge_index = edge_index_2
    return data_aug1, data_aug2

def permute_edges(data, aug_ratio):
    edge_num = data.num_edges
    permute_num = int(edge_num * aug_ratio)
    edge_index = data.edge_index.transpose(0, 1)
    edge_index = edge_index[torch.LongTensor(np.random.choice(edge_num, edge_num-permute_num, replace=False))]
    data.edge_index = edge_index.transpose_(0, 1)
    return data

def mask_nodes(data, aug_ratio):
    num = data.num_nodes
    features = data.x
    drop_rates = torch.FloatTensor(np.ones(num) * aug_ratio)
    masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
    features = masks.to(device) * features  
    data.x = features
    return data