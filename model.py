import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as utils
from torch_geometric.nn import GCNConv

import time

EPS = 1e-8
MAX_ALPHA = 1e6


def stabilize_alpha(alpha):
    return torch.nan_to_num(alpha, nan=1.0, posinf=MAX_ALPHA, neginf=EPS).clamp(EPS, MAX_ALPHA)

# loss function
def KL(alpha, beta):
    alpha = stabilize_alpha(alpha)
    beta = stabilize_alpha(beta.to(alpha.device))
    S_alpha = torch.sum(alpha, dim=1, keepdim=True).clamp_min(EPS)
    S_beta = torch.sum(beta, dim=1, keepdim=True).clamp_min(EPS)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    alpha = stabilize_alpha(alpha)
    S = torch.sum(alpha, dim=1, keepdim=True).clamp_min(EPS)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    beta = torch.ones((1, c), device=alpha.device)
    B = annealing_coef * KL(alp, beta)
    return (A + B)


def mse_loss(p, alpha, c, global_step, annealing_step=1):
    alpha = stabilize_alpha(alpha)
    S = torch.sum(alpha, dim=1, keepdim=True).clamp_min(EPS)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    beta = torch.ones((1, c), device=alpha.device)
    C = annealing_coef * KL(alp, beta)
    return (A + B) + C

def js_distance(p, q):
    m = 0.5 * (p + q)
    # return 0.5 * (F.kl_div(torch.log(p), m) + F.kl_div(torch.log(q), m))
    return 0.5 * (torch.mean(F.kl_div(torch.log(p+1e-5), m, reduction='none'), dim=1) + torch.mean(F.kl_div(torch.log(q+1e-5), m, reduction='none'), dim=1))

class TMC(nn.Module):

    def __init__(self, args):
        """
        :param classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(TMC, self).__init__()
        self.device = args.device
        self.views = args.views
        self.lambda_epochs = args.lambda_epochs
        self.classes = args.classes
        self.classifiers = nn.ModuleList([GCN(args).to(args.device) for i in range(self.views)])
        self.fusion = FusionBlock(args)

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = stabilize_alpha(alpha1), stabilize_alpha(alpha2)
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True).clamp_min(EPS)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.classes/S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            conflict = (1-C).clamp_min(EPS)
            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/(conflict.view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/(conflict.view(-1, 1).expand(u[0].shape))
            u_a = u_a.clamp_min(EPS)

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a
        
        dist = []
        for v in range(self.views):
            alpha_each_expert = stabilize_alpha(alpha[v])
            S = torch.sum(alpha_each_expert, dim=1, keepdim=True).clamp_min(EPS)
            E = alpha_each_expert-1
            b = E/(S.expand(E.shape))
            u = self.classes/S
            p = torch.cat([b, u], dim=1)
            dist.append(p)

        D = torch.zeros(len(alpha[0]), self.views, self.views).to(self.device)
        for i in range(self.views):
            for j in range(self.views):
                p, q = dist[i], dist[j]
                D[:,i,j] = js_distance(p, q)+1e-5
        # D = torch.exp(D)

        # Entropy       
        c = dict()
        M_bar = torch.zeros(len(alpha[0]), self.classes+1).to(self.device)
        row_sum = 1 / torch.sum(D, dim=-1).clamp_min(EPS) # self.views-1 include all views
        for v in range(self.views):
            p = dist[v]
            entropy = -torch.mean(p * torch.log(p.clamp_min(EPS)), dim=1) # each view
            credibility = (entropy * row_sum[:,v]).clamp_min(EPS) # torch.exp(entropy) * row_sum[:,v]
            c[v] = credibility.to(self.device)

        sum_c = torch.zeros(len(c[0]), 1).to(self.device)
        for v in range(self.views):
            c[v] = c[v].reshape(-1,1)
            sum_c += c[v]

        for v in range(self.views):
            c[v] = c[v]/sum_c.clamp_min(EPS)
            M_bar += torch.mul(c[v].expand(dist[v].shape), dist[v])

        uncertainty = M_bar[:, -1].clamp_min(EPS)
        
        S = self.classes / uncertainty # M_bar[:, -1] uncertainty
        E = M_bar[:, :-1] * (S.reshape(-1, 1).expand(M_bar[:, :-1].shape))
        alpha_new = dict()
        for v in range(self.views):
            alpha_new[v] = stabilize_alpha(E + 1)

        for v in range(len(alpha_new)-1):            
            if v==0:
                alpha_a = DS_Combin_two(alpha_new[0], alpha_new[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha_new[v+1])
        return alpha_a, uncertainty
    
    def infer(self, args, input, mask):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        rep = dict() 
        evidence = dict() 
        rep_all = dict() 
        evidence_all = dict() 
        for v_num in range(self.views):
            rep_all[v_num], evidence_all[v_num] = self.classifiers[v_num](input[v_num])
            rep[v_num], evidence[v_num] = rep_all[v_num][mask], evidence_all[v_num][mask]
        return rep, evidence, rep_all, evidence_all

    def forward(self, args, X, y, mask, global_step, adj_list, targets_u, pseudo_mask
                , idx_clean, idx_noisy, epoch, labels):
        rep, evidence, rep_all, evidence_all = self.infer(args, X, mask)
        rep_a_evidence = self.fusion(rep_all)
        rep_a_alpha = rep_a_evidence + 1
        loss = 0
        alpha = dict()
        alpha_all = dict()
        for v_num in range(len(X)):
            alpha[v_num] = stabilize_alpha(evidence[v_num] + 1)
            alpha_all[v_num] = stabilize_alpha(evidence_all[v_num] + 1)
            # risk
            loss += ce_loss(y, alpha[v_num], args.classes, global_step, self.lambda_epochs)
            # cross entropy
            # loss += F.cross_entropy(evidence[v_num], y, reduction='mean')
        if len(X) ==1: 
            alpha_a = alpha[0]
            alpha_a_all = alpha_all[0]
        else:
            alpha_a, uncertaity_a = self.DS_Combin(alpha)
            alpha_a_all, uncertaity_a_all = self.DS_Combin(alpha_all)

        uncertaity_a_clean = uncertaity_a_all[idx_clean]
        uncertaity_a_noisy = uncertaity_a_all[idx_noisy]

        print('Epoch: {:04d}'.format(epoch + 1),
                'Clean nodes energy: {:.4f}'.format(torch.mean(uncertaity_a_clean).item()),
                'Noisy nodes energy: {:.4f}'.format(torch.mean(uncertaity_a_noisy).item())
                )

        evidence_a = alpha_a - 1
        evidence_a_all = alpha_a_all - 1
        # risk
        loss += args.weight_all * ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epochs)
        # cross entropy
        # loss += args.weight_all * F.cross_entropy(evidence_a, y, reduction='mean')

        loss_consist = args.weight_con * KL(rep_a_alpha, alpha_a_all) # 在所有样本（标记+无标记）表示和决策的一致性

        if targets_u != None:
            # cross entropy
            loss_pseudo = args.weight_pse * (F.cross_entropy(evidence_a_all, targets_u, reduction='none') * pseudo_mask).mean()
            # risk
            # loss_pseudo = args.weight_pse * (ce_loss(targets_u, alpha_a_all, self.classes, global_step, self.lambda_epochs) * pseudo_mask).mean()
        else:
            loss_pseudo = 0

        loss = torch.mean(loss) + torch.mean(loss_consist) + loss_pseudo
        return evidence, evidence_a, loss, evidence_a_all


class GCN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = GCNConv(args.num_features, args.hidden1)
        self.conv2 = GCNConv(args.hidden1, args.hidden)
        # self.conv3 = GCNConv(args.hidden, args.classes)
        self.fc = nn.Linear(args.hidden, args.classes)
        self.bn1 = torch.nn.BatchNorm1d(args.hidden1)
        self.bn2 = torch.nn.BatchNorm1d(args.hidden)
        self.act = nn.Softplus() # Softplus, ReLU
        # self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)# + x
        x = self.bn1(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        z = F.relu(x)# + x
        # x = self.bn2(x)
        x = self.fc(z)
        logits = self.act(x)
        return z, logits


class FusionBlock(nn.Module):
    expand = 2
    def __init__(self, args, act_func='relu', dropout=0., norm_eps=1e-5):
        super().__init__()
        input_dim = args.hidden * args.views
        hidden_dim1 = input_dim * self.expand
        hidden_dim2 = input_dim // self.expand
        if act_func == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_func == 'tanh':
            self.act = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError('Activate function must be ReLU or Tanh.')
        self.linear1 = nn.Linear(input_dim, hidden_dim1, bias=False)
        self.linear2 = nn.Linear(hidden_dim1, input_dim, bias=False)
        
        self.linear3 = nn.Linear(input_dim, hidden_dim2, bias=False)
        self.linear4 = nn.Linear(hidden_dim2, input_dim, bias=False)

        self.norm1 = nn.BatchNorm1d(input_dim, eps=norm_eps)
        self.norm2 = nn.BatchNorm1d(input_dim, eps=norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.fc = nn.Linear(input_dim, args.classes)
        self.act = nn.Softplus()
        
    def forward(self, rep):
        x = []
        for i in range(len(rep)):
            x.append(rep[i])
        x = torch.concat(x, dim=-1)
        x = x + self.block1(x)
        x = x + self.block2(x)
        # x = x + self.block1(self.norm1(x))
        # x = x + self.block2(self.norm2(x))
        x = self.fc(x)
        logits = self.act(x)
        return logits
        
    def block1(self, x):
        return self.linear2(self.dropout1(self.act(self.linear1(x))))
    
    def block2(self, x):
        return self.linear4(self.dropout2(self.act(self.linear3(x))))


# def r_loss(AZ, Z):
#     """
#     the loss of propagated regularization (L_R)
#     Args:
#         AZ: the propagated embedding
#         Z: embedding
#     Returns: L_R
#     """
#     p_output = F.softmax(AZ, dim=1)
#     q_output = F.softmax(Z, dim=1)
#     log_mean_output = ((p_output + q_output) / 2).log()
#     loss = (F.kl_div(log_mean_output, p_output, reduction='mean') +
#                 F.kl_div(log_mean_output, p_output, reduction='mean')) / 2
#     return loss