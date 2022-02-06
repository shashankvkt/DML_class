import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning.distances import CosineSimilarity


distance = CosineSimilarity()


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class ProxyAnchorLoss_mixup(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies_mixup = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies_mixup, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
      
        
    def forward(self, embedding, pos_labels, neg_labels, lam):
        P = self.proxies_mixup

        # s(x,p) - similarity between input embedding and proxy
        sim_mat = distance(embedding, P)#F.linear(l2_norm(embedding), l2_norm(P)) 

        P_one_hot = binarize(T = pos_labels, nb_classes = self.nb_classes)
        N_one_hot = (1-P_one_hot)#binarize(T = neg_labels, nb_classes = self.nb_classes)
        
        P_one_hot = lam*P_one_hot
        N_one_hot = (1-lam)*N_one_hot

        pos_exp = lam*torch.exp(-self.alpha * distance.margin(sim_mat,self.mrg))
        neg_exp = (1-lam)*torch.exp(self.alpha * (self.mrg + sim_mat))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == lam, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == (1-lam), neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     

        return loss
       