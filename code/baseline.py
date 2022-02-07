import sys
import torch, math, time, argparse, os
import random
import numpy as np

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import dataset, net

from net.resnet import *
# from baseline_utils import utils
from torch import nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate

from pytorch_metric_learning.losses.contrastive_loss import ContrastiveLoss
from pytorch_metric_learning.distances.lp_distance import LpDistance
from pytorch_metric_learning.reducers.avg_non_zero_reducer import AvgNonZeroReducer
from pytorch_metric_learning.reducers.multiple_reducers import MultipleReducers
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from tqdm import *
import random

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datasets = 'cub'  #change this to 'cars' to perform experiment on CARS-196

data_root = root_folder + '/data'
saveDir = root_folder + '/checkpoint/' + datasets + '/'

if not os.path.exists(saveDir):
    os.makedirs(saveDir)



models = 'resnet50'  # network
sz_batch = 100       # batch size
nb_workers = 4       # number of workers for dataloader (dont change)
sz_embedding = 512   # embedding dimension
bn_freeze = 1        # freeze batch normalization
l2_norm = 1          # l2 normalize embeddings
nb_epochs = 60       # total training epochs
gpu_id = 0           # you will use one gpu, this is the ID of the gpu
warm = 5             # number of epochs of warmup
lr_decay_step = 5    # decay learning rate after 5 epochs
lr_decay_gamma = 0.1 # learning rate decay factor


# load your training set
trn_dataset = dataset.load(
            name = datasets,
            root = data_root,
            mode = 'train',
            transform = dataset.utils.make_transform(
                is_train = True, 
                is_inception = (models == 'bn_inception')
            ))


dl_tr = torch.utils.data.DataLoader(
    trn_dataset,
    batch_size = sz_batch,
    shuffle = True,
    num_workers = nb_workers,
    drop_last = True,
    pin_memory = True
)
print('Random Sampling')


# load your testing set
ev_dataset = dataset.load(
            name = datasets,
            root = data_root,
            mode = 'eval',
            transform = dataset.utils.make_transform(
                is_train = False, 
                is_inception = (models == 'bn_inception')
            ))

dl_ev = torch.utils.data.DataLoader(
    ev_dataset,
    batch_size = sz_batch,
    shuffle = False,
    num_workers = nb_workers,
    pin_memory = True
)


nb_classes = trn_dataset.nb_classes()
print(nb_classes)


#initialize your model

model = Resnet50(embedding_size=sz_embedding, pretrained=True, is_norm=l2_norm, bn_freeze = bn_freeze)

'''
load model to GPU

for initial part of your experiment, comment out Line 112 and only use it for the final run

'''
model.to(device) 


def margin(x, y):
    return x - y


lr = 1e-4
weight_decay = 1e-4
margin_value = 0.5
criterion = ContrastiveLoss(neg_margin=0.5)
distance = LpDistance()
reducer_dict = {"pos_loss" : AvgNonZeroReducer(), "neg_loss" : AvgNonZeroReducer()}
reducer = MultipleReducers(reducer_dict)


param_groups = [
    {'params': list(set(model.parameters()).difference(set(model.model.embedding.parameters()))) if gpu_id != -1 else 
                 list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
    {'params': model.model.embedding.parameters() if gpu_id != -1 else model.module.model.embedding.parameters(), 'lr':float(lr) * 1},
]
# exit()
opt = torch.optim.Adam(param_groups, lr=float(lr), weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_decay_step, gamma = lr_decay_gamma)


losses_list = []
best_recall=[0]
best_epoch = 0

def margin(x, y):
    return x - y

for epoch in range(0, nb_epochs):
    model.train()
    bn_freeze = bn_freeze
    if bn_freeze:
        modules = model.model.modules() if gpu_id != -1 else model.module.model.modules()
        for m in modules: 
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    losses_per_epoch = []
    
    # Warmup: Train only new params, helps stabilize learning.
    if warm > 0:
        if gpu_id != -1:
            unfreeze_model_param = list(model.model.embedding.parameters()) + list(criterion.parameters())
        else:
            unfreeze_model_param = list(model.module.model.embedding.parameters()) + list(criterion.parameters())

        if epoch == 0:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = False
        if epoch == warm:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = True


    pbar = tqdm(enumerate(dl_tr))

    for batch_idx, (inputs, target) in pbar: 

        inputs = inputs.cuda()
        target = target.cuda()  

        #get your pairs
        a1, p, a2, n = lmu.get_all_pairs_indices(target)

        # pass inputs into the model
        out = model(inputs)

        # compute similarity with respect to embeddings
        embedding_similarity = distance(out)
        
        
        # Get similarity of positive and negative pairs
      
        pos_pair_dist, neg_pair_dist = [], []
        if len(a1) > 0:
            pos_pair_dist = embedding_similarity[a1, p]
        if len(a2) > 0:
            neg_pair_dist = embedding_similarity[a2, n]

        indices_tuple = (a1, p, a2, n)
        pos_pairs = lmu.pos_pairs_from_tuple(indices_tuple)
        neg_pairs = lmu.neg_pairs_from_tuple(indices_tuple)

        
        # compute loss using margin value

        if len(pos_pair_dist) > 0:
            pos_loss = torch.nn.functional.relu(margin(pos_pair_dist, 0.0))
        if len(neg_pair_dist) > 0:
            neg_loss = torch.nn.functional.relu(margin(margin_value, neg_pair_dist))


        loss_dict =  {
            "pos_loss": {
                "losses": pos_loss,
                "indices": pos_pairs,
                "reduction_type": "pos_pair",
            },
            "neg_loss": {
                "losses": neg_loss,
                "indices": neg_pairs,
                "reduction_type": "neg_pair",
            },
        }


        loss = reducer(loss_dict, embedding_similarity, target)

        # set gradients to 0
        
        opt.zero_grad()

        # compute gradients
        loss.backward()

        # clip gradients for stability
        
        torch.nn.utils.clip_grad_value_(model.parameters(), 10)

        losses_per_epoch.append(loss.data.cpu().numpy())

        # update weights
        opt.step()

        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dl_tr),
                100. * batch_idx / len(dl_tr),
                loss.item()))
        

    mean_loss = np.mean(losses_per_epoch)
    print('------> epoch: {} -->  loss = {:.4f} '.format(epoch,mean_loss))
    scheduler.step()
    
    if(epoch >= 0):
        with torch.no_grad():
            print("**Evaluating...**")
            if datasets == 'Inshop':
                Recalls = utils.evaluate_cos_Inshop(model, dl_query, dl_gallery)
            elif datasets != 'SOP':
                Recalls = utils.evaluate_cos(model, dl_ev)
            else:
                Recalls = utils.evaluate_cos_SOP(model, dl_ev)
                
        
        # Best model save
        if best_recall[0] < Recalls[0]:
            
            print('Saving..')
            best_recall = Recalls
            best_epoch = epoch
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            state = {
                'model': model,
                'Recall': Recalls,
                'epoch': epoch,
                'rng_state': torch.get_rng_state()
                }
       
            torch.save(state, saveDir + 'baseline_R50_d512.t7')


print(best_recall)

            

'''
CUB

d = 512

R@1 : 64.711   
R@2 : 75.454
R@4 : 83.291
R@8 : 89.664
R@16 : 93.169
R@32 : 95.945

'''
