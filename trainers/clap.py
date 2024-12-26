import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .train_utils import build_optimizer, build_lr_scheduler
from copy import deepcopy
from tqdm import trange 
from utils import TensorDataset
import json 
from utils import validate
def CLAP(vecs, labels, val_features, val_labels, test_features, clip_weights, dataset, shots, seed, hp_selection, backbone='RN50'):
    """
        CLAP method "A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models" CVPR 2024.
    """
    path = f'./configs/{backbone}/configs_clap/hyperparameters.json'
    trainCfg = json.load(open(path, 'r'))[str(shots)] if os.path.exists(path) else None
    clip_weights = F.normalize(clip_weights, dim=0)
    device = vecs.device
    logits_zs = torch.einsum('sd, cd -> sc', vecs.float(), clip_weights.float().T) # b,c
    model = CLAP_Head(clip_weights.T).to(device)
    res = train_clap(model, vecs, labels, logits_zs, trainCfg=trainCfg)
    model = CLAP_Head(clip_weights.T).to(device)     
    model.load_state_dict(res['state'])
    test_logits = validate(model, test_features, device=device)
    return test_logits

class CLAP_Head(nn.Module): 
    def __init__(self, init): 
        super().__init__()
        self.logit_scale = torch.FloatTensor([4.60517]).to(init.device)
        self.prototypes = nn.Parameter(init.clone())
        self.apply_constraint = "l2"
        self.device = init.device
        self.base_text_features = init.clone()

    def forward(self, x):
        x = x / x.norm(dim=-1, keepdim=True)
        prototypes_norm = self.prototypes / self.prototypes.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = x @ prototypes_norm.t() * logit_scale
        return logits
    
    def init_lagrangian_multipliers(self, labels_one_hot, logits_ds):
        if "balanced" in self.apply_constraint:
            performance = torch.ones(logits_ds.shape[-1]).to(torch.float)
        else:
            with torch.no_grad():
                # Get zero_shot performance
                performance = torch.diag(torch.softmax(logits_ds, -1).t() @ labels_one_hot.to(torch.float32)) /\
                                      labels_one_hot.sum(0)

                if "corrected" in self.apply_constraint:
                    performance *= (logits_ds.shape[-1] / torch.sum(performance).item())
                if "constant" in self.apply_constraint:
                    performance = torch.ones(logits_ds.shape[-1]).to(torch.float) * torch.mean(performance).item()
        # set new alphas
        self.alpha_constraint = torch.clone(performance).to(self.device)
        self.penalty_parameter = torch.zeros_like(self.alpha_constraint).to(self.device)
    
    def outer_step(self):
        def phr(h, lambd, rho):
            x = lambd + rho * h
            y_sup = 1 / (2 * rho) * (x ** 2 - lambd ** 2)
            y_inf = - 1 / (2 * rho) * (lambd ** 2)

            grad_y_sup = x
            grad_y_inf = torch.zeros_like(h)

            sup = x >= 0
            return (
                torch.where(sup, y_sup, y_inf),
                torch.where(sup, grad_y_sup, grad_y_inf)
            )

        print("Outer step on Augmented Lagrangian Multiplier")

        # Cmpute current constraints
        disimilitude = (self.prototypes - self.base_text_features.clone()).pow(2).sum(-1)

        # Compute phr
        _, phr_grad = phr(disimilitude, self.alpha_constraint, self.penalty_parameter)

        # Update lagrangian multipliers
        self.alpha_constraint = phr_grad.detach().clone()

        # Update penalty parameters rho
        self.penalty_parameter = disimilitude.detach().clone()

        print("New lagrangian multipliers:")
        print(self.alpha_constraint[0:5].detach().cpu().numpy())

    def zero_shot_constraint(self):
        # Compute constraint
        if "l2" in self.apply_constraint:
            disimilitude = (self.prototypes - self.base_text_features.clone()).pow(2).sum(-1)
        elif "cosine" in self.apply_constraint:
            disimilitude = (1 - torch.nn.functional.cosine_similarity(self.prototypes, self.base_text_features.clone()))
        else:
            print("Dissimilitude metric for constraint not implemented")
            assert False
        return torch.mean(self.alpha_constraint * disimilitude)

def train_clap(model, shots, labels, logits_zs, trainCfg=None): 
    """
        Training routine of clap
    """
    if trainCfg is None: # default values 
        trainCfg = {'iters':2000, 'lr':0.0001, 'optimizer':'adamw', 'batch_size':64, 'weight_decay': 0.01, 'scale':1, 'lr_scheduler':'cosine', 'warmup_type':'linear', 'warmup_min_lr': 1e-5, 'warmup_iter':50, 'eval_freq':20}
    criterion = nn.CrossEntropyLoss()
    device = shots.device
    optimizer = build_optimizer([{'params':model.parameters()}], trainCfg['optimizer'], lr=trainCfg['lr'], weight_decay=trainCfg['weight_decay'])
    scheduler = build_lr_scheduler(optimizer,trainCfg['lr_scheduler'],trainCfg['warmup_iter'],trainCfg['iters'],warmup_type=trainCfg['warmup_type'],warmup_lr=trainCfg['warmup_min_lr'])
    
    model.init_lagrangian_multipliers(labels, logits_zs)
    best = {'state':model.state_dict()}
    pbar = trange(trainCfg['iters'], leave=True)
    loader = DataLoader(
                TensorDataset(shots.cpu(), labels.cpu()),
                batch_size=trainCfg['batch_size'],
                shuffle=True,
                num_workers=min(os.cpu_count(), 1),
                pin_memory=True,
                drop_last=True,
            )
    device = shots.device
    loader_iter = iter(loader)
    for _ in pbar: 
        optimizer.zero_grad()
        try:
            shots_batch, labels_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            shots_batch, labels_batch = next(loader_iter)
        shots_batch = shots_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        logits = model(shots_batch)
        loss_constraint = model.zero_shot_constraint()
        logits *= trainCfg['scale']
        loss = criterion(logits, labels_batch) + loss_constraint
        loss.backward()
        optimizer.step()
        scheduler.step()
        best['state'] = deepcopy(model.state_dict())
        pbar.set_postfix({'Loss':f'{loss.item():.3f}'})
    return best 
