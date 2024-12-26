import torch
import torch.nn.functional as F
import os
import json 
from utils import validate
from .proker import RBF_Kernel
from .clap import CLAP_Head

def ProKeR_CLAP_joint(vecs, labels, val_features, val_labels, test_features, clip_weights, dataset, shots, seed, hp_selection, backbone='RN50'):
    """
        Jointly train ProKeR and CLAP
    """
    from trainers.proker_clap import ProKeR_CLAP_Head
    from trainers.clap import train_clap
    path = f'./configs/{backbone}/configs_proker_clap_joint/hyperparameters.json' # get hp training of CLAP
    assert os.path.exists(path), "Path required to get hyperparamters"
    trainCfg = json.load(open(path, 'r'))[str(shots)] 
    best_beta, best_lmbda = trainCfg['beta'], trainCfg['lmbda']
    device = vecs.device
    test_features = test_features.to(device)
    clip_weights = F.normalize(clip_weights, dim=0)

    logits_text_shots = torch.einsum('sd, cd -> sc', vecs.float(), clip_weights.float().T) # b,c
    K_XS = lambda x: RBF_Kernel(x, vecs[:], beta=best_beta) # b,s
    K_SS = RBF_Kernel(vecs[:], vecs[:], beta=best_beta) # s,s
    model = ProKeR_CLAP_Head(clip_weights.T, K_XS, K_SS, best_lmbda, vecs, labels).to(device)
    res = train_clap(model, vecs, labels, logits_text_shots, trainCfg=trainCfg)
    model.load_state_dict(res['state'])
    test_logits = validate(model, test_features, device=device)
    return test_logits.cpu()

class ProKeR_CLAP_Head(CLAP_Head): 
    def __init__(self, init, K_XS, K_SS, lmbda, vecs, labels):
        super().__init__(init)
        self.K_XS = K_XS
        self.D = K_SS.shape[0]
        self.K_SS = K_SS
        self.lmbda = lmbda
        n_classes = len(labels.unique())
        self.labels = F.one_hot(labels, num_classes = n_classes).float()
        self.vecs = vecs
    def forward(self, x):
        x = x / x.norm(dim=-1, keepdim=True)
        prototypes_norm = self.prototypes / self.prototypes.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_text_shots = (self.vecs @ prototypes_norm.t()).detach()
        alpha_i = torch.linalg.solve(1 / self.lmbda * self.K_SS + torch.eye(self.D).to(self.device), self.labels - logits_text_shots)
        logits = x @ prototypes_norm.t() + self.K_XS(x) @ alpha_i
        return logits * logit_scale


