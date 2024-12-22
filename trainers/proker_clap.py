import torch
import torch.nn.functional as F
from .clap import CLAP_Head

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


