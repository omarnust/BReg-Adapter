import torch 
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import json
from utils import save_hps, cls_acc

def RBF_Kernel(X,Y, beta):
    return (-beta*(1-X.float()@Y.float().T)).exp()  

def ProKeR(vecs, labels, val_features, val_labels, test_features, clip_weights, dataset, shots, seed, hp_selection, backbone='RN50'):
    """
        Training-Free version of ProKeR
    """
    n_classes = len(labels.unique())
    cache = F.one_hot(labels, num_classes = n_classes).float()
    device = vecs.device
    logits_text_shots = torch.einsum('sd, cd -> sc', vecs.float(), clip_weights.float().T) # b,c

    if 'imagenet' in dataset: # if imagenet-r/a/v2/sketch, use cfg of imagenet
        dataset = 'imagenet' # use hps of imagenet

    if hp_selection != 'imagenet': # grid search for hyper-parameters
        cfg = yaml.load(open(f'./configs/{backbone}/configs_proker/{dataset}.yaml', 'r'), Loader=yaml.Loader)
        best_val_acc = 0
        val_features = val_features.to(device)
        val_labels = val_labels.to(device)
        logits_text_val = torch.einsum('bd, cd -> bc', val_features.float(), clip_weights.float().T) # b,c

        beta_search = torch.linspace(float(cfg['beta'][0]), float(cfg['beta'][1]), cfg['beta'][2])
        lmbda_search = torch.linspace(float(cfg['lmbda'][0]), float(cfg['lmbda'][1]), cfg['lmbda'][2])
        for beta in tqdm(beta_search):
            K_SS = RBF_Kernel(vecs[:], vecs[:], beta=beta) # s,s
            K_XS = RBF_Kernel(val_features, vecs[:], beta=beta) # b,s
            for lmbda in lmbda_search:
                alpha_i = torch.linalg.solve(1/lmbda * K_SS + torch.eye(vecs.size(0)).to(device), cache - logits_text_shots)
                val_logits = logits_text_val + K_XS @ alpha_i
                acc = cls_acc(val_logits, val_labels)
                if acc > best_val_acc:
                    best_val_acc, best_lmbda, best_beta = acc, lmbda, beta
        best_hp = {'lmbda': best_lmbda.item(), 'beta': best_beta.item()}
        save_hps(best_hp, f'./configs/{backbone}/configs_proker/{dataset}', seed) # save best hp
        print(f'best: beta={best_beta}, lmbda={best_lmbda}, val acc={best_val_acc:.2f}%')
    else: # load best hp from imagenet 
        best_hp = json.load(open(f'./configs/{backbone}/configs_proker/hyperparameters.json', 'r'))[str(shots)] # Pre-computed hps and averaged per seed
        best_beta, best_lmbda = best_hp['beta'], best_hp['lmbda']

    # solve linear system with best hp
    K_SS = RBF_Kernel(vecs[:], vecs[:], beta=best_beta)
    alpha_i = torch.linalg.solve(1 / best_lmbda * K_SS + torch.eye(vecs.size(0)).to(device), cache - logits_text_shots)
    
    test_features = test_features.to(device)
    K_XS = RBF_Kernel(test_features, vecs[:], beta=best_beta) # b,s
    logits_text_test = torch.einsum('bd, cd -> bc', test_features.float(), clip_weights.float().T) # b,c
    test_logits = logits_text_test + K_XS @ alpha_i
    return test_logits.cpu()
