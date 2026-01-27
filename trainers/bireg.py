from genericpath import exists
import torch 
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import json
from utils import save_hps, cls_acc

def RBF_Kernel(X, Y, beta):
    return (-beta*(1-X.float()@Y.float().T)).exp()  

def BiReg(vecs, labels, val_features, val_labels, test_features, clip_weights, dataset, shots, seed, hp_selection, backbone='RN50'):
    """
        Bilateral Regularization for Zero-Shot Learning (Kernel Space)
        Adds (AA^T + lambda I)^-1 term to ProKeR formulation
    """
    n_classes = len(labels.unique())
    cache = F.one_hot(labels, num_classes=n_classes).float()
    device = vecs.device
    
    # CLIP text embeddings for training classes
    A = clip_weights.float()  # [n_classes, d]
    logits_text_shots = torch.einsum('sd, cd -> sc', vecs.float(), A.T)  # [N, n_classes]

    if 'imagenet' in dataset:
        dataset = 'imagenet'

    if hp_selection != 'imagenet':
        
        if exists(f'./configs/{backbone}/configs_bireg/{dataset}/seed{seed}_shots{shots}.json'):
            best_hp = json.load(open(f'./configs/{backbone}/configs_bireg/{dataset}/seed{seed}_shots{shots}.json', 'r'))
            best_beta, best_gamma, best_lmbda = best_hp['beta'], best_hp['gamma'], best_hp['lmbda']
            print(f'Loaded best hp: beta={best_beta}, gamma={best_gamma}, lmbda={best_lmbda}')
        else:
            cfg = yaml.load(open(f'./configs/{backbone}/configs_bireg/{dataset}.yaml', 'r'), Loader=yaml.Loader)
            best_val_acc = 0
            val_features = val_features.to(device)
            val_labels = val_labels.to(device)
            logits_text_val = torch.einsum('bd, cd -> bc', val_features.float(), A.T)

            beta_search = torch.linspace(float(cfg['beta'][0]), float(cfg['beta'][1]), cfg['beta'][2])
            gamma_search = torch.linspace(float(cfg['gamma'][0]), float(cfg['gamma'][1]), cfg['gamma'][2])
            lmbda_search = torch.linspace(float(cfg['lmbda'][0]), float(cfg['lmbda'][1]), cfg['lmbda'][2])
            
            for beta in tqdm(beta_search):
                K_SS = RBF_Kernel(vecs[:], vecs[:], beta=beta)  # [N, N]
                K_XS = RBF_Kernel(val_features, vecs[:], beta=beta)  # [N_val, N]
                
                for gamma in gamma_search:
                    for lmbda in lmbda_search:
                        # Bilateral regularization: solve for alpha
                        # Left: (1/lmbda * K + I)
                        left = 1/lmbda * K_SS + torch.eye(vecs.size(0)).to(device)
                        
                        # Right: (AA^T + gamma I)
                        right = A @ A.T + gamma * torch.eye(A.shape[0]).to(device)
                        
                        # Middle: (Y - F_s) @ A^T where F_s = logits_text_shots
                        middle = (cache - logits_text_shots) @ A.T
                        
                        # Solve: alpha = left^-1 @ (middle @ right^-1)^T
                        tmp = torch.linalg.solve(right, middle.T).T
                        alpha = torch.linalg.solve(left, tmp)
                        
                        # Prediction: F_t + K_XS @ alpha @ A
                        val_logits = logits_text_val + K_XS @ alpha @ A
                        acc = cls_acc(val_logits, val_labels)
                        
                        if acc > best_val_acc:
                            best_val_acc = acc
                            best_beta = beta
                            best_gamma = gamma
                            best_lmbda = lmbda
            
            best_hp = {
                'beta': best_beta.item() if torch.is_tensor(best_beta) else best_beta,
                'gamma': best_gamma.item() if torch.is_tensor(best_gamma) else best_gamma,
                'lmbda': best_lmbda.item() if torch.is_tensor(best_lmbda) else best_lmbda
            }
            save_hps(best_hp, f'./configs/{backbone}/configs_bireg/{dataset}', seed, shots)
            print(f'best: beta={best_beta}, gamma={best_gamma}, lmbda={best_lmbda}, val acc={best_val_acc:.2f}%')
    else:
        best_hp = json.load(open(f'./configs/{backbone}/configs_bireg/hyperparameters.json', 'r'))[str(shots)]
        best_beta, best_gamma, best_lmbda = best_hp['beta'], best_hp['gamma'], best_hp['lmbda']

    # Final prediction with best hyperparameters
    K_SS = RBF_Kernel(vecs[:], vecs[:], beta=best_beta)
    
    # Bilateral regularization
    left = 1/best_lmbda * K_SS + torch.eye(vecs.size(0)).to(device)
    right = A @ A.T + best_gamma * torch.eye(A.shape[0]).to(device)
    #right = 1/best_gamma * A @ A.T + torch.eye(A.shape[0]).to(device)
    
    middle = (cache - logits_text_shots) @ A.T
    
    tmp = torch.linalg.solve(right, middle.T).T
    alpha = torch.linalg.solve(left, tmp)
    #alpha = torch.linalg.solve(left, (cache - logits_text_shots))
    
    # Test prediction
    test_features = test_features.to(device)
    K_XS = RBF_Kernel(test_features, vecs[:], beta=best_beta)
    logits_text_test = torch.einsum('bd, cd -> bc', test_features.float(), A.T)
    test_logits = logits_text_test + K_XS @ alpha @ A
    
    return test_logits.cpu()