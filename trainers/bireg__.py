from genericpath import exists
import torch 
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import json
from utils import save_hps, cls_acc

def RBF_Kernel(X, Y, beta):
    return (-beta*(1-X.float()@Y.float().T)).exp()  

def BiReg(vecs, labels, val_features, val_labels, test_features, clip_weights,
          dataset, shots, seed, hp_selection, backbone='RN50'):
    """
    Bilateral Regularization for Zero-Shot Learning (Kernel Space)
    Memory-safe version for ImageNet-scale datasets
    """

    device = vecs.device
    n_classes = len(labels.unique())
    cache = F.one_hot(labels, num_classes=n_classes).float()

    # Normalize dataset name
    if 'imagenet' in dataset:
        dataset = 'imagenet'

    # CLIP text embeddings
    A = clip_weights.float()  # [C, d]
    logits_text_shots = torch.einsum('nd, cd -> nc', vecs.float(), A)

    # =========================
    # Hyperparameter selection
    # =========================
    if hp_selection != 'imagenet':

        cfg_path = f'./configs/{backbone}/configs_bireg/{dataset}/seed{seed}_shots{shots}.json'
        if exists(cfg_path):
            best_hp = json.load(open(cfg_path, 'r'))
            best_beta = best_hp['beta']
            best_gamma = best_hp['gamma']
            best_lmbda = best_hp['lmbda']
            print(f'Loaded best hp: beta={best_beta}, gamma={best_gamma}, lmbda={best_lmbda}')
        else:
            cfg = yaml.load(
                open(f'./configs/{backbone}/configs_bireg/{dataset}.yaml', 'r'),
                Loader=yaml.Loader
            )

            # ---- SUBSAMPLE FOR HP SEARCH (CRITICAL) ----
            MAX_HP_SAMPLES = 2000
            if vecs.size(0) > MAX_HP_SAMPLES:
                idx = torch.randperm(vecs.size(0))[:MAX_HP_SAMPLES]
                vecs_hp = vecs[idx]
                cache_hp = cache[idx]
                logits_text_hp = logits_text_shots[idx]
            else:
                vecs_hp = vecs
                cache_hp = cache
                logits_text_hp = logits_text_shots

            # Move to GPU only what is needed
            vecs_hp = vecs_hp.to(device).half()
            cache_hp = cache_hp.to(device)
            logits_text_hp = logits_text_hp.to(device)
            val_features = val_features.to(device).half()
            val_labels = val_labels.to(device)

            logits_text_val = torch.einsum(
                'nd, cd -> nc', val_features.float(), A
            )

            beta_search = torch.linspace(*cfg['beta'])
            gamma_search = torch.linspace(*cfg['gamma'])
            lmbda_search = torch.linspace(*cfg['lmbda'])

            best_val_acc = 0.0

            eye_N = torch.eye(vecs_hp.size(0), device=device)
            eye_C = torch.eye(A.size(0), device=device)

            for beta in tqdm(beta_search):

                # ---- Kernel computed ONCE per beta ----
                K_SS = RBF_Kernel(vecs_hp, vecs_hp, beta).half()
                K_XS = RBF_Kernel(val_features, vecs_hp, beta).half()

                for gamma in gamma_search:
                    right = A @ A.T + gamma * eye_C
                    right_inv = torch.linalg.inv(right)

                    for lmbda in lmbda_search:
                        left = (1.0 / lmbda) * K_SS + eye_N

                        middle = (cache_hp - logits_text_hp) @ A.T
                        tmp = (right_inv @ middle.T).T
                        alpha = torch.linalg.solve(left, tmp)

                        val_logits = logits_text_val + K_XS @ alpha @ A
                        acc = cls_acc(val_logits, val_labels)

                        if acc > best_val_acc:
                            best_val_acc = acc
                            best_beta = beta
                            best_gamma = gamma
                            best_lmbda = lmbda

                        # ---- explicit cleanup ----
                        del left, middle, tmp, alpha, val_logits
                        torch.cuda.empty_cache()

                del K_SS, K_XS
                torch.cuda.empty_cache()

            best_hp = {
                'beta': float(best_beta),
                'gamma': float(best_gamma),
                'lmbda': float(best_lmbda)
            }
            save_hps(best_hp, f'./configs/{backbone}/configs_bireg/{dataset}', seed, shots)
            print(f'best: beta={best_beta}, gamma={best_gamma}, lmbda={best_lmbda}, val acc={best_val_acc:.2f}%')

    else:
        best_hp = json.load(
            open(f'./configs/{backbone}/configs_bireg/hyperparameters.json', 'r')
        )[str(shots)]
        best_beta = best_hp['beta']
        best_gamma = best_hp['gamma']
        best_lmbda = best_hp['lmbda']

    # =========================
    # Final training (FULL SET)
    # =========================
    vecs = vecs.to(device).half()
    cache = cache.to(device)
    logits_text_shots = logits_text_shots.to(device)

    eye_N = torch.eye(vecs.size(0), device=device)
    eye_C = torch.eye(A.size(0), device=device)

    K_SS = RBF_Kernel(vecs, vecs, best_beta).half()
    left = (1.0 / best_lmbda) * K_SS + eye_N
    right = A @ A.T + best_gamma * eye_C

    middle = (cache - logits_text_shots) @ A.T
    tmp = torch.linalg.solve(right, middle.T).T
    alpha = torch.linalg.solve(left, tmp)

    # =========================
    # Test prediction
    # =========================
    test_features = test_features.to(device).half()
    K_XS = RBF_Kernel(test_features, vecs, best_beta).half()
    logits_text_test = torch.einsum(
        'nd, cd -> nc', test_features.float(), A
    )
    test_logits = logits_text_test + K_XS @ alpha @ A

    return test_logits.cpu()
