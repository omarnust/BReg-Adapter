import yaml
import json
from utils import save_hps, cls_acc
from tqdm import trange
import torch.nn.functional as F

def TIP(vecs, labels, val_features, val_labels, test_features, clip_weights, dataset, shots, seed, hp_selection, backbone='RN50'):
    """
        Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification, ECCV 2022.
    """
    n_classes = len(labels.unique())
    cache = F.one_hot(labels, num_classes = n_classes).float()
    device = vecs.device

    if 'imagenet' in dataset: # if imagenet-r/a/v2/sketch, use cfg of imagenet
        dataset = 'imagenet'

    if hp_selection != 'imagenet': # grid search for hyper-parameters
        cfg = yaml.load(open(f'./configs/{backbone}/configs_tip/{dataset}.yaml', 'r'), Loader=yaml.Loader) # cfg for hp search
        best_val_acc = 0
        val_features = val_features.to(device)
        val_labels = val_labels.to(device)
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        from itertools import product
        beta_alpha = product(beta_list, alpha_list)
        pbar = trange(cfg['search_step'][0] * cfg['search_step'][1])
        for i in pbar:
            beta, alpha = next(beta_alpha)
            val_logits = 100. * val_features.float() @ clip_weights.float() + alpha * (-(1 - val_features.float() @ vecs.T) * beta).exp() @ cache
            acc = cls_acc(val_logits.cpu(), val_labels.cpu())
            if acc > best_val_acc:
                best_val_acc = acc
                best_alpha = alpha
                best_beta =beta
        best_hp = {'alpha': best_alpha, 'beta': best_beta}
        save_hps(best_hp, f'./configs/{backbone}/configs_tip/{dataset}', seed) # save best hp
        print(f"best_val_alpha: {best_alpha} \t best_val_beta: {best_beta} \t best_val_acc: {best_val_acc}")
    else: # load best hp from imagenet 
        best_hp = json.load(open(f'./configs/{backbone}/configs_tip/hyperparameters.json', 'r'))[str(shots)] # Pre-computed hps and averaged per seed
        best_alpha, best_beta = best_hp['alpha'], best_hp['beta']

    test_features = test_features.to(device)
    test_logits = 100. * test_features.float() @ clip_weights.float() + best_alpha * (-(1 - test_features.float() @ vecs.T) * best_beta).exp() @ cache
    return test_logits.cpu()
