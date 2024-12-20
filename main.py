#!/usr/bin/env python
# coding: utf-8
import os
import random
import yaml
from tqdm import tqdm, trange
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets.imagenet import ImageNet
from datasets.imagenet_a import ImageNet_A
from datasets.imagenet_r import ImageNet_R
from datasets.imagenet_sketch import ImageNet_Sketch
from datasets.imagenet_v2 import ImageNet_V2
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
from tqdm import trange
import json


def TIP(vecs, labels, val_features, val_labels, test_features, clip_weights, dataset, seed, hp_selection, backbone='RN50'):
    """
        Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification, ECCV 2022.
    """
    if 'imagenet' in dataset: 
        dataset = 'imagenet'
    path = f'./configs/{backbone}/configs_tip/{dataset}.yaml'
    cfg = yaml.load(open(path, 'r'), Loader=yaml.Loader)
    if not os.path.exists(f'./configs/{backbone}/configs_tip/{dataset}'):
        os.makedirs(f'./configs/{backbone}/configs_tip/{dataset}')
    path = f'./configs/{backbone}/configs_tip/{dataset}/seed_{seed}.json'
    best_hp = json.load(open(path, 'r')) if os.path.exists(path) else {}
    
    cache = F.one_hot(labels).float()
    best_val_acc = 0
    device = vecs.device
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
    json.dump(best_hp, open(path, 'w'))
    print(f"best_val_alpha: {best_alpha} \t best_val_beta: {best_beta} \t best_val_acc: {best_val_acc}")
    test_features = test_features.to(device)
    test_logits = 100. * test_features.float() @ clip_weights.float() + best_alpha * (-(1 - test_features.float() @ vecs.T) * best_beta).exp() @ cache

    return test_logits.cpu()

def Zeroshot(vecs, labels, val_features, val_labels, test_features, clip_weights, dataset, seed, hp_selection, backbone='RN50'):
    """
        Zero-shot CLIP
    """
    device = vecs.device
    test_features = test_features.to(device)
    test_logits = 100. * test_features.float() @ clip_weights.float() 
    return test_logits.cpu()

def GDA(vecs, labels, val_features, val_labels, test_features, clip_weights, dataset, seed, hp_selection, backbone='RN50'):
    """
        A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation, ICLR 2024.
    """
    # normal distribution
    mus = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(clip_weights.shape[1])])
    device = vecs.device
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    # KS Estimator.  
    center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in range(clip_weights.shape[1])])
    cov_inv = center_vecs.shape[1] * torch.linalg.pinv((center_vecs.shape[0] - 1) * center_vecs.T.cov() + center_vecs.T.cov().trace() * torch.eye(center_vecs.shape[1]).to(device))    

    ps = torch.ones(clip_weights.shape[1]).to(device) * 1. / clip_weights.shape[1]
    W = torch.einsum('nd, dc -> cn', mus, cov_inv)
    b = ps.log() - torch.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2
    # Evaluate
    # Grid search for hyper-parameter alpha
    best_val_acc = 0
    best_alpha = 0.1
    for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        val_logits = 100. * val_features.float() @ clip_weights.float() + alpha * (val_features.float() @ W + b)
    
        acc = cls_acc(val_logits, val_labels)
        if acc > best_val_acc:
            best_val_acc = acc
            best_alpha = alpha
    
    print("best_val_alpha: %s \t best_val_acc: %s" % (best_alpha, best_val_acc))
    alpha = best_alpha
    test_features = test_features.to(device)
    test_logits = 100. * test_features.float() @ clip_weights.float() + alpha * (test_features.float() @ W + b)
    return test_logits.cpu()

def RBF_Kernel(X,Y, beta):
    return (-beta*(1-X.float()@Y.float().T)).exp()  

def KRRRBF(vecs, labels, val_features, val_labels, test_features, clip_weights, dataset, seed, hp_selection, backbone='RN50'):
    path = f'./configs/{backbone}/configs_krrrbf/{dataset}.yaml'
    cfg = yaml.load(open(path, 'r'), Loader=yaml.Loader)
    if not os.path.exists(f'./configs/{backbone}/configs_krrrbf/{dataset}'):
        os.makedirs(f'./configs/{backbone}/configs_krrrbf/{dataset}')
    hp_path = f'./configs/{backbone}/configs_krrrbf/{dataset}/seed_{seed}.json'
    best_hp = json.load(open(hp_path, 'r')) if os.path.exists(hp_path) else {}

    n_classes = len(labels.unique())
    labels = labels
    cache = F.one_hot(labels, num_classes = n_classes).float()
    device = vecs.device
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    logits_text_shots = torch.einsum('sd, cd -> sc', vecs.float(), clip_weights.float().T) # b,c
    logits_text = torch.einsum('bd, cd -> bc', val_features.float(), clip_weights.float().T) # b,c

    best_val_acc = 0.
    hp_search = True
    if hp_search: 
        beta_search = torch.linspace(float(cfg['beta'][0]), float(cfg['beta'][1]), cfg['beta'][2])
        lmbda_search = torch.linspace(float(cfg['lmbda'][0]), float(cfg['lmbda'][1]), cfg['lmbda'][2])
        for beta in tqdm(beta_search):
            K_SS = RBF_Kernel(vecs[:], vecs[:], beta=beta) # s,s
            K_XS = RBF_Kernel(val_features, vecs[:], beta=beta) # b,s
            for lmbda in lmbda_search:
                alpha_i = torch.linalg.solve(1/lmbda * K_SS + torch.eye(vecs.size(0)).to(device), cache - logits_text_shots)
                val_logits = logits_text + K_XS @ alpha_i
                acc = cls_acc(val_logits, val_labels)
                if acc > best_val_acc:
                    best_val_acc, best_lmbda, best_beta = acc, lmbda, beta
        best_hp = {'lmbda': best_lmbda.item(), 'beta': best_beta.item()}
        json.dump(best_hp, open(hp_path, 'w'))
    else: 
        best_beta, best_lmbda = best_hp[str(seed)]['beta'], best_hp[str(seed)]['lmbda']
    print(f'best: beta={best_beta}, lmbda={best_lmbda}, val acc={best_val_acc:.2f}%')

    # solve linear system with best hp
    K_SS = RBF_Kernel(vecs[:], vecs[:], beta=best_beta)
    alpha_i = torch.linalg.solve(1 / best_lmbda * K_SS + torch.eye(vecs.size(0)).to(device), cache - logits_text_shots)
    
    test_features = test_features.to(device)
    K_XS = RBF_Kernel(test_features, vecs[:], beta=best_beta) # b,s
    logits_text = torch.einsum('bd, cd -> bc', test_features.float(), clip_weights.float().T) # b,c
    test_logits = logits_text + K_XS @ alpha_i
    return test_logits.cpu()

def CLAP(vecs, labels, val_features, val_labels, test_features, clip_weights, dataset, seed, hp_selection, backbone='RN50'):
    """
        CLAP method "A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models" CVPR 2024.
    """
    from trainers.clap import CLAP_Head, train_clap
    clip_weights = F.normalize(clip_weights, dim=0)
    device = vecs.device
    logits_zs = torch.einsum('sd, cd -> sc', vecs.float(), clip_weights.float().T) # b,c
    model = CLAP_Head(clip_weights.T).to(device)
    res = train_clap(model, vecs, labels, logits_zs)
    model = CLAP_Head(clip_weights.T).to(device)     
    model.load_state_dict(res['state'])
    test_logits = validate(model, test_features, device=device)
    return test_logits

def KRR_CLAP(vecs, labels, val_features, val_labels, test_features, clip_weights, dataset, seed, hp_selection, backbone='RN50'):
    """
        Base learner of CLAP + KRR (without joint training)
    """
    from trainers.clap import CLAP_Head, train_clap
    # get hps of KRRRBF of imagenet, make sure to run KRRRBF alone before to get its hyperparameters
    path = f'./configs/{backbone}/configs_krrrbf/imagenet/seed_{seed}.json'
    best_hp = json.load(open(path, 'r')) if os.path.exists(path) else {}
    best_beta, best_lmbda = best_hp['beta'], best_hp['lmbda']
    
    device = vecs.device
    n_classes = len(labels.unique())
    test_features = test_features.to(device)
    cache = F.one_hot(labels, num_classes = n_classes).float()
    clip_weights = F.normalize(clip_weights, dim=0)

    logits_text_shots = torch.einsum('sd, cd -> sc', vecs.float(), clip_weights.float().T) # b,c
    
    model = CLAP_Head(clip_weights.T).to(device)
    res = train_clap(model, vecs, labels, logits_text_shots)
    clip_weights = res['state']['prototypes']
    K_SS = RBF_Kernel(vecs[:], vecs[:], beta=best_beta)
    K_XS = RBF_Kernel(test_features, vecs[:], beta=best_beta) # b,s
    logits_text = torch.einsum('bd, cd -> bc', test_features.float(), clip_weights.float()) # b,c
    alpha_i = torch.linalg.solve(1 / best_lmbda * K_SS + torch.eye(vecs.size(0)).to(device), cache - logits_text_shots)
    test_logits = logits_text + K_XS @ alpha_i
    return test_logits.cpu()

def KRR_CLAP_joint(vecs, labels, val_features, val_labels, test_features, clip_weights, dataset, seed, hp_selection, backbone='RN50'):
    """
        Jointly train KRR and CLAP
    """
    from trainers.krr_clap import KRR_CLAP_Head, train_clap
    path = f'./configs/{backbone}/configs_krrrbf_ft/imagenet/seed_{seed}.json'
    best_hp = json.load(open(path, 'r')) if os.path.exists(path) else {}
    best_beta, best_lmbda = best_hp['beta'], best_hp['lmbda']
    device = vecs.device
    test_features = test_features.to(device)
    clip_weights = F.normalize(clip_weights, dim=0)


    logits_text_shots = torch.einsum('sd, cd -> sc', vecs.float(), clip_weights.float().T) # b,c
    K_XS = lambda x: RBF_Kernel(x, vecs[:], beta=best_beta) # b,s
    K_SS = RBF_Kernel(vecs[:], vecs[:], beta=best_beta) # s,s
    model = KRR_CLAP_Head(clip_weights.T, K_XS, K_SS, best_lmbda, vecs, labels).to(device)
    res = train_clap(model, vecs, labels, logits_text_shots)
    model.load_state_dict(res['state'])
    test_logits = validate(model, test_features, device=device)
    return test_logits.cpu()


def validate(model, val_features, batch_size=256, device='cuda:0'): 
    val_logits = []
    with torch.no_grad():
        for i in range(math.ceil(len(val_features)/batch_size)):
            val_batch = val_features[i*batch_size:(i+1)*batch_size].float().to(device)
            val_logits.append(model(val_batch).cpu())
    return torch.cat(val_logits)

def run(classifier, cfg, train_loader_cache, test_features, test_labels, val_features, val_labels, clip_weights, clip_model, shots_path, label_mapping=None, device='cuda:0'):
    vecs = []
    labels = []
    try:
        cache = torch.load(shots_path, map_location=device)
        vecs, labels = cache['vecs'].to(device), cache['labels'].to(device)
    except Exception as e:
        print(e)
        cache = {}
        for _ in range(cfg["augment_epoch"]):
            for image, target in tqdm(train_loader_cache):
                image, target = image.to(device), target.to(device)
                with torch.no_grad():  
                    image_features = clip_model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                vecs.append(image_features)
                labels.append(target)
        vecs = torch.cat(vecs)
        labels = torch.cat(labels)
        torch.save({'vecs':vecs.cpu(), 'labels':labels.cpu()}, shots_path)
    test_logits = classifier(vecs, labels, val_features, val_labels, test_features, clip_weights, cfg['dataset'], seed=cfg['seed'], hp_selection=cfg['hp_selection'], backbone=cfg['backbone'])
    if label_mapping is not None: # for imagenet-r
        notune_acc = cls_acc(test_logits @ label_mapping.to(test_logits.device), test_labels)  
    else: 
        notune_acc = cls_acc(test_logits, test_labels)    
    print("Nonetune acc:", notune_acc)
    return notune_acc

def main(args):
    classifier = eval(args.model)
    # Load config file
    dataset = args.dataset
    cfg = {'root_path':args.path, 'subsample_classes':'all', 'dataset':dataset, 'augment_epoch':args.augment_epoch, 'backbone':args.backbone, 'hp_selection':args.hp_selection, 'device':args.device}
    print("\nRunning configs.")
    print(cfg, "\n")
    backbone_names = {'RN50': 'RN50', 'RN101': 'RN101', 'RN50x4': 'RN50x4', 'RN50x16': 'RN50x16', 'ViT-B-32': 'ViT-B/32', 'ViT-B-16': 'ViT-B/16', 'ViT-L-14': 'ViT-L/14'}
    # CLIP
    test_path = os.path.join(args.test_path, args.backbone)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    test_path = os.path.join(args.test_path, args.backbone, cfg['dataset'])
    if not os.path.exists(test_path):
        os.makedirs(test_path)
        
    if os.path.exists("/nasbrain/y17bendo/cache"):
        clip_model, preprocess = clip.load(backbone_names.get(cfg['backbone'], cfg['backbone']), device=args.device, download_root="/nasbrain/y17bendo/cache")
        clip_model.eval()
        clip_model = clip_model.float().to(args.device)
        for p in clip_model.parameters():
            p.requires_grad = False
    else:
        clip_model, preprocess = None, None        
    notune_accs = {"1": [], "2": [], "3": []}
    for seed in args.seeds:
        cfg["seed"] = seed
        random.seed(seed)
        torch.manual_seed(seed) 
        print(f"---- Seed {seed} ----")
        for shots in args.shots:
            shots_path = os.path.join(args.shots_path, args.backbone, f'augment{cfg["augment_epoch"]}')
            if not os.path.exists(shots_path):
                os.makedirs(shots_path)
            shots_path = os.path.join(args.shots_path, args.backbone, f'augment{cfg["augment_epoch"]}', cfg['dataset'])
            if not os.path.exists(shots_path):
                os.makedirs(shots_path)
            shots_path = os.path.join(args.shots_path, args.backbone,f'augment{cfg["augment_epoch"]}', cfg['dataset'])
            if not os.path.exists(shots_path):
                os.makedirs(shots_path)
            print(shots_path)
            clip_weights_path = os.path.join(args.shots_path, args.backbone,f'augment10', cfg['dataset'], f'textweights_s1_k1.pt')
            cfg["shots"] = shots
            print(f"---- {shots}-shots ----")
            if cfg['dataset'] != "imagenet":
                dataset = build_dataset(cfg, cfg['dataset'], cfg['root_path'], cfg['shots']) 
                train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform if cfg['augment_epoch']>1 else train_tranform_clean, is_train=True, shuffle=False)
                test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False)
                val_loader = build_data_loader(data_source=dataset.val, batch_size=256, is_train=False, tfm=preprocess, shuffle=False)
                test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader, load_path=os.path.join(test_path, f'test_s{seed}_k{shots}.pt'), device=args.device)
                val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader, load_path=os.path.join(test_path, f'val_s{seed}_k{shots}.pt'), device=args.device, n_shots=-1 if args.hp_selection == 'tip-adapter' else shots)
                classnames, template = dataset.classnames, dataset.template
            else:
                try: 
                    if not os.path.exists(os.path.join(shots_path, f'shots_s{seed}_k{shots}.pt')):
                        assert 1==2, 'get a loader'
                    train_loader_cache, test_loader = None, None
                    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader, load_path=os.path.join(test_path, f'test_s{seed}_k{shots}.pt'), device=args.device)   
                    classnames, template = None, None
                except: 
                    dataset = ImageNet(cfg, cfg['root_path'], cfg['shots'], preprocess)
                    train_loader_cache = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=False)
                    test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)
                    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader, load_path=os.path.join(test_path, f'test_s{seed}_k{shots}.pt'), device=args.device)   
                    classnames, template = dataset.classnames, dataset.template
                
                # on imagenet, val and test are the same:
                # https://github.com/KaiyangZhou/CoOp/blob/main/datasets/imagenet.py#L61
                # https://github.com/jusiro/CLAP/blob/main/datasets/imagenet.py#L51
                val_features, val_labels = test_features, test_labels 
                
            test_features = test_features.cpu()
            test_labels = test_labels.cpu()
            val_features = val_features.cpu()
            val_labels = val_labels.cpu()
            
            try:
                clip_weights = torch.load(clip_weights_path, map_location=args.device).to(args.device)
            except Exception as e:
                print(e)
                clip_weights = clip_classifier(classnames, template, clip_model, device=args.device)   
                torch.save(clip_weights.cpu(), clip_weights_path)
            notune_acc = run(classifier, cfg, train_loader_cache, test_features, test_labels, val_features, val_labels, clip_weights, clip_model, shots_path=os.path.join(shots_path, f'shots_s{seed}_k{shots}.pt'), device=args.device)
            notune_accs[str(cfg["seed"])].append(notune_acc)
    print("Evaluate on dataset:", cfg['dataset'])
    notune = []
    for seed in ["1", "2", "3"]:
        print("seed %s" % seed, notune_accs[str(seed)])
        notune.append(notune_accs[seed])
    notune = torch.tensor(notune)
    print("Average: ", notune.mean(dim=0))
    return notune
def robustness(target_dataset):
    target_datasets = ['imagenet-v2', 'imagenet-sketch', 'imagenet-a', 'imagenet-r']
    assert target_dataset in target_datasets, f"target_dataset should be one of {target_datasets}"
    dataset_list = {
            'imagenet-v2': ImageNet_V2,
            'imagenet-sketch': ImageNet_Sketch,
            'imagenet-a': ImageNet_A,
            'imagenet-r': ImageNet_R
    }
    classifier = eval(args.model)
    # Load config file
    cfg = {'root_path':args.path, 'subsample_classes':'all', 'dataset':target_dataset, 'augment_epoch':args.augment_epoch, 'backbone':args.backbone, 'hp_selection':args.hp_selection, 'device':args.device}
    # Load cfg for conditional prompt.
    print("\nRunning configs.")
    print(cfg, "\n")
    backbone_names = {'RN50': 'RN50', 'RN101': 'RN101', 'RN50x4': 'RN50x4', 'RN50x16': 'RN50x16', 'ViT-B-32': 'ViT-B/32', 'ViT-B-16': 'ViT-B/16', 'ViT-L-14': 'ViT-L/14'}
    # CLIP
    test_path = os.path.join(args.test_path, args.backbone)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    test_path = os.path.join(args.test_path, args.backbone, cfg['dataset'])
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    val_path = os.path.join(args.test_path, args.backbone, 'imagenet')
    if os.path.exists("/nasbrain/y17bendo/cache"):
        clip_model, preprocess = clip.load(backbone_names.get(cfg['backbone'], cfg['backbone']), download_root="/nasbrain/y17bendo/cache")
        clip_model.eval()
        clip_model = clip_model.float().to(args.device)
        for p in clip_model.parameters():
            p.requires_grad = False
    else:
        clip_model, preprocess = None, None        
    accs = {"1": [], "2": [], "3": []}
    for seed in [1, 2, 3]:
        cfg["seed"] = seed
        random.seed(seed)
        torch.manual_seed(seed) 
        print(f"---- Seed {seed} ----")
        # Source dataset
        print('Augment:', cfg["augment_epoch"])
        for shots in args.shots:
            shots_path = os.path.join(args.shots_path, args.backbone, f'augment{cfg["augment_epoch"]}')
            if not os.path.exists(shots_path):
                os.makedirs(shots_path)
            shots_path = os.path.join(args.shots_path, args.backbone, f'augment{cfg["augment_epoch"]}', 'imagenet')
            if not os.path.exists(shots_path):
                os.makedirs(shots_path)
            shots_path = os.path.join(args.shots_path, args.backbone,f'augment{cfg["augment_epoch"]}', 'imagenet')
            if not os.path.exists(shots_path):
                os.makedirs(shots_path)
            print(shots_path)
            clip_weights_path = os.path.join(args.shots_path, args.backbone,f'augment10', 'imagenet', f'textweights_s1_k1.pt')
            cfg["shots"] = shots
            print(f"---- {shots}-shots ----")

            train_loader_cache, test_loader = None, None
            val_features, val_labels = pre_load_features(cfg, "test", clip_model, test_loader, load_path=os.path.join(val_path, f'test_s{seed}_k{shots}.pt'), device=args.device)   
            classnames, template = None, None
            test_path_ = os.path.join(test_path, f'test_s{seed}_k{shots}.pt')
            dataset = dataset_list[target_dataset](cfg, cfg['root_path'], cfg['shots'], preprocess) 
            if os.path.exists(test_path_):
                test_loader = None 
            else:
                test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)
            test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader, load_path=os.path.join(test_path, f'test_s{seed}_k{shots}.pt'), device=args.device)   
            
            test_features = test_features.cpu()
            test_labels = test_labels.cpu()
            val_features = val_features.cpu()
            val_labels = val_labels.cpu()
            try:
                clip_weights = torch.load(clip_weights_path, map_location=args.device).to(args.device)
            except Exception as e:
                print(e)
                clip_weights = clip_classifier(classnames, template, clip_model, device=args.device)   
                torch.save(clip_weights.cpu(), clip_weights_path)
            acc_one_seed = run(classifier, cfg, train_loader_cache, test_features, test_labels, val_features, val_labels, clip_weights, clip_model, shots_path=os.path.join(shots_path, f'shots_s{seed}_k{shots}.pt'), label_mapping=dataset.label_mapping, device=args.device)
            accs[str(cfg["seed"])].append(acc_one_seed)
    print("Evaluate on dataset:", cfg['dataset'])
    accuracies = []
    for seed in ["1", "2", "3"]:
        print("seed %s" % seed, accs[str(seed)])
        accuracies.append(accs[seed])
    accuracies = torch.tensor(accuracies)
    print("Average: ", accuracies.mean(dim=0))
    return accuracies
if __name__ == '__main__':
    args = get_arguments()
    robust_imagenet = ['imagenet-v2','imagenet-sketch','imagenet-a','imagenet-r']
    
    if args.dataset in robust_imagenet: 
        res = robustness(args.dataset)
    else:
        res = main(args)

    print(f'{args.model} lmbda={args.lmbda}: {res.mean(dim=0).item()}')
