#!/usr/bin/env python
# coding: utf-8
from genericpath import exists
import os
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from datasets.imagenet import ImageNet
from datasets.imagenet_a import ImageNet_A
from datasets.imagenet_r import ImageNet_R
from datasets.imagenet_sketch import ImageNet_Sketch
from datasets.imagenet_v2 import ImageNet_V2
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
from trainers import *
from pathlib import Path
from datetime import datetime
import pickle


def run(classifier, cfg, train_loader_cache, test_features, test_labels, val_features, val_labels, clip_weights, clip_model, shots_path, label_mapping=None, device='cuda:0'):
    """
        Run the few-shot classification
    """
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
                vecs.append(image_features.cpu())
                labels.append(target.cpu())
        vecs = torch.cat(vecs)
        labels = torch.cat(labels)
        torch.save({'vecs':vecs.cpu(), 'labels':labels.cpu()}, shots_path)
    test_logits = classifier(vecs, labels, val_features, val_labels, test_features, clip_weights, cfg['dataset'], shots=cfg['shots'], seed=cfg['seed'], hp_selection=cfg['hp_selection'], backbone=cfg['backbone'])
    if label_mapping is not None: # for imagenet-r
        notune_acc = cls_acc(test_logits @ label_mapping.to(test_logits.device), test_labels)  
    else: 
        notune_acc = cls_acc(test_logits, test_labels)    
    return notune_acc

def main(args):
    classifier = eval(args.method) # trainers are stored in trainers folder
    # Load config file
    dataset = args.dataset
    cfg = {'root_path':args.dataset_path, 'subsample_classes':'all', 'dataset':dataset, 'augment_epoch':args.augment_epoch, 'backbone':args.backbone, 'hp_selection':args.hp_selection, 'device':args.device}
    print("\nRunning config: ")
    print(cfg, "\n")
    backbone_names = {'RN50': 'RN50', 'RN101': 'RN101', 'RN50x4': 'RN50x4', 'RN50x16': 'RN50x16', 'ViT-B-32': 'ViT-B/32', 'ViT-B-16': 'ViT-B/16', 'ViT-L-14': 'ViT-L/14'}
    # CLIP
    test_path = os.path.join(args.test_path, args.backbone)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    test_path = os.path.join(args.test_path, args.backbone, cfg['dataset'])
    if not os.path.exists(test_path):
        os.makedirs(test_path)
        
    if os.path.exists(args.cache_dir):
        clip_model, preprocess = clip.load(backbone_names.get(cfg['backbone'], cfg['backbone']), device=args.device, download_root=args.cache_dir)
        clip_model.eval()
        clip_model = clip_model.float().to(args.device)
        for p in clip_model.parameters():
            p.requires_grad = False
    else:
        clip_model, preprocess = None, None        
    accs = {"1": [], "2": [], "3": []}
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
            clip_weights_path = os.path.join(args.shots_path, args.backbone,f'augment10', cfg['dataset'], f'textweights_s1_k1.pt')
            cfg["shots"] = shots
            if cfg['dataset'] != "imagenet":
                dataset = build_dataset(cfg, cfg['dataset'], cfg['root_path'], cfg['shots']) 
                train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform if cfg['augment_epoch']>1 else train_tranform_clean, is_train=True, shuffle=False)
                test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False)
                val_loader = build_data_loader(data_source=dataset.val, batch_size=256, is_train=False, tfm=preprocess, shuffle=False)
                test_features, test_labels = pre_load_features(clip_model, test_loader, load_path=os.path.join(test_path, f'test.pt'), device=args.device)

                if args.hp_selection != 'imagenet':
                  val_features, val_labels = pre_load_features(clip_model, val_loader, load_path=os.path.join(test_path, f'val.pt'), device=args.device, n_shots=-1 if args.hp_selection == 'tip-adapter' else shots)
                  
                else:
                  val_features, val_labels = None, None
                classnames, template = dataset.classnames, dataset.template
            else: # imagenet
                try: 
                    if not os.path.exists(os.path.join(shots_path, f'shots_s{seed}_k{shots}.pt')):
                        assert 1==2, 'get a loader'
                    train_loader_cache, test_loader = None, None
                    test_features, test_labels = pre_load_features(clip_model, test_loader, load_path=os.path.join(test_path, f'test.pt'), device=args.device)   
                    classnames, template = None, None
                except: 
                    
                    pickle_file = os.path.join(shots_path, f'dataset_s{seed}_k{shots}.pkl')
                    if os.path.exists(pickle_file):
                      print(f"Loading subsampled dataset from {pickle_file}")
                      dataset = ImageNet(cfg, cfg['root_path'], -1, preprocess)  # -1 so no subsampling inside
                      with open(pickle_file, 'rb') as f:
                        subset_data = pickle.load(f)
                        dataset.train.samples = subset_data['train_samples']
                        dataset.train.targets = subset_data['train_targets']
                    else:
                      print("No saved pickle found, creating ImageNet and subsampling...")
                      random.seed(seed)
                      dataset = ImageNet(cfg, cfg['root_path'], shots, preprocess)
                      # Save the subsampled dataset
                      subset_data = {
                          'train_samples': dataset.train.samples,
                          'train_targets': dataset.train.targets,
                      }
                      with open(pickle_file, 'wb') as f:
                          pickle.dump(subset_data, f)
                      print(f"Subsampled dataset saved to {pickle_file}")

                  
                    train_loader_cache = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=False)

                    # -  save dataset images
                    #materialize_imagenet_subset(dataset, dst_root=os.path.join(args.dataset_path, 'imagenet_fewshot'))
                    remap_dataset_paths(dataset.train,
                      old_root="/Users/oarif/Documents",
                      new_root="/workspace")

                    remap_dataset_paths(dataset.test,
                      old_root="/Users/oarif/Documents",
                      new_root="/workspace")

                
                    test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)
                    test_features, test_labels = pre_load_features(clip_model, test_loader, load_path=os.path.join(test_path, f'test.pt'), device=args.device)   
                    classnames, template = dataset.classnames, dataset.template
                
                # on imagenet, val and test are the same:
                # https://github.com/KaiyangZhou/CoOp/blob/main/datasets/imagenet.py#L61
                # https://github.com/jusiro/CLAP/blob/main/datasets/imagenet.py#L51
                val_features, val_labels = test_features, test_labels 
                
            test_features = test_features.cpu()
            test_labels = test_labels.cpu()
            if val_features is not None:
              val_features = val_features.cpu()
              val_labels = val_labels.cpu()
            
            try:
                clip_weights = torch.load(clip_weights_path, map_location=args.device).to(args.device)
            except Exception as e:
                print(e)
                dataset = ImageNet(cfg, cfg['root_path'], cfg['shots'], preprocess)
                classnames, template = dataset.classnames, dataset.template

                clip_weights = get_clip_weights(classnames, template, clip_model, device=args.device)   
                torch.save(clip_weights.cpu(), clip_weights_path)

            acc = run(classifier, cfg, train_loader_cache, test_features, test_labels, val_features, val_labels, clip_weights, clip_model, shots_path=os.path.join(shots_path, f'shots_s{seed}_k{shots}.pt'), device=args.device)
            accs[str(cfg["seed"])].append(acc)
            print(f"{shots}-shots : {acc:.2f}%")
    accuracies = []
    for seed in ["1", "2", "3"]:
        accuracies.append(accs[seed])
    accuracies = torch.tensor(accuracies)
    return accuracies
def main_robustness(target_dataset):
    """
        Train on ImageNet and evaluate on robustness datasets (imagenet-v2, imagenet-sketch, imagenet-a, imagenet-r)
    """
    target_datasets = ['imagenet-v2', 'imagenet-sketch', 'imagenet-a', 'imagenet-r']
    assert target_dataset in target_datasets, f"target_dataset should be one of {target_datasets}"
    dataset_list = {
            'imagenet-v2': ImageNet_V2,
            'imagenet-sketch': ImageNet_Sketch,
            'imagenet-a': ImageNet_A,
            'imagenet-r': ImageNet_R
    }
    classifier = eval(args.method)
    # Load config file
    cfg = {'root_path':args.dataset_path, 'subsample_classes':'all', 'dataset':target_dataset, 'augment_epoch':args.augment_epoch, 'backbone':args.backbone, 'hp_selection':args.hp_selection, 'device':args.device}
    # Load cfg for conditional prompt.
    print("\nRunning config: ")
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
            clip_weights_path = os.path.join(args.shots_path, args.backbone,f'augment10', 'imagenet', f'textweights_s1_k1.pt')
            cfg["shots"] = shots

            train_loader_cache, test_loader = None, None
            val_features, val_labels = pre_load_features(clip_model, test_loader, load_path=os.path.join(val_path, f'test_s{seed}_k{shots}.pt'), device=args.device)   
            classnames, template = None, None
            test_path_ = os.path.join(test_path, f'test_s{seed}_k{shots}.pt')
            dataset = dataset_list[target_dataset](cfg, cfg['root_path'], cfg['shots'], preprocess) 
            if os.path.exists(test_path_):
                test_loader = None 
            else:
                test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)
            test_features, test_labels = pre_load_features(clip_model, test_loader, load_path=os.path.join(test_path, f'test_s{seed}_k{shots}.pt'), device=args.device)   
            
            test_features = test_features.cpu()
            test_labels = test_labels.cpu()
            val_features = val_features.cpu()
            val_labels = val_labels.cpu()
            try:
                clip_weights = torch.load(clip_weights_path, map_location=args.device).to(args.device)
            except Exception as e:
                print(e)
                clip_weights = get_clip_weights(classnames, template, clip_model, device=args.device)   
                torch.save(clip_weights.cpu(), clip_weights_path)
            acc = run(classifier, cfg, train_loader_cache, test_features, test_labels, val_features, val_labels, clip_weights, clip_model, shots_path=os.path.join(shots_path, f'shots_s{seed}_k{shots}.pt'), label_mapping=dataset.label_mapping, device=args.device)
            accs[str(cfg["seed"])].append(acc)
            print(f"{shots}-shots : {acc:.2f}%")

    accuracies = []
    for seed in ["1", "2", "3"]:
        print("seed %s" % seed, accs[str(seed)])
        accuracies.append(accs[seed])
    accuracies = torch.tensor(accuracies)
    return accuracies


if __name__ == '__main__':
    args = get_arguments()
    robust_imagenet = ['imagenet-v2','imagenet-sketch','imagenet-a','imagenet-r']
    print("Evaluate on dataset:", args.dataset)
    if args.dataset in robust_imagenet: 
        res = main_robustness(args.dataset)
    else:
        res = main(args)

    print(f'{args.method} on {args.dataset}:', {k:round(v, 2) for k,v in zip(args.shots, res.mean(dim=0).tolist())})

    # save results to json
    results = {
      "method": args.method,
      "dataset": args.dataset,
      "backbone": args.backbone,
      "hp_selection": args.hp_selection,
      "augment_epoch": args.augment_epoch, 
      "results": {
          str(k): round(v, 2)
          for k, v in zip(args.shots, res.mean(dim=0).tolist())
      }
    }

    filename = (
      f"results_"
      f"{args.method}_"
      f"{args.dataset}"
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    path = out_dir / f"{filename}.json"

    if path.exists():
      timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
      path = out_dir / f"{filename}_{timestamp}.json"

    # write json
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {path}")

