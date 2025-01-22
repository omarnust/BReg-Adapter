from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import clip
import argparse 
import os
import json
import math 

class SmartFormatter(argparse.HelpFormatter):
    """
        Custom help message formatter for argparse
    """
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()  
        return argparse.HelpFormatter._split_lines(self, text, width)
def get_arguments():
    parser = argparse.ArgumentParser(formatter_class=SmartFormatter)
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--method', type=str, default='tip', help='method to use')
    parser.add_argument('--dataset-path', type=str, default='/nasbrain/datasets', help='path where datasets are stored')
    parser.add_argument('--shots-path', type=str, default='/nasbrain/y17bendo/ProKeR/shots/', help='path where to store shot features')
    parser.add_argument('--test-path', type=str, default='/nasbrain/y17bendo/ProKeR/test/', help='paths where to store validation / test features and clip weights')
    parser.add_argument('--cache-dir', type=str, default='/nasbrain/y17bendo/cache/', help='cache directory')
    parser.add_argument('--augment-epoch', type=int, default=10, help='nb of augmentations for shots')
    parser.add_argument('--shots', nargs='+', type=int, default=-1, help='number of shots')
    parser.add_argument('--seeds', nargs='+', type=int, default=-1, help='seeds')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--backbone', type=str, default='RN50', help='backbone')
    parser.add_argument('--hp-selection', type=str, default='tip-adapter', help='R|Hyperparameter selection. \n- coop: uses min(4, n_shots) from Prompt Learning paper.\n- tip-adapter: uses the entire validation set (used by Tip-Adapter, APE and GDA).\n- imagenet: transfers from imagenet.', choices=['tip-adapter', 'coop', 'imagenet'])
     
    args = parser.parse_args()
    if args.shots == -1 or (type(args.shots) == list and len(args.shots) == -1):
        args.shots = [1, 2, 4, 8, 16]
    if args.seeds == -1 or (type(args.seeds) == list and len(args.seeds) == -1):
        args.seeds = [1, 2, 3]
    return args        

def cls_acc(output, target, topk=1):
    """
        Computes accuracy
    """
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model, device='cuda:0'):
    """
        Compute the clip weights for the classifier
    """
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(device)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights

def pre_load_features(clip_model, loader, norm=True, load_path='', device='cuda:0', overwrite=False, n_shots=-1):
    try:
        if overwrite: 
            assert 1==2, 'Overwritting regardless of the file'
        f = torch.load(load_path, map_location='cpu')
        features, labels = f['features'], f['labels']
    # else:
    except Exception as e:
        print(e)
        clip_model = clip_model.to(device)
        features, labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.to(device), target.to(device)
                image_features = clip_model.encode_image(images)
                if norm:
                    image_features /= image_features.norm(dim=-1, keepdim=True) 
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)         
        if load_path != '':
            torch.save({'features': features.cpu(), 'labels': labels.cpu()}, load_path)
    # workaround over the original dataloader tip-adapter to avoid using a large validation set
    # could be parallelized if validation features had the same nb of elements per class
    if n_shots > 0: 
        _, count_per_class = torch.unique(labels, return_counts=True)
        val_shots = min(n_shots, 4) # number of obtained shots
        cum_sum = torch.cat([torch.tensor([0]), torch.cumsum(count_per_class, dim=0)])[:-1] # start with 0 
        random_idx = torch.stack([torch.randperm(count)[:val_shots] + cum_sum[c] for c, count in enumerate(count_per_class)])
        random_idx = random_idx.flatten()
        features, labels = features[random_idx], labels[random_idx]
    return features, labels

def build_cache_model(cfg, clip_model, train_loader_cache):  
    cache_keys = []
    cache_values = []

    with torch.no_grad():
        # Data augmentation for the cache model
        for augment_idx in range(cfg['augment_epoch']):
            train_features = []

            print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                images = images.cuda()
                image_features = clip_model.encode_image(images)
                train_features.append(image_features)
                if augment_idx == 0:
                    target = target.cuda()
                    cache_values.append(target)
            cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

    cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
    cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)
    cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

    return cache_keys, cache_values

def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha

train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
train_tranform_clean = transforms.Compose([
        transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

def get_clip_weights(classnames, template, clip_model, device='cuda:0'):
    clip_model = clip_model.to(device)
    with torch.no_grad():
        clip_weights = []
        
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(device)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights
class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index]
    
    def __len__(self):
        return self.input_tensor.size(0)

def save_hps(hps, directory_path, seed):
    if not os.path.exists(directory_path): 
        os.makedirs(directory_path)
    path = os.path.join(directory_path, f'seed_{seed}.json')
    json.dump(hps, open(path, 'w'))

def validate(model, val_features, batch_size=256, device='cuda:0'): 
    """
        Run model on validation set
    """
    val_logits = []
    with torch.no_grad():
        for i in range(math.ceil(len(val_features)/batch_size)):
            val_batch = val_features[i*batch_size:(i+1)*batch_size].float().to(device)
            val_logits.append(model(val_batch).cpu())
    return torch.cat(val_logits)