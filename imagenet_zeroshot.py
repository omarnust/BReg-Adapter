import os
import torch
import clip
import random
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

# 1. Configuration & Constants
DATAPATH = "/Users/oarif/Documents/datasets/imagenet/images/val"
MODEL_NAME = "RN50" # You can use RN50, ViT-B/16, etc.
DEVICE = "cuda" if torch.cuda.is_available() else "mps"
BATCH_SIZE = 64
NUM_IMAGES = -1 # NUM_IMAGES per class, Set to -1 to run on the entire validation set (50k images) 

# 2. ImageNet Classes & Prompts
# This list MUST be in alphabetical order of WordNet IDs to match ImageFolder
from datasets.imagenet import imagenet_classes # Using the list you provided earlier
from datasets.imagenet import imagenet_templates # Using the templates you provided

def zeroshot_classifier(classnames, templates, model):
    """
    Creates the 'weights' for the zero-shot classifier by encoding the 
    text prompts for every class.
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, desc="Encoding Text Labels"):
            # Create prompts (e.g., "a photo of a tench")
            texts = [template.format(classname) for template in templates] 
            texts = clip.tokenize(texts).to(DEVICE)
            # Encode and normalize
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # Average the embeddings across all templates for this class
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(DEVICE)
    return zeroshot_weights

def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def run_zeroshot():
    # 1. Load Model and force to Float32 for Mac MPS compatibility
    print(f"Loading CLIP model: {MODEL_NAME}...")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model = model.float() 
    model.eval()

    # 2. Load the full dataset first
    print(f"Loading Dataset from: {DATAPATH}")
    full_dataset = ImageFolder(DATAPATH, transform=preprocess)
    
    # 3. Balanced Subsetting Logic: Pick NUM_IMAGES from EACH class
    if NUM_IMAGES > 0:
        # group all image indices by their class label
        indices_per_class = defaultdict(list)
        for idx, (_, label) in enumerate(full_dataset.samples):
            indices_per_class[label].append(idx)
        
        selected_indices = []
        for label in indices_per_class:
            # Randomly sample images from each class
            # We use min() just in case a class has fewer than requested
            available_samples = indices_per_class[label]
            count = min(len(available_samples), NUM_IMAGES)
            selected_indices.extend(random.sample(available_samples, count))
        
        dataset = Subset(full_dataset, selected_indices)
        print(f"Balanced Sampling: {NUM_IMAGES} images/class. Total: {len(dataset)}")
    else:
        dataset = full_dataset
        print(f"Using full dataset. Total: {len(dataset)}")

    # 4. DataLoader (Using 0 workers on Mac often prevents the Semaphore Leak error)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)

    # 5. Build Zero-Shot Classifier weights
    print("Building zero-shot classifier weights...")
    # Ensure weights are also float32
    zshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates, model).float()

    # 6. Evaluation Loop
    top1, top5, n = 0., 0., 0.
    features, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader, desc="Classifying")):
            images = images.to(DEVICE).float()
            target = target.to(DEVICE)

            # Encode image features
            image_features = model.encode_image(images)
            
            # L2 Normalization
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Compute Cosine Similarity (Logits)
            # scale by 100 (Temperature tau=0.01)
            logits = 100. * image_features @ zshot_weights

            # Measure Accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1.item()
            top5 += acc5.item()
            n += 1

            features.append(image_features.cpu())
            labels.append(target.cpu())

    features, labels = torch.cat(features), torch.cat(labels)                 
    torch.save({'features': features.cpu(), 'labels': labels.cpu()}, 'test.pt')

    print(f"\nFinal Results for {len(dataset)} images:")
    print(f"Top-1 Accuracy: {top1 / n:.2f}%")
    print(f"Top-5 Accuracy: {top5 / n:.2f}%")

if __name__ == "__main__":
    run_zeroshot()