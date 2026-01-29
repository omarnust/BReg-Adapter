import os
import test
import torch
import clip
from tqdm import tqdm

# 1. Configuration & Constants
CACHE_DIR = "./cache_dir"
CLIP_MODEL_PATH = os.path.join(CACHE_DIR, "cache/")
TEST_FEATURES_PATH = os.path.join("./test.pt")
MODEL_NAME = "RN50"
DEVICE = "cuda" if torch.cuda.is_available() else "mps"

# 2. ImageNet Classes & Prompts
from datasets.imagenet import imagenet_classes
from datasets.imagenet import imagenet_templates

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

def run_zeroshot_cached():
    print("="*60)
    print("Zero-Shot ImageNet Classification (Cached Version)")
    print("="*60)
    
    # 1. Load CLIP Model from cached weights
    print(f"\n[1/4] Loading CLIP model from: {CLIP_MODEL_PATH}")
    # Load the base model architecture first
    model, _ = clip.load(MODEL_NAME, device=DEVICE, download_root=CLIP_MODEL_PATH)
    print(f"✓ Successfully loaded cached model")
    model = model.float()
    model.eval()

    # 2. Load Pre-computed Test Features
    print(f"\n[2/4] Loading cached test features from: {TEST_FEATURES_PATH}")
    if not os.path.exists(TEST_FEATURES_PATH):
        print(f"Error: Cached features not found at {TEST_FEATURES_PATH}")
        print("Please run the feature extraction script first.")
        return
    
    cache = torch.load(TEST_FEATURES_PATH, map_location=DEVICE)
    
    # Extract features and labels from cache
    # The cache structure might vary, so we'll try different possible keys
    if isinstance(cache, dict):
        if 'features' in cache and 'labels' in cache:
            test_features = cache['features']
            test_labels = cache['labels']
        elif 'vecs' in cache and 'labels' in cache:
            test_features = cache['vecs']
            test_labels = cache['labels']
        else:
            # If it's a dict but doesn't have expected keys, try to infer
            print(f"Cache keys: {cache.keys()}")
            raise ValueError("Unexpected cache structure. Expected 'features' and 'labels' or 'vecs' and 'labels'")
    else:
        # If it's a tuple or list
        test_features, test_labels = cache
    
    test_features = test_features.to(DEVICE).float()
    test_labels = test_labels.to(DEVICE)
    
    print(f"✓ Loaded {test_features.shape[0]} test samples")
    print(f"  Feature shape: {test_features.shape}")
    print(f"  Labels shape: {test_labels.shape}")

    # 3. Build Zero-Shot Classifier Weights from Text
    print(f"\n[3/4] Building zero-shot classifier weights...")
    print(f"  Processing {len(imagenet_classes)} classes with {len(imagenet_templates)} templates each")
    zshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates, model).float()
    print(f"✓ Text classifier shape: {zshot_weights.shape}")

    # 4. Evaluation (Compute Logits and Accuracy)
    print(f"\n[4/4] Evaluating on cached features...")
    with torch.no_grad():
        # Ensure test features are normalized (they should be from cache, but double-check)
        #test_features = test_features / test_features.norm(dim=-1, keepdim=True)
        
        # Compute Cosine Similarity (Logits)
        # Scale by 100 (Temperature tau=0.01)
        logits = 100. * test_features @ zshot_weights

        # Measure Accuracy
        acc1, acc5 = accuracy(logits, test_labels, topk=(1, 5))

    print("\n" + "="*60)
    print(f"RESULTS (on {test_features.shape[0]} images)")
    print("="*60)
    print(f"Top-1 Accuracy: {acc1.item():.2f}%")
    print(f"Top-5 Accuracy: {acc5.item():.2f}%")
    print("="*60)

if __name__ == "__main__":
    run_zeroshot_cached()
