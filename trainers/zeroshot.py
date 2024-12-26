def ZeroShot(vecs, labels, val_features, val_labels, test_features, clip_weights, dataset, shots, seed, hp_selection, backbone='RN50'):
    """
        Zero-shot CLIP
    """
    device = vecs.device
    test_features = test_features.to(device)
    test_logits = 100. * test_features.float() @ clip_weights.float() 
    return test_logits.cpu()