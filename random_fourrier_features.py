import torch
from sklearn.utils import check_random_state, check_array
from scipy.linalg import qr_multiply
from scipy.stats import chi
import numpy as np
import warnings
import torch
from math import sqrt

def _get_random_matrix(distribution):
    return lambda rng, size:  rng.randn(*size)
def generate_orthogonal_matrix(d, D, device, sigma,use_offset = True,seed = None):

    random_state = check_random_state(seed)
    n_features, n_components_ = d, D
    n_stacks = int(np.ceil(n_components_/n_features))
    n_components = n_stacks * n_features
    if n_components != n_components_:
        msg = "n_components is changed from {0} to {1}.".format(
                D, n_components
        )
        msg += " You should set D to an n-tuple of n_features."
        warnings.warn(msg)
        n_components_ = n_components

    if not use_offset:
        n_stacks = int(np.ceil(n_stacks / 2))
        n_components = n_stacks*n_features
        if n_components*2 != n_components_:
            msg = "n_components is changed from {0} to {1}.".format(
                n_components_, n_components*2
            )
            msg += " When random_fourier=True and use_offset=False, "
            msg += " n_components should be larger than 2*n_features."
            warnings.warn(msg)
            n_components_ = n_components * 2

    if sigma == 'auto':
        gamma = 1.0 / n_features
    else:
        gamma = sigma
    size = (n_features, n_features)

    distribution = _get_random_matrix('gaussian')
    
    random_weights_ = []
    for _ in range(n_stacks):
        W = distribution(random_state, size)
        S = np.diag(chi.rvs(df=n_features, size=n_features,
                            random_state=random_state))
        SQ, _ = qr_multiply(W, S)
        random_weights_ += [SQ]

    random_weights_ = np.vstack(random_weights_)#.T
    random_offset_ = None
    random_weights_ *= sqrt(2*gamma)
    if use_offset:
        random_offset_ = random_state.uniform(
                0, 2*np.pi, size=n_components
            )
    return torch.from_numpy(random_weights_).to(device).float(),torch.from_numpy(random_offset_).to(device).float()
    
def create_random_fourier_features(X, W, offset):
    """
    Creates Gaussian random features using PyTorch, with updated input shapes.
    Parameters:
    - X: The input data, PyTorch tensor (d, N).
    - D: The number of random features to generate.
    - sigma: The standard deviation of the Gaussian distribution.    
    Returns:
    - Z: The generated random Fourier features, PyTorch tensor.
    """
    # sample weights from a Gaussian distribution
    #W = torch.randn(X.shape[1], D) * sigma
    D = W.shape[1]
    # Z = torch.sqrt(torch.tensor(1/D)) * torch.cat([torch.sin(torch.mm(X, W)), torch.cos(torch.mm(X, W))], dim=1)
    Z = torch.sqrt(torch.tensor(2/D)) * torch.cos(torch.mm(X, W.T)+offset) # out is (N, D)
    return Z

def generate_random_matrix(d, D, device, sigma=1.0):
    """
    Generates a random matrix for the random Fourier features.
    
    Parameters:
    - d: The input dimension.
    - D: The number of random features to generate.
    - sigma: The standard deviation of the Gaussian distribution.
    Returns:
    - W: The generated random matrix.
    - b: offset for the random features
    """
    W = torch.randn(D, d) * sigma
    b = torch.rand(1, D)*2*torch.pi
    W = W.to(device).float()#.half()
    b = b.to(device).float()#.half()
    return W, b