import torch

def create_sparse_projection(n_src, n_dst, density=0.1, device='cpu'):
    """
    Creates a sparse weight matrix. 
    Most biological connections are sparse (density < 10%).
    """
    # Create random mask
    mask = torch.rand((n_src, n_dst), device=device) < density
    
    # Initialize weights (Xavier-like)
    weights = torch.randn((n_src, n_dst), device=device) * 0.05
    
    # Apply mask
    weights = weights * mask.float()
    
    return weights

def create_topological_projection(size, radius=2, device='cpu'):
    """
    Creates a 1-to-1 mapping with slight spread.
    Useful for connecting Layer 4 to Layer 2/3 directly above it.
    
    (Simplified Identity matrix with noise for now)
    """
    # For now, a simple Identity matrix is the best topological map
    # In V3.1 we can add Gaussian spread.
    return torch.eye(size, device=device)