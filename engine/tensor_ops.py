import torch
import torch.fft

def fft_convolve2d(input_field: torch.Tensor, kernel_fft: torch.Tensor) -> torch.Tensor:
    """
    Performs a 2D Circular Convolution using the Fast Fourier Transform (FFT).
    
    Why this is legendary:
    Standard convolution is O(N^2). This is O(N log N).
    This allows us to simulate MILLIONS of neurons interacting locally 
    (Mexican Hat interaction) without crashing the computer.
    
    Args:
        input_field: The current voltage state of the cortex (Batch, H, W).
        kernel_fft: The pre-computed FFT of the connectivity kernel.
        
    Returns:
        The synaptic input current resulting from local field interactions.
    """
    # 1. Convert input to Frequency Domain
    input_fft = torch.fft.rfft2(input_field)
    
    # 2. Multiply in Frequency Domain (This equals convolution in Space Domain)
    # We assume the kernel is already in the frequency domain for speed.
    convolved_fft = input_fft * kernel_fft
    
    # 3. Convert back to Space Domain (Real numbers)
    output_field = torch.fft.irfft2(convolved_fft, s=input_field.shape[-2:])
    
    return output_field

def normalize_tensor(x: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    """
    Keeps the brain energy within physical limits.
    Prevents "Runaway Excitation" (Epilepsy).
    """
    return torch.clamp(x, min_val, max_val)

def sparse_mask_projection(source: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, output_shape: tuple) -> torch.Tensor:
    """
    Handles Long-Range connections (The White Matter).
    Instead of a dense matrix, we use sparse gathering for global communication.
    
    Args:
        source: Activity of source neurons.
        indices: Connection map (Who connects to whom).
        values: Synaptic weights.
    """
    # Create a sparse tensor representing long-range tracts
    sparse_adj = torch.sparse_coo_tensor(indices, values, size=(source.shape[0], output_shape[0]))
    
    # Efficient Sparse Matrix-Vector Multiplication (SpMM)
    return torch.sparse.mm(sparse_adj.t(), source.unsqueeze(1)).squeeze(1)