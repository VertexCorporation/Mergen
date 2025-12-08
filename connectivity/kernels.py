import torch
import math

class MexicanHatKernel:
    """
    Generates a 2D Difference-of-Gaussians (DoG) kernel.
    Essential for creating 'Bubbles of Activity' in the cortex.
    """
    @staticmethod
    def create(size_h, size_w, exc_sigma=2.0, inh_sigma=6.0, exc_gain=1.5, inh_gain=1.2, device='cpu'):
        """
        Creates a normalized Mexican Hat Kernel tensor.
        """
        # Create coordinate grid
        y = torch.arange(size_h, device=device) - size_h // 2
        x = torch.arange(size_w, device=device) - size_w // 2
        Y, X = torch.meshgrid(y, x, indexing='ij')
        
        # Calculate radius squared
        R2 = X**2 + Y**2
        
        # Excitation Gaussian
        exc = exc_gain * torch.exp(-R2 / (2 * exc_sigma**2))
        
        # Inhibition Gaussian
        inh = inh_gain * torch.exp(-R2 / (2 * inh_sigma**2))
        
        # Combine
        kernel = exc - inh
        
        # Zero-Sum Balancing (CRITICAL for stability)
        # We ensure the total energy of the kernel is 0.
        # This prevents the brain from exploding with energy over time.
        kernel -= kernel.mean()
        
        return kernel

class GaborKernel:
    """
    Generates oriented filters (like in Visual Cortex V1).
    Used for detecting edges/lines in sensory input.
    """
    @staticmethod
    def create(size, sigma, theta, lambda_, gamma, psi, device='cpu'):
        # Implementation of 2D Gabor formula
        sigma_x = sigma
        sigma_y = sigma / gamma
        
        y, x = torch.meshgrid(
            torch.linspace(-size//2, size//2, size, device=device),
            torch.linspace(-size//2, size//2, size, device=device),
            indexing='ij'
        )
        
        # Rotation
        x_theta = x * math.cos(theta) + y * math.sin(theta)
        y_theta = -x * math.sin(theta) + y * math.cos(theta)
        
        gb = torch.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * \
             torch.cos(2 * math.pi * x_theta / lambda_ + psi)
             
        return gb