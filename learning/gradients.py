import torch
import torch.nn as nn

class SurrogateSpike(torch.autograd.Function):
    """
    The Bridge between Digital Optimization and Biological Spiking.
    
    Problem: Spikes are discrete (0 or 1). Gradients cannot flow through them.
    Solution: We use a 'Surrogate Gradient' during backpropagation.
    
    Forward pass:  Heaviside step function (Hard spike).
    Backward pass: Fast Sigmoid derivative (Soft curve).
    """
    
    @staticmethod
    def forward(ctx, input_voltage, threshold=1.0):
        # Save input for backward pass
        ctx.save_for_backward(input_voltage)
        ctx.threshold = threshold
        
        # Hard Spike: 1 if V > Threshold, else 0
        return (input_voltage > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved input
        input_voltage, = ctx.saved_tensors
        threshold = ctx.threshold
        
        # --- THE TRICK ---
        # Instead of the true derivative of a step function (which is 0 or infinity),
        # we calculate the derivative of a Sigmoid function centered at the threshold.
        # This allows the "Error Signal" to flow back through time.
        
        # Fast Sigmoid Derivative approximation
        # steepness (beta) controls how picky the gradient is.
        beta = 10.0 
        grad_input = grad_output / (1 + beta * torch.abs(input_voltage - threshold)).pow(2)
        
        return grad_input, None

class SpikingActivation(nn.Module):
    """
    Layer wrapper to easily use Surrogate Spikes in the model.
    """
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, x):
        return SurrogateSpike.apply(x, self.threshold)