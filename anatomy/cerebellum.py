import torch
import torch.nn as nn

class Cerebellum(nn.Module):
    """
    Supervised Error Correction Module.
    
    Function:
    - Predicts the sensory consequence of a motor command.
    - If there is a mismatch (Error), it learns instantly to correct it.
    - Crucial for smooth motor control and cognitive precision.
    """
    def __init__(self, input_dim, output_dim, device='cpu'):
        super().__init__()
        self.device = device
        
        # The Cerebellum is essentially a massive perceptron 
        # that learns from error signals.
        self.weights = torch.zeros((input_dim, output_dim), device=device)
        self.learning_rate = 0.5 # Cerebellum learns FAST

    def forward(self, context_input):
        """
        Predict the outcome/correction based on context.
        """
        prediction = torch.matmul(context_input.flatten(), self.weights)
        return prediction

    def learn(self, context_input, error_signal):
        """
        LMS (Least Mean Squares) Learning.
        Also known as the Delta Rule.
        
        w += rate * error * input
        """
        # Outer product to update the weight matrix
        delta = self.learning_rate * torch.outer(context_input.flatten(), error_signal.flatten())
        self.weights += delta