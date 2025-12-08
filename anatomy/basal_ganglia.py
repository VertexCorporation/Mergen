import torch
import torch.nn as nn

class BasalGanglia(nn.Module):
    """
    Action Selection & Gating Mechanism.
    
    Biophysics:
    - Receives inputs from Cortex.
    - Calculates 'Value' (Expected Reward).
    - Performs 'Winner-Take-All' to select one action.
    - Modulated by Dopamine (Reward Prediction Error).
    """
    def __init__(self, n_inputs, n_actions, device='cpu'):
        super().__init__()
        self.device = device
        
        # Weights determining value of actions
        # Plasticity here is driven by Dopamine (RL)
        self.weights = torch.randn((n_inputs, n_actions), device=device) * 0.01
        
        # Gating Threshold
        self.threshold = 0.5

    def forward(self, cortical_input):
        """
        Decide which action to take.
        """
        # Calculate Action Values (Q-values)
        action_values = torch.matmul(cortical_input.flatten(), self.weights)
        
        # Softmax selection (Exploration vs Exploitation)
        action_probs = torch.softmax(action_values, dim=0)
        
        # Select the winner
        winner_idx = torch.argmax(action_probs)
        winner_val = action_values[winner_idx]
        
        # Gating: Only act if confidence is high enough
        gating_signal = 1.0 if winner_val > self.threshold else 0.0
        
        return winner_idx, gating_signal, action_values

    def learn(self, input_trace, action_idx, reward, prediction_error):
        """
        Dopaminergic Learning (Reinforcement Learning).
        
        If Reward > Expected: Strengthen connection (LTP).
        If Reward < Expected: Weaken connection (LTD).
        """
        learning_rate = 0.1
        
        # Create a mask for the chosen action
        # We only update the weights responsible for the chosen action
        d_weights = input_trace.unsqueeze(1) * prediction_error
        
        # Update only the column of the selected action
        self.weights[:, action_idx] += learning_rate * d_weights.squeeze()