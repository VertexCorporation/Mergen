import torch

class HomeostaticRegulator:
    """
    Maintains the stability of neural activity.
    
    Mechanism:
    Tracks the 'Running Average' of firing rates.
    If a neuron fires too much -> Raise Threshold (Make it harder to fire).
    If a neuron is too silent -> Lower Threshold (Make it easier to fire).
    
    Goal:
    Keep the network in a 'Critical State' (Edge of Chaos), where information processing is optimal.
    """
    def __init__(self, n_neurons, target_rate=0.02, adjustment_speed=0.01, device='cpu'):
        """
        Args:
            target_rate: Desired firing probability (e.g., 2% of neurons active per step).
            adjustment_speed: How fast the threshold changes (Learning Rate for Homeostasis).
        """
        self.target_rate = target_rate
        self.alpha = adjustment_speed
        self.device = device
        
        # Moving average of activity
        self.activity_trace = torch.zeros(n_neurons, device=device)

    def update(self, current_spikes, current_thresholds):
        """
        Adjusts thresholds based on recent activity.
        
        Formula:
        d_Threshold = alpha * (Current_Activity - Target_Activity)
        """
        # Update trace (Low-pass filter of spikes)
        # trace = 0.95 * trace + 0.05 * spikes
        self.activity_trace = 0.95 * self.activity_trace + 0.05 * current_spikes
        
        # Calculate error (Too hot or Too cold?)
        rate_error = self.activity_trace - self.target_rate
        
        # Adjust threshold
        # If Error > 0 (Too active) -> Increase Threshold
        # If Error < 0 (Too silent) -> Decrease Threshold
        new_thresholds = current_thresholds + (self.alpha * rate_error)
        
        # Safety clamp to prevent negative or infinite thresholds
        return torch.clamp(new_thresholds, min=0.1, max=5.0)