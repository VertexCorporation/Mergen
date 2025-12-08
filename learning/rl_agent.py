import torch

class DopamineModulator:
    """
    Reinforcement Learning Adapter (Actor-Critic style logic).
    
    Function:
    Calculates the Reward Prediction Error (RPE).
    This signal modulates ALL plasticity in the brain.
    
    "Dopamine is the teacher of the brain."
    """
    def __init__(self, gamma=0.99, lr_critic=0.1):
        self.gamma = gamma # Discount factor for future rewards
        self.value_estimate = 0.0 # Expected reward
        self.lr_critic = lr_critic # How fast we update expectation

    def compute_rpe(self, reward, new_value_estimate=None):
        """
        Calculates: Surprise = Actual - Expected
        """
        # If we have a new state value estimate (Critic), use Temporal Difference
        if new_value_estimate is not None:
            target = reward + self.gamma * new_value_estimate
        else:
            target = reward
            
        # RPE (Dopamine Signal)
        rpe = target - self.value_estimate
        
        # Update internal expectation (The Critic learns)
        self.value_estimate += self.lr_critic * rpe
        
        return rpe

    def modulate_gradients(self, gradients, rpe):
        """
        Neuromodulation:
        Scales the gradients based on the Dopamine signal.
        
        If RPE > 0 (Good surprise): Amplify learning (LTP).
        If RPE < 0 (Bad surprise): Reverse/Suppress learning (LTD).
        """
        return gradients * rpe