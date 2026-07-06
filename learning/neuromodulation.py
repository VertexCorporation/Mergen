import torch
from typing import Dict

class NeuromodulationSystem:
    """
    Global Neuromodulation System for Mergen Brain.
    
    Manages 4 primary neuromodulators:
    - Dopamine (DA): Reward, motivation, learning rate.
    - Serotonin (5-HT): Inhibition, patience, mood.
    - Noradrenaline (NE): Attention span, urgency, arousal.
    - Acetylcholine (ACh): Memory encoding, novelty sensitivity.
    
    GPU-Safe Design:
    All modulators are stored as requires_grad=False tensors to prevent
    autograd graph memory leaks during continuous execution. Updates are
    performed strictly in-place.
    """
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Base (Homeostatic) Levels
        self.BASE_DA = 0.1
        self.BASE_5HT = 0.5
        self.BASE_NE = 0.2
        self.BASE_ACH = 0.3
        
        # Current Levels (Initialized to Base)
        # requires_grad=False is the default for torch.tensor, but we make it explicit
        self.da = torch.tensor([self.BASE_DA], device=self.device, dtype=torch.float32, requires_grad=False)
        self.serotonin = torch.tensor([self.BASE_5HT], device=self.device, dtype=torch.float32, requires_grad=False)
        self.noradrenaline = torch.tensor([self.BASE_NE], device=self.device, dtype=torch.float32, requires_grad=False)
        self.acetylcholine = torch.tensor([self.BASE_ACH], device=self.device, dtype=torch.float32, requires_grad=False)
        
        # Decay rates per tick (how fast they return to baseline)
        self.decay_rate = 0.05
        
    def apply_signal(self, da_delta: float = 0.0, ht_delta: float = 0.0, 
                     ne_delta: float = 0.0, ach_delta: float = 0.0):
        """
        Apply external signals to the neuromodulators.
        Uses in-place addition and clamping.
        """
        if da_delta != 0.0:
            self.da.add_(da_delta).clamp_(0.0, 1.0)
        if ht_delta != 0.0:
            self.serotonin.add_(ht_delta).clamp_(0.0, 1.0)
        if ne_delta != 0.0:
            self.noradrenaline.add_(ne_delta).clamp_(0.0, 1.0)
        if ach_delta != 0.0:
            self.acetylcholine.add_(ach_delta).clamp_(0.0, 1.0)

    def tick_homeostasis(self):
        """
        Pulls all neuromodulators back towards their base levels.
        Must use in-place operations exclusively.
        Formula: current += (base - current) * decay_rate
        """
        # We can perform the whole operation in place without intermediate tensors:
        # Since decay_rate and base are scalars, this is fully in-place.
        # But to avoid allocating a new tensor for `self.BASE - self.val`, we can do:
        # self.da = self.da + (BASE - self.da) * rate  -> this allocates!
        # To do it truly in-place:
        # self.da.lerp_(torch.tensor([self.BASE_DA], device=self.device), self.decay_rate)
        
        # lerp_ requires the end tensor to be on the same device.
        # Creating a scalar tensor every tick might allocate on CPU/GPU, but it's small.
        # However, a simpler way is just to manually do the math with floats since we have 1D scalar tensors,
        # but to keep it strictly in-place on the tensor:
        
        self.da.mul_(1.0 - self.decay_rate).add_(self.BASE_DA * self.decay_rate).clamp_(0.0, 1.0)
        self.serotonin.mul_(1.0 - self.decay_rate).add_(self.BASE_5HT * self.decay_rate).clamp_(0.0, 1.0)
        self.noradrenaline.mul_(1.0 - self.decay_rate).add_(self.BASE_NE * self.decay_rate).clamp_(0.0, 1.0)
        self.acetylcholine.mul_(1.0 - self.decay_rate).add_(self.BASE_ACH * self.decay_rate).clamp_(0.0, 1.0)

    def get_levels(self) -> Dict[str, float]:
        """Returns current levels as standard Python floats (useful for logging/saving)."""
        return {
            'DA': self.da.item(),
            '5-HT': self.serotonin.item(),
            'NE': self.noradrenaline.item(),
            'ACh': self.acetylcholine.item()
        }

    def set_levels(self, levels: Dict[str, float]):
        """Sets levels from a dictionary (useful for loading state). Uses in-place modification."""
        if not levels: return
        if 'DA' in levels:
            self.da.fill_(levels['DA']).clamp_(0.0, 1.0)
        if '5-HT' in levels:
            self.serotonin.fill_(levels['5-HT']).clamp_(0.0, 1.0)
        if 'NE' in levels:
            self.noradrenaline.fill_(levels['NE']).clamp_(0.0, 1.0)
        if 'ACh' in levels:
            self.acetylcholine.fill_(levels['ACh']).clamp_(0.0, 1.0)
