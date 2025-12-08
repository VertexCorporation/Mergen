import torch
import torch.nn as nn

class GlobalWorkspace(nn.Module):
    """
    The 'Router' of the brain. Implements the Global Neuronal Workspace Theory.
    
    Mechanism:
    Instead of Dense N-to-N connections, we use a bottleneck:
    Cortex (N) -> Workspace (K) -> Cortex (N)
    
    This is computationally O(N*K) which is linear-ish, allowing massive scaling.
    """
    def __init__(self, input_dim, workspace_dim, output_dim, device='cpu'):
        super().__init__()
        self.device = device
        
        # 1. UPLINK (Compression): Cortex -> Workspace
        # Compresses massive cortical activity into 'Latent Tokens'
        self.W_up = nn.Linear(input_dim, workspace_dim, bias=False).to(device)
        
        # 2. LATERAL (Reasoning): Workspace -> Workspace
        # The workspace neurons talk to each other (Recurrent / Attention)
        # We use a simple recurrent matrix here to sustain thought.
        self.W_lat = nn.Linear(workspace_dim, workspace_dim, bias=False).to(device)
        
        # 3. DOWNLINK (Broadcast): Workspace -> Cortex
        # Spreads the selected information back to the brain
        self.W_down = nn.Linear(workspace_dim, output_dim, bias=False).to(device)
        
        # State
        self.state = torch.zeros(workspace_dim, device=device)
        
        # Initialization
        nn.init.orthogonal_(self.W_up.weight, gain=0.5)
        nn.init.orthogonal_(self.W_lat.weight, gain=0.9) # High gain to sustain memory
        nn.init.orthogonal_(self.W_down.weight, gain=0.5)

    def forward(self, cortical_input):
        """
        Args:
            cortical_input: Activity from sensory/motor areas (Batch, N)
            
        Returns:
            broadcast_signal: Feedback to the cortex (Batch, N)
        """
        # A. Compression (Bottom-Up)
        # Info enters the workspace
        incoming = self.W_up(cortical_input)
        
        # B. Ignition & Competition (Internal)
        # Combine new input with old state (Recurring thought)
        # Activation function is ReLU to ensure sparsity (Only strong ideas survive)
        internal_drive = incoming + self.W_lat(self.state)
        self.state = torch.relu(internal_drive)
        
        # Gating / Thresholding (Attention)
        # If activity is too low, kill it (Silence noise).
        # This creates the "Ignition" effect where only strong stimuli enter consciousness.
        mask = self.state.float()
        self.state = self.state * mask
        
        # C. Broadcast (Top-Down)
        # Send the processed thought back to the cortex
        broadcast = self.W_down(self.state)
        
        return broadcast