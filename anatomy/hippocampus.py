import torch
import torch.nn as nn
import torch.nn.functional as F

class Hippocampus(nn.Module):
    """
    MERGEN - Fast Episodic Memory (Vector Associative Memory)
    
    - Key memory  : sensory trace
    - Value memory: 1-hot class vector or latent concept vector
    - Retrieval   : cosine similarity + adaptive threshold
    """

    def __init__(self, input_dim, value_dim, memory_capacity,
                 threshold=0.85, device='cpu'):
        super().__init__()
        self.capacity = memory_capacity
        self.input_dim = input_dim
        self.value_dim = value_dim
        self.device = device
        
        self.keys = torch.zeros((memory_capacity, input_dim), device=device)
        self.values = torch.zeros((memory_capacity, value_dim), device=device)

        self.size = 0
        self.ptr = 0

        self.base_threshold = threshold
        self.dynamic_gain = 0.05

    # ----------------------------------------------------------
    # STORE (One-shot learning)
    # ----------------------------------------------------------
    def store(self, key_pattern, value_pattern):
        key_norm = F.normalize(key_pattern.flatten(), dim=0)
        val_norm = F.normalize(value_pattern.flatten(), dim=0)

        self.keys[self.ptr] = key_norm
        self.values[self.ptr] = val_norm

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ----------------------------------------------------------
    # RETRIEVE (Cosine similarity + pattern completion)
    # ----------------------------------------------------------
    def retrieve(self, query_pattern):
        if self.size == 0:
            return None, 0.0

        q = F.normalize(query_pattern.flatten(), dim=0)

        keys = self.keys[:self.size]  # (M, D)

        # GPU optimized: (M, D) • (D,) → (M,)
        sims = torch.matmul(keys, q)

        best_sim, idx = sims.max(dim=0)

        threshold = self.base_threshold + self.dynamic_gain * (self.size / self.capacity)

        if best_sim >= threshold:
            weights = torch.softmax(sims * 10, dim=0)
            recall = torch.matmul(weights.unsqueeze(0), self.values[:self.size]).squeeze(0)
            return recall, best_sim.item()

        return None, best_sim.item()