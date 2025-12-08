import torch

class STDPMechanism:
    """
    Spike-Timing-Dependent Plasticity (Hebbian Learning).
    
    Function:
    Unsupervised learning based on causal relationships between neurons.
    If Input causes Output -> Strengthen weight.
    If Output happens before Input -> Weaken weight.
    """
    def __init__(self, learning_rate=0.001, tau_trace=20.0, dt=1.0):
        self.lr = learning_rate
        self.tau_trace = tau_trace
        self.dt = dt
    
    def update_weights(self, weights, pre_spikes, post_spikes, pre_trace, post_trace):
        """
        Adjusts synaptic weights based on spike timing.
        
        Args:
            weights: The weight matrix (Pre x Post).
            pre_spikes: Activity of input neurons (Batch x Pre).
            post_spikes: Activity of output neurons (Batch x Post).
            pre_trace: Filtered history of input activity.
            post_trace: Filtered history of output activity.
        """
        # 1. Long-Term Potentiation (LTP)
        # Event: Post-synaptic neuron fires.
        # Logic: Check if Pre-synaptic neuron fired recently (pre_trace).
        # Formula: dW += lr * (pre_trace * post_spike)
        
        # We use batch matrix multiplication for efficiency
        # (Pre_Trace.T @ Post_Spike)
        delta_w_ltp = torch.matmul(pre_trace.t(), post_spikes)
        
        # 2. Long-Term Depression (LTD)
        # Event: Pre-synaptic neuron fires.
        # Logic: Check if Post-synaptic neuron fired recently (post_trace) 
        # (meaning Pre was too late to cause it).
        # Formula: dW -= lr * (pre_spike * post_trace)
        
        delta_w_ltd = torch.matmul(pre_spikes.t(), post_trace)
        
        # 3. Combine and Update
        # Soft-bound normalization (prevent weights from exploding)
        # If weight is high, LTP is weaker. If low, LTD is weaker.
        w_max = 1.0
        delta_w = (self.lr * delta_w_ltp * (w_max - weights)) - (self.lr * delta_w_ltd * weights)
        
        return delta_w

    def update_trace(self, trace, spikes):
        """
        Updates the memory trace of spiking activity.
        dx/dt = -x/tau + spike
        """
        decay = 1.0 - (self.dt / self.tau_trace)
        return (trace * decay) + spikes