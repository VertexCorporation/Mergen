import torch
import torch.nn as nn

from learning.hebbian_engine import HybridHebbianLearner


class IntegratedBrainWrapper(nn.Module):
    """
    MWCC Milestone 1: Wrapper Skeleton.
    
    Provides the exact interface and serialization compatibility 
    required by LimbicExecutiveLayer, delegating the state to the 
    internal HybridHebbianLearner.
    
    Biological processing, Cortex, and Workspace are NOT yet implemented.
    """
    def __init__(self, n_neurons: int, device: str = 'cpu'):
        super().__init__()
        self.n_neurons = n_neurons
        
        # Instantiate the Hebbian learner directly as the sole component for now.
        self.learner = HybridHebbianLearner(
            n_pre=n_neurons,
            n_post=n_neurons,
            device=device
        )
        
    # ==========================================
    # RUNTIME INTERFACE (DUMMY/PASSTHROUGH)
    # ==========================================
    def forward(self, pre_1d: torch.Tensor) -> torch.Tensor:
        """
        Placeholder forward pass. 
        Will eventually route through Cortex -> Workspace -> Cortex.
        Currently delegates directly to the learner.
        """
        return self.learner.forward(pre_1d)

    def update_traces(self, pre_1d: torch.Tensor, post_1d: torch.Tensor):
        self.learner.update_traces(pre_1d, post_1d)

    def apply_dopamine(self, reward: float):
        self.learner.apply_dopamine(reward)

    # ==========================================
    # PASSTHROUGH PROPERTIES FOR LIMBIC LAYER .mx
    # ==========================================
    
    @property
    def device(self):
        return self.learner.device
        
    @property
    def n_pre(self):
        return self.learner.n_pre
        
    @property
    def n_post(self):
        return self.learner.n_post

    @property
    def weights(self):
        # Getter only because load_state modifies .data in place:
        # self.engine.weights.data = w
        return self.learner.weights

    @property
    def eligibility(self):
        return self.learner.eligibility
        
    @eligibility.setter
    def eligibility(self, value):
        self.learner.eligibility = value

    @property
    def trace_pre(self):
        return self.learner.trace_pre
        
    @trace_pre.setter
    def trace_pre(self, value):
        self.learner.trace_pre = value

    @property
    def trace_post(self):
        return self.learner.trace_post
        
    @trace_post.setter
    def trace_post(self, value):
        self.learner.trace_post = value

    @property
    def firing_rate_ema(self):
        # load_state does not restore firing_rate_ema, only save_state reads it.
        return self.learner.firing_rate_ema

    @property
    def _step_count(self):
        return self.learner._step_count
        
    @_step_count.setter
    def _step_count(self, value):
        self.learner._step_count = value

    @property
    def _da_event_count(self):
        return self.learner._da_event_count
        
    @_da_event_count.setter
    def _da_event_count(self, value):
        self.learner._da_event_count = value
