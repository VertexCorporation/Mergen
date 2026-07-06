"""
╔══════════════════════════════════════════════════════════════════════╗
║              MERGEN — HYBRID HEBBIAN LEARNER v2.0                    ║
║         Three-Factor Biologically-Grounded Learning Engine           ║
║                                                                      ║
║  "Zeka statik bir haritalama fonksiyonu değil,                       ║
║   yaşayan ve ritmik bir süreçtir."                                   ║
║                                                                      ║
║  Author:  Vertex Corporation — Mergen Project                        ║
║  License: Apache-2.0                                                 ║
║  Compat:  gradients.py | stdp.py | rl_agent.py                       ║
╚══════════════════════════════════════════════════════════════════════╝

15 BIOLOGICAL PRINCIPLES IMPLEMENTED:
  1.  Continuous Time (dt)           — Not tokens, real temporal flow
  2.  Pre-Synaptic Traces            — Input echo / residual calcium
  3.  Post-Synaptic Traces           — Output echo / back-propagating AP
  4.  STDP (LTP + LTD)              — Timing-based Hebbian plasticity
  5.  Eligibility Traces             — Synaptic tagging for delayed reward
  6.  Dopamine Gating                — Global neuromodulatory validation
  7.  RPE (Reward Prediction Error)  — Surprise-driven learning signal
  8.  Three-Factor Rule              — Dw = n * RPE * eligibility
  9.  Homeostatic Synaptic Scaling   — TNF-a mediated receptor scaling
  10. Soft-Bounding                  — Physical synapse saturation limits
  11. Surrogate Gradient Compat.     — Coexistence with autograd
  12. Sparse Computation             — Compute only when spikes occur
  13. Exponential Trace Decay        — Controlled forgetting via tau
  14. Adaptive Learning Rate (n)     — Stubbornness vs. flexibility
  15. Cognitive Emergence            — Unprogrammed behavior from dynamics
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .gradients import SurrogateSpike, SpikingActivation
from .stdp import STDPMechanism
from .rl_agent import DopamineModulator


# Number of special tokens + punctuation at the start of vocabulary
# 6 special (<bos>, <eos>, <pad>, <unk>, <sep>, <cls>) + 15 punctuation = 21
NUM_MASKED_TOKENS = 21


class HybridHebbianLearner(nn.Module):

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        # PRINCIPLE 1: Continuous Time
        dt: float = 1.0,
        # PRINCIPLE 2/3: Trace Time Constants
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        # PRINCIPLE 5: Eligibility Trace Decay
        tau_eligibility: float = 25.0,
        # PRINCIPLE 14: Learning Rate Amplitudes
        A_ltp: float = 0.005,
        A_ltd: float = 0.003,
        # PRINCIPLE 10: Soft-Bounding
        w_max: float = 1.0,
        w_min: float = 0.0,
        # PRINCIPLE 9: Homeostatic Scaling
        target_input_sum: float = 5.0,
        scaling_speed: float = 0.01,
        target_firing_rate: float = 0.1,
        # PRINCIPLE 6/7: Dopamine / RL
        gamma: float = 0.99,
        lr_critic: float = 0.1,
        dopamine_threshold: float = 0.01,
        # PRINCIPLE 11: Spike Threshold
        spike_threshold: float = 1.0,
        # PRINCIPLE 16: Lateral Inhibition (k-WTA)
        lateral_k: int = 10,
        # PRINCIPLE 13: EMA Decay
        ema_decay: float = 0.99,
        # PRINCIPLE 12: Device
        device: str = 'cpu',
    ):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.device = device
        self.dt = dt
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.tau_eligibility = tau_eligibility
        self.A_ltp = A_ltp
        self.A_ltd = A_ltd
        self.w_max = w_max
        self.w_min = w_min
        self.target_input_sum = target_input_sum
        self.scaling_speed = scaling_speed
        self.target_firing_rate = target_firing_rate
        self.dopamine_threshold = dopamine_threshold
        self.ema_decay = ema_decay
        self.lateral_k = lateral_k

        # Precomputed decay factors
        self._decay_pre = 1.0 - (dt / tau_pre)
        self._decay_post = 1.0 - (dt / tau_post)
        self._decay_elig = 1.0 - (dt / tau_eligibility)

        # Synaptic weights
        self.weights = nn.Parameter(torch.rand(n_pre, n_post, device=device) * 0.3)

        # Dynamic state buffers
        self.register_buffer('trace_pre', torch.zeros(n_pre, device=device))
        self.register_buffer('trace_post', torch.zeros(n_post, device=device))
        self.register_buffer('eligibility', torch.zeros(n_pre, n_post, device=device))
        self.register_buffer('firing_rate_ema', torch.zeros(n_post, device=device))

        # Mergen components
        self.spike_fn = SpikingActivation(threshold=spike_threshold)
        self.stdp = STDPMechanism(learning_rate=A_ltp, tau_trace=tau_pre, dt=dt)
        self.dopamine = DopamineModulator(gamma=gamma, lr_critic=lr_critic)

        # Telemetry
        self._step_count = 0
        self._da_event_count = 0
        self._last_rpe = 0.0
        self._last_ltp_mag = 0.0
        self._last_ltd_mag = 0.0
        self._last_delta_w = 0.0
        self._last_sparsity = 0.0

    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """
        PRINCIPLE 11+12+16: Surrogate gradient forward pass
        with lateral inhibition.

        Pipeline:
          1. Compute membrane potentials via matmul
          2. Suppress special tokens (structural guard)
          3. Apply spike activation (Heaviside threshold)
          4. Lateral Inhibition (k-WTA): only the top-K
             strongest post-synaptic neurons survive.
             All others are silenced, enforcing sparse coding.
        """
        sq = pre_spikes.dim() == 1
        if sq:
            pre_spikes = pre_spikes.unsqueeze(0)

        membrane = torch.matmul(pre_spikes, self.weights)

        # Structural guard: special tokens must never fire
        if membrane.shape[-1] > NUM_MASKED_TOKENS:
            membrane[..., :NUM_MASKED_TOKENS] = -1e9

        # Spike activation (Heaviside with surrogate gradient)
        out = self.spike_fn(membrane)

        # ── LATERAL INHIBITION (k-Winners-Take-All) ──
        # After spiking, only the K neurons with the highest
        # membrane potential are allowed to remain active.
        # This mimics inhibitory interneurons in biological cortex.
        n_active = (out > 0).sum(dim=-1)
        if (n_active > self.lateral_k).any():
            # Use membrane potential as competition criterion
            kth_val, _ = torch.kthvalue(
                membrane, membrane.shape[-1] - self.lateral_k, dim=-1
            )
            inhibition_mask = (membrane >= kth_val.unsqueeze(-1)).float()
            out = out * inhibition_mask

        return out.squeeze(0) if sq else out

    @torch.no_grad()
    def update_traces(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> Dict[str, float]:
        """PRINCIPLES 1-5, 10, 12, 13: Local Hebbian trace update."""
        self._step_count += 1
        if pre_spikes.dim() > 1: pre_spikes = pre_spikes.mean(0)
        if post_spikes.dim() > 1: post_spikes = post_spikes.mean(0)

        # PRINCIPLE 2/3: Decay + spike
        self.trace_pre.mul_(self._decay_pre).add_(pre_spikes)
        self.trace_post.mul_(self._decay_post).add_(post_spikes)

        # PRINCIPLE 4+10: Soft-bounded STDP
        soft_ltp = (self.w_max - self.weights.data).clamp_(min=0.0)
        soft_ltd = (self.weights.data - self.w_min).clamp_(min=0.0)
        raw_ltp = torch.outer(self.trace_pre, post_spikes)
        raw_ltd = torch.outer(pre_spikes, self.trace_post)
        stdp_signal = self.A_ltp * raw_ltp * soft_ltp - self.A_ltd * raw_ltd * soft_ltd

        self._last_ltp_mag = (self.A_ltp * raw_ltp * soft_ltp).abs().mean().item()
        self._last_ltd_mag = (self.A_ltd * raw_ltd * soft_ltd).abs().mean().item()

        # PRINCIPLE 5: Eligibility accumulation
        self.eligibility.mul_(self._decay_elig).add_(stdp_signal)

        # Firing rate EMA
        self.firing_rate_ema.mul_(self.ema_decay).add_((1.0 - self.ema_decay) * post_spikes)
        self._last_sparsity = 1.0 - ((pre_spikes.sum() + post_spikes.sum()).item() / (self.n_pre + self.n_post))

        return {'ltp': self._last_ltp_mag, 'ltd': self._last_ltd_mag, 'elig': self.eligibility.abs().sum().item()}

    @torch.no_grad()
    def apply_dopamine(self, reward: float, new_value_estimate: Optional[float] = None) -> torch.Tensor:
        """PRINCIPLES 6-9: Dopamine-gated weight update."""
        self._da_event_count += 1
        rpe = self.dopamine.compute_rpe(reward=reward, new_value_estimate=new_value_estimate)
        self._last_rpe = rpe

        if abs(rpe) < self.dopamine_threshold:
            return torch.zeros_like(self.weights.data)

        delta_w = self.dopamine.modulate_gradients(gradients=self.eligibility, rpe=rpe)
        self.weights.data.add_(delta_w)
        self._last_delta_w = delta_w.abs().mean().item()
        self.eligibility.mul_(0.1)
        self._homeostatic_normalization()
        return delta_w

    @torch.no_grad()
    def _homeostatic_normalization(self) -> None:
        """PRINCIPLE 9: Three-mechanism homeostatic regulation."""
        W = self.weights.data
        W.clamp_(min=self.w_min, max=self.w_max)

        col_sums = W.sum(dim=0)
        over = col_sums > self.target_input_sum
        if over.any():
            ideal = self.target_input_sum / col_sums.clamp(min=1e-8)
            scale = torch.where(over, 1.0 + self.scaling_speed * (ideal - 1.0), torch.ones_like(col_sums))
            scale.clamp_(min=0.5, max=1.5)
            W.mul_(scale.unsqueeze(0))

        if self.firing_rate_ema.sum() > 0:
            rr = (self.target_firing_rate / (self.firing_rate_ema + 1e-8)).clamp_(0.85, 1.15)
            W.mul_((1.0 + self.scaling_speed * (rr - 1.0)).unsqueeze(0))

        W.clamp_(min=self.w_min, max=self.w_max)

    def learning_step(self, pre_spikes, post_spikes, reward, new_value_estimate=None) -> Dict[str, float]:
        """Complete learning step: traces + dopamine + homeostasis."""
        self.update_traces(pre_spikes, post_spikes)
        self.apply_dopamine(reward, new_value_estimate)
        return self.get_telemetry()

    @torch.no_grad()
    def spreading_activation(self, initial_activation: torch.Tensor, steps: int = 3, alpha: float = 0.85) -> torch.Tensor:
        """
        PRINCIPLE 15: Cognitive Emergence via Spreading Activation.
        Performs search-free deductive reasoning by propagating neural activity 
        through the dynamically learned conceptual graph (W^T @ W).
        """
        W = self.weights.data

        # L2 Normalize the columns (concepts) to get cosine similarities
        norms = torch.norm(W, p=2, dim=0, keepdim=True).clamp_min(1e-8)
        W_norm = W / norms

        # Concept correlation matrix (n_post, n_post)
        T = torch.matmul(W_norm.t(), W_norm)

        # Ensure non-negative and zero out self-loops
        T = torch.relu(T)
        T.fill_diagonal_(0.0)

        # Row-normalize to create a stochastic transition matrix
        row_sums = T.sum(dim=1, keepdim=True).clamp_min(1e-8)
        T_trans = T / row_sums

        x = initial_activation.clone()
        x0 = initial_activation.clone()

        for _ in range(steps):
            x = alpha * torch.matmul(T_trans.t(), x) + (1 - alpha) * x0

        return x

    def get_telemetry(self) -> Dict[str, float]:
        W = self.weights.data
        return {
            'step': self._step_count, 'da_events': self._da_event_count,
            'rpe': self._last_rpe, 'ltp': self._last_ltp_mag, 'ltd': self._last_ltd_mag,
            'delta_w': self._last_delta_w, 'w_mean': W.mean().item(), 'w_std': W.std().item(),
            'w_sparsity': (W < 0.01).float().mean().item(),
            'elig_energy': self.eligibility.abs().sum().item(),
            'rate_mean': self.firing_rate_ema.mean().item(),
            'rate_std': self.firing_rate_ema.std().item(),
            'spike_sparsity': self._last_sparsity,
        }

    def reset_traces(self):
        self.trace_pre.zero_(); self.trace_post.zero_()
        self.eligibility.zero_(); self.firing_rate_ema.zero_()
        self.dopamine.value_estimate = 0.0

    def reset_all(self):
        self.reset_traces()
        with torch.no_grad(): self.weights.data.uniform_(0, 0.3)
        self._step_count = 0; self._da_event_count = 0

    def __repr__(self):
        W = self.weights.data
        return (f"HybridHebbianLearner({self.n_pre}->{self.n_post}, "
                f"{self.n_pre*self.n_post:,} synapses, "
                f"W={W.mean():.3f}+/-{W.std():.3f}, "
                f"{self._step_count} steps)")


if __name__ == "__main__":
    print("=" * 60)
    print("MERGEN Hybrid Hebbian Learner v2.0 — Test")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learner = HybridHebbianLearner(
        n_pre=128, n_post=64, dt=1.0, tau_pre=20.0, tau_post=20.0,
        tau_eligibility=30.0, A_ltp=0.005, A_ltd=0.003, w_max=1.0,
        target_input_sum=8.0, target_firing_rate=0.12, gamma=0.95,
        lr_critic=0.1, dopamine_threshold=0.02, device=device,
    )
    print(f"\n{learner}\nDevice: {device}\n")

    for ep in range(8):
        learner.reset_traces()
        for t in range(10):
            pre = torch.zeros(128, device=device)
            pre[:10] = (torch.rand(10, device=device) < 0.2).float()
            pre[10:20] = (torch.rand(10, device=device) < 0.3).float()
            post = learner.forward(pre)
            learner.update_traces(pre, post)

        correct = post[5].item() > 0.5
        learner.apply_dopamine(reward=1.0 if correct else -0.3)
        t = learner.get_telemetry()
        print(f"  Ep {ep+1} | RPE:{t['rpe']:+.3f} | LTP:{t['ltp']:.4f} | "
              f"LTD:{t['ltd']:.4f} | dW:{t['delta_w']:.4f} | "
              f"W:{t['w_mean']:.3f} | {'OK' if correct else '--'}")

    print(f"\nFinal: {learner}")
    print("=" * 60)
