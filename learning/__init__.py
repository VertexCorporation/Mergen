"""
MERGEN LEARNING MODULE
The Hybrid Learning Engine.

Combines three distinct learning paradigms:
1. Surrogate Gradients: Enables PyTorch to optimize Spiking Networks (Supervised).
2. STDP (Spike-Timing-Dependent Plasticity): Biological, unsupervised association.
3. RL (Reinforcement Learning): Dopamine-based reward modulation.
"""

from .gradients import SurrogateSpike
from .stdp import STDPMechanism
from .rl_agent import DopamineModulator