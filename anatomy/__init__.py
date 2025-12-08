"""
MERGEN ANATOMY MODULE
Defines the biological components of the cognitive architecture.

- CorticalLayer: A 2D sheet of neurons with local field dynamics.
- Hippocampus: Fast episodic memory and pattern completion.
- BasalGanglia: Action selection and reinforcement learning gating.
- Cerebellum: Error correction and fine-motor tuning.
"""

from .cortical_sheet import CorticalLayer
from .hippocampus import Hippocampus
from .basal_ganglia import BasalGanglia
from .cerebellum import Cerebellum