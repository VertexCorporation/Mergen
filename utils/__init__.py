"""
MERGEN UTILS MODULE
Safety mechanisms and Data translation tools.

- Stability: Prevents runaway excitation (Epilepsy control).
- Encoder: Converts real-world data (Text/Numbers) into neural spikes.
"""

from .stability import HomeostaticRegulator
from .encoder import SpikeEncoder