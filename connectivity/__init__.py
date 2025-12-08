"""
MERGEN CONNECTIVITY MODULE
The wiring of the brain.

- Kernels: Define local interactions (Short-range).
- Projections: Define point-to-point wirings (Mid-range).
- GlobalWorkspace: Defines the attention-based router (Long-range).
"""

from .kernels import MexicanHatKernel, GaborKernel
from .projections import create_sparse_projection, create_topological_projection
from .global_workspace import GlobalWorkspace