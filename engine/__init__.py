"""
MERGEN ENGINE MODULE
The physical substrate of the Cognitive Engine.

This module handles:
1. Tensor Operations (GPU-accelerated math)
2. Numerical Integration (Making time flow)
3. Delay Management (The speed of light in the brain)
"""

from .tensor_ops import fft_convolve2d, normalize_tensor
from .integrators import EulerSolver, RungeKutta4Solver
from .delays import DelayBuffer