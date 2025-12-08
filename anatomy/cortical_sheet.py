# anatomy/cortical_sheet.py
import torch
import torch.nn as nn

from engine.tensor_ops import fft_convolve2d
import config as cfg


class CorticalLayer(nn.Module):
    """
    A 2D sheet of spiking neurons with local field connectivity.

    Biophysics (simplified LIF with adaptation):
    - v: membrane potential
    - a: adaptation / fatigue
    - theta: dynamic firing threshold (homeostasis)
    - spikes: 0/1 output

    Lateral connectivity is implemented via a precomputed
    Mexican-hat kernel in the frequency domain (FFT).
    """

    def __init__(
        self,
        height: int,
        width: int,
        dt: float,
        kernel_tensor: torch.Tensor,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.height = height
        self.width = width
        self.n_neurons = height * width
        self.dt = dt
        self.device = device

        # --- 1. NEURAL STATE (all on the right device) ---
        self.v = torch.zeros((height, width), device=device, dtype=cfg.DTYPE)
        self.a = torch.zeros((height, width), device=device, dtype=cfg.DTYPE)

        # Start thresholds from config (e.g., very low to encourage activity)
        self.theta = torch.ones((height, width), device=device, dtype=cfg.DTYPE) * cfg.THETA_BASE

        self.spikes = torch.zeros((height, width), device=device, dtype=cfg.DTYPE)

        # Refractory bookkeeping (in ms / steps)
        self.refractory_period = 2.0  # can be exposed to config later
        self.refractory_count = torch.zeros_like(self.v)

        # --- 2. CONNECTIVITY (Mexican hat in frequency domain) ---
        self.kernel_fft = torch.fft.rfft2(kernel_tensor.to(device))

        # --- 3. DYNAMICS PARAMETERS FROM CONFIG ---
        self.tau_mem = cfg.TAU_MEM       # membrane time constant
        self.tau_adapt = cfg.TAU_ADAPT   # adaptation time constant
        self.v_reset = cfg.V_RESET

    # --------------------------------------------------------- #
    # FORWARD STEP
    # --------------------------------------------------------- #

    def forward(self, external_input: torch.Tensor) -> torch.Tensor:
        """
        Runs one time step (dt) of cortical dynamics.

        Args:
            external_input: Tensor (H, W) with incoming current.

        Returns:
            spikes: Tensor (H, W) with 0/1 spiking pattern.
        """
        # Ensure shape is (H, W)
        x = external_input.view(self.height, self.width)

        # A) Lateral field via FFT convolution
        i_local = fft_convolve2d(self.spikes, self.kernel_fft)

        # B) Total input current: external + local - adaptation
        total_input = x + i_local - (1.5 * self.a)

        # C) Membrane integration (simple Euler)
        # dV/dt = (-V + Input) / tau_mem
        dv = (-self.v + total_input) / self.tau_mem

        active_mask = (self.refractory_count <= 0).to(self.v.dtype)
        self.v = self.v + (dv * self.dt * active_mask)

        # D) Spike generation
        new_spikes = (self.v > self.theta).to(self.v.dtype)

        # E) Reset and refractory update
        self.v = self.v * (1.0 - new_spikes) + (self.v_reset * new_spikes)

        self.refractory_count = torch.where(
            new_spikes > 0,
            torch.full_like(self.refractory_count, self.refractory_period),
            self.refractory_count - self.dt,
        )

        # F) Adaptation dynamics
        # dA/dt = -A / tau_adapt + spike
        da = (-self.a / self.tau_adapt) + new_spikes
        self.a = self.a + (da * self.dt)

        # G) Threshold dynamics (simple decay to baseline + spike-dependent boost)
        self.theta = cfg.THETA_BASE + (self.theta - cfg.THETA_BASE) * cfg.THETA_DECAY
        self.theta = self.theta + (new_spikes * 0.5)  # spike → temporarily harder to fire

        self.spikes = new_spikes
        return self.spikes