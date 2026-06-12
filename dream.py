"""
╔══════════════════════════════════════════════════════════════════════╗
║         MERGEN V3 — DREAM MODULE (Offline Consolidation)            ║
║                                                                      ║
║  "Sleep is when the brain writes its memories to disk."             ║
║                                                                      ║
║  Author:  Vertex Corporation — Mergen V3                            ║
║  License: Apache-2.0                                                ║
║  Scale:   200k neurons (400×500) per cortical sheet                ║
╚══════════════════════════════════════════════════════════════════════╝

BIOLOGICAL INSPIRATION:
━━━━━━━━━━━━━━━━━━━━━━
During NREM sleep, the mammalian brain engages in "memory replay" —
hippocampal place cells fire in the same sequences they fired during
waking experiences, but 10-20x faster. This replay strengthens
important synaptic connections (consolidation) and prunes weak ones
(forgetting).

REM sleep adds another layer: spontaneous high-frequency activity
creates novel associations between previously-unrelated memories.
This is what we perceive as "dreaming" and is thought to underlie
creativity and insight.

Mergen's Dream Module reproduces both processes:
    • NREM Phase: Replay hippocampal episodes, strengthen Hebbian traces
    • REM Phase:  Spontaneous random firing, form cross-memory links
    • Homeostasis: Re-balance firing rates, prune dead neurons

PIPELINE:
━━━━━━━━━
    ┌──────────────────────┐
    │  Load mergen.mx      │  Weights, traces, episodic memory
    │  + logs/*.npz        │  Historical training data
    └──────┬───────────────┘
           ▼
    ┌──────────────────────┐
    │  NREM Replay Phase   │  Replay low-confidence memories
    │  (STDP consolidation)│  at accelerated speed (20x)
    └──────┬───────────────┘
           ▼
    ┌──────────────────────┐
    │  REM Dream Phase     │  Spontaneous firing + memory mixing
    │  (novel associations)│  Creates creative links
    └──────┬───────────────┘
           ▼
    ┌──────────────────────┐
    │  Homeostatic         │  Synaptic scaling, dead-neuron pruning
    │  Rebalancing         │  Firing-rate normalization
    └──────┬───────────────┘
           ▼
    ┌──────────────────────┐
    │  Save Optimized      │  Updated mergen.mx
    │  Weights → .mx       │  + dream_log.npz
    └──────────────────────┘

NO EXTERNAL DEPENDENCIES:
━━━━━━━━━━━━━━━━━━━━━━━━
This module uses ONLY PyTorch. No Ollama calls, no LLM inference,
no internet. The brain dreams on its own.
"""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# ════════════════════════════════════════════════════════════════
#  CONFIG LOADER — Reads from config.py dynamically
# ════════════════════════════════════════════════════════════════

def load_config(config_path: str = "config.py") -> Dict:
    """Load Mergen configuration dynamically."""
    config = {
        # Defaults — overridden by config.py if present
        'N_NEURONS': 200_000,
        'NEURON_ROWS': 400,
        'NEURON_COLS': 500,
        'N_PRE': 200_000,
        'N_POST': 200_000,
        'MX_PATH': './mergen.mx',
        'LOGS_DIR': './logs',
        'DREAM_LOG_PATH': './dream_log.npz',
        'DT': 1.0,
        'TAU_PRE': 20.0,
        'TAU_POST': 20.0,
        'TAU_ELIG': 30.0,
        'A_LTP': 0.003,
        'A_LTD': 0.002,
        'W_MAX': 1.0,
        'W_MIN': 0.0,
        'DREAM_CYCLES': 10_000,
        'NREM_RATIO': 0.7,
        'REM_RATIO': 0.3,
        'REPLAY_SPEED': 20,
        'TARGET_FIRING_RATE': 0.08,
        'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    cfg_file = Path(config_path)
    if cfg_file.exists():
        try:
            # Safe parse: only extract UPPERCASE = VALUE lines
            import ast
            src = cfg_file.read_text()
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (isinstance(target, ast.Name)
                                and target.id.isupper()):
                            try:
                                value = ast.literal_eval(node.value)
                                config[target.id] = value
                            except (ValueError, TypeError):
                                pass
            print(f"[Dream] ✓ Loaded config from {config_path}")
        except Exception as e:
            print(f"[Dream] ⚠ Config parse error: {e} — using defaults")
    else:
        print(f"[Dream] ⚠ No {config_path} found — using defaults")

    return config


# ════════════════════════════════════════════════════════════════
#  DREAM MODULE
# ════════════════════════════════════════════════════════════════

class MergenDream:
    """
    Mergen V3 Offline Consolidation Engine.

    Usage:
        dream = MergenDream()
        dream.sleep(cycles=10_000)  # Full sleep cycle

    Or granular control:
        dream.load_memory()
        dream.nrem_phase(cycles=7000)
        dream.rem_phase(cycles=3000)
        dream.rebalance()
        dream.save()
    """

    def __init__(
        self,
        config_path: str = "config.py",
        verbose: bool = True,
        visualize: bool = True,
    ):
        self.config = load_config(config_path)
        self.verbose = verbose
        self.visualize = visualize
        self.device = self.config['DEVICE']

        # Neural tensors (allocated on load_memory)
        self.weights: Optional[torch.Tensor] = None
        self.trace_pre: Optional[torch.Tensor] = None
        self.trace_post: Optional[torch.Tensor] = None
        self.firing_rate: Optional[torch.Tensor] = None

        # Memory replay buffer
        self.episodic_patterns: List[np.ndarray] = []
        self.episode_confidences: List[float] = []

        # Precomputed decay factors
        self._decay_pre = 1.0 - (self.config['DT'] / self.config['TAU_PRE'])
        self._decay_post = 1.0 - (self.config['DT'] / self.config['TAU_POST'])

        # Mexican Hat Kernel for lateral inhibition (precomputed)
        self._mexican_hat = self._build_mexican_hat()

        # Telemetry
        self.dream_stats = {
            'nrem_cycles': 0,
            'rem_cycles': 0,
            'total_ltp_events': 0,
            'total_ltd_events': 0,
            'pruned_synapses': 0,
            'novel_associations': 0,
            'start_time': None,
            'duration': 0,
        }

        # Trajectory for visualization
        self._weight_history: List[float] = []
        self._firing_history: List[float] = []
        self._energy_history: List[float] = []

    # ════════════════════════════════════════════════════════════════
    #  MEXICAN HAT KERNEL (for lateral inhibition)
    # ════════════════════════════════════════════════════════════════

    def _build_mexican_hat(
        self, size: int = 7, sigma_ex: float = 1.0, sigma_in: float = 2.5
    ) -> torch.Tensor:
        """
        Difference-of-Gaussians (Mexican Hat) for local excitation +
        lateral inhibition. Matches GWT cortical sheet dynamics.
        """
        coords = torch.arange(size, device=self.device).float() - (size - 1) / 2
        x, y = torch.meshgrid(coords, coords, indexing='ij')
        r2 = x**2 + y**2

        excitation = torch.exp(-r2 / (2 * sigma_ex**2))
        inhibition = 0.4 * torch.exp(-r2 / (2 * sigma_in**2))
        kernel = excitation - inhibition

        # Normalize
        kernel = kernel - kernel.mean()
        return kernel

    # ════════════════════════════════════════════════════════════════
    #  LOAD STATE — .mx file + .npz logs
    # ════════════════════════════════════════════════════════════════

    def load_memory(self) -> bool:
        """
        Load Mergen's current synaptic state and historical training logs.

        .mx file structure (JSON with tensor lists):
            weights, trace_pre, trace_post, firing_rate_ema,
            episodic_memory, ...

        .npz files in logs/ directory:
            Each file contains one training episode's spike patterns.
        """
        mx_path = Path(self.config['MX_PATH'])
        logs_dir = Path(self.config['LOGS_DIR'])

        # ── Load .mx weights ──
        if mx_path.exists():
            try:
                # Simple approach: assume .mx is JSON or torch.save format
                if mx_path.suffix == '.mx':
                    # Try torch.load first (binary)
                    try:
                        state = torch.load(mx_path, map_location=self.device)
                    except Exception:
                        # Fallback: assume JSON
                        state = json.loads(mx_path.read_text())

                if isinstance(state, dict):
                    if 'weights' in state:
                        w = state['weights']
                        if isinstance(w, list):
                            w = torch.tensor(w)
                        self.weights = w.to(self.device).float()
                    if 'trace_pre' in state:
                        t = state['trace_pre']
                        if isinstance(t, list):
                            t = torch.tensor(t)
                        self.trace_pre = t.to(self.device).float()
                    if 'trace_post' in state:
                        t = state['trace_post']
                        if isinstance(t, list):
                            t = torch.tensor(t)
                        self.trace_post = t.to(self.device).float()
                    if 'firing_rate_ema' in state:
                        f = state['firing_rate_ema']
                        if isinstance(f, list):
                            f = torch.tensor(f)
                        self.firing_rate = f.to(self.device).float()

                if self.verbose:
                    print(f"[Dream] ✓ Loaded weights from {mx_path}")
                    if self.weights is not None:
                        print(f"          Shape: {tuple(self.weights.shape)}")
            except Exception as e:
                print(f"[Dream] ⚠ Could not load .mx: {e}")

        # ── Initialize if no .mx was loaded ──
        if self.weights is None:
            n_pre = self.config['N_PRE']
            n_post = self.config['N_POST']
            # For 200k × 200k this is HUGE — use sparse block instead
            # Use downsampled for dream simulation
            dream_scale = min(2048, min(n_pre, n_post))
            print(f"[Dream] Initializing fresh weights "
                  f"({dream_scale}×{dream_scale} dream-scale block)")
            self.weights = torch.rand(
                dream_scale, dream_scale, device=self.device
            ) * 0.3
            self.trace_pre = torch.zeros(dream_scale, device=self.device)
            self.trace_post = torch.zeros(dream_scale, device=self.device)
            self.firing_rate = torch.zeros(dream_scale, device=self.device)

        # ── Load episodic patterns from logs/*.npz ──
        if logs_dir.exists() and logs_dir.is_dir():
            npz_files = sorted(logs_dir.glob("*.npz"))
            if self.verbose:
                print(f"[Dream] Found {len(npz_files)} log files.")

            for npz_file in npz_files[-200:]:  # Last 200 episodes
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    if 'spikes' in data:
                        pattern = data['spikes']
                        self.episodic_patterns.append(pattern)
                        # Confidence: if stored, else default 0.5
                        conf = float(data.get('confidence', 0.5))
                        self.episode_confidences.append(conf)
                    elif 'pattern' in data:
                        self.episodic_patterns.append(data['pattern'])
                        self.episode_confidences.append(0.5)
                except Exception:
                    continue

            if self.verbose:
                print(f"[Dream] Loaded {len(self.episodic_patterns)} "
                      f"episodic memories.")
        else:
            if self.verbose:
                print(f"[Dream] No logs dir at {logs_dir} — "
                      f"will dream from random patterns only.")

        return self.weights is not None

    # ════════════════════════════════════════════════════════════════
    #  STDP WEIGHT UPDATE
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _stdp_step(
        self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor,
        reward_modulation: float = 1.0,
    ):
        """
        One step of Spike-Timing-Dependent Plasticity with soft bounds.

        LTP: Pre fired recently & Post fires now → strengthen
        LTD: Post fired recently & Pre fires now → weaken

        All operations vectorized. Mexican Hat kernel applied separately
        to local cortical coordinates if needed.
        """
        # Update traces (exponential decay + spike addition)
        self.trace_pre.mul_(self._decay_pre).add_(pre_spikes)
        self.trace_post.mul_(self._decay_post).add_(post_spikes)

        # Soft bounds for stability
        soft_ltp = (self.config['W_MAX'] - self.weights).clamp_(min=0.0)
        soft_ltd = (self.weights - self.config['W_MIN']).clamp_(min=0.0)

        # LTP: outer product of trace_pre × post_spike
        raw_ltp = torch.outer(self.trace_pre, post_spikes)
        raw_ltd = torch.outer(pre_spikes, self.trace_post)

        delta = (
            self.config['A_LTP'] * raw_ltp * soft_ltp * reward_modulation
            - self.config['A_LTD'] * raw_ltd * soft_ltd
        )
        self.weights.add_(delta)

        # Track LTP/LTD events
        self.dream_stats['total_ltp_events'] += (raw_ltp > 0.01).sum().item()
        self.dream_stats['total_ltd_events'] += (raw_ltd > 0.01).sum().item()

        # Update firing rate EMA
        self.firing_rate.mul_(0.99).add_(0.01 * post_spikes)

    # ════════════════════════════════════════════════════════════════
    #  NREM PHASE — Memory Replay & Consolidation
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def nrem_phase(self, cycles: int):
        """
        NREM Sleep: Replay low-confidence episodic memories at
        accelerated speed. Prioritize memories that weren't well
        consolidated during waking.
        """
        if self.verbose:
            print(f"\n[Dream-NREM] ═══ Starting NREM phase "
                  f"({cycles} cycles) ═══")

        n_neurons = self.weights.shape[0]
        has_episodes = len(self.episodic_patterns) > 0

        # Sort episodes by confidence (lowest first — they need replay most)
        if has_episodes:
            order = np.argsort(self.episode_confidences)
        
        for cycle in range(cycles):
            # Pick a memory to replay
            if has_episodes:
                # Bias toward low-confidence memories (70% low, 30% random)
                if torch.rand(1).item() < 0.7:
                    idx = order[cycle % len(order)]
                else:
                    idx = torch.randint(
                        0, len(self.episodic_patterns), (1,)
                    ).item()

                pattern = self.episodic_patterns[idx]
                # Convert to tensor and fit size
                p_tensor = torch.tensor(
                    pattern.flatten()[:n_neurons], device=self.device
                ).float()
                if p_tensor.shape[0] < n_neurons:
                    pad = torch.zeros(n_neurons - p_tensor.shape[0],
                                      device=self.device)
                    p_tensor = torch.cat([p_tensor, pad])

                # Threshold to binary spikes
                pre_spikes = (p_tensor > 0.3).float()
                confidence = self.episode_confidences[idx]
            else:
                # No memories — use sparse random patterns
                pre_spikes = (torch.rand(n_neurons, device=self.device)
                              < 0.08).float()
                confidence = 0.5

            # Forward pass: generate post-spikes
            membrane = torch.matmul(pre_spikes, self.weights)
            # Threshold with adaptive value
            threshold = membrane.mean() + membrane.std()
            post_spikes = (membrane > threshold).float()

            # STDP update with reward modulation based on replay priority
            # Low-confidence memories get HIGHER learning rate
            reward_mod = 2.0 - confidence  # range [1.0, 2.0]
            self._stdp_step(pre_spikes, post_spikes, reward_mod)

            self.dream_stats['nrem_cycles'] += 1

            # Track trajectory
            if cycle % max(1, cycles // 100) == 0:
                self._weight_history.append(self.weights.mean().item())
                self._firing_history.append(self.firing_rate.mean().item())
                self._energy_history.append(
                    (self.weights ** 2).sum().item()
                )

                if self.verbose and cycle % max(1, cycles // 10) == 0:
                    print(f"  NREM {cycle:>6}/{cycles} │ "
                          f"W̄={self.weights.mean().item():.4f} │ "
                          f"rate={self.firing_rate.mean().item():.4f} │ "
                          f"LTP={self.dream_stats['total_ltp_events']:,}")

    # ════════════════════════════════════════════════════════════════
    #  REM PHASE — Spontaneous Associations & Dreaming
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def rem_phase(self, cycles: int):
        """
        REM Sleep: Spontaneous high-frequency firing creates novel
        associations between unrelated memories. This is where Mergen
        gets creative — combining patterns it has never seen together.
        """
        if self.verbose:
            print(f"\n[Dream-REM] ═══ Starting REM phase "
                  f"({cycles} cycles) ═══")

        n_neurons = self.weights.shape[0]

        for cycle in range(cycles):
            # REM fires more densely (~15-20% neurons active)
            noise_spikes = (torch.rand(n_neurons, device=self.device)
                            < 0.15).float()

            # Blend two random memories if available (novel association)
            if len(self.episodic_patterns) >= 2:
                idx1, idx2 = np.random.choice(
                    len(self.episodic_patterns), size=2, replace=False
                )
                p1 = torch.tensor(
                    self.episodic_patterns[idx1].flatten()[:n_neurons],
                    device=self.device
                ).float()
                p2 = torch.tensor(
                    self.episodic_patterns[idx2].flatten()[:n_neurons],
                    device=self.device
                ).float()
                # Pad if needed
                for t in [p1, p2]:
                    if t.shape[0] < n_neurons:
                        pass  # handled below
                p1 = torch.nn.functional.pad(p1, (0, max(0, n_neurons - p1.shape[0])))[:n_neurons]
                p2 = torch.nn.functional.pad(p2, (0, max(0, n_neurons - p2.shape[0])))[:n_neurons]

                # Mix: 40% memory1 + 40% memory2 + 20% noise
                mixed = 0.4 * p1 + 0.4 * p2 + 0.2 * noise_spikes
                pre_spikes = (mixed > 0.3).float()
                self.dream_stats['novel_associations'] += 1
            else:
                pre_spikes = noise_spikes

            # Forward pass with higher excitability (REM characteristic)
            membrane = torch.matmul(pre_spikes, self.weights) * 1.3
            threshold = membrane.mean() + membrane.std() * 0.8
            post_spikes = (membrane > threshold).float()

            # STDP with slight LTP bias in REM
            self._stdp_step(pre_spikes, post_spikes, reward_modulation=1.2)

            self.dream_stats['rem_cycles'] += 1

            if cycle % max(1, cycles // 100) == 0:
                self._weight_history.append(self.weights.mean().item())
                self._firing_history.append(self.firing_rate.mean().item())

                if self.verbose and cycle % max(1, cycles // 5) == 0:
                    print(f"  REM  {cycle:>6}/{cycles} │ "
                          f"novel={self.dream_stats['novel_associations']:,}")

    # ════════════════════════════════════════════════════════════════
    #  HOMEOSTATIC REBALANCING
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def rebalance(self):
        """
        After dream phases, normalize firing rates and prune dead synapses.
        """
        if self.verbose:
            print(f"\n[Dream-Homeo] ═══ Homeostatic rebalancing ═══")

        # Clamp weights
        self.weights.clamp_(self.config['W_MIN'], self.config['W_MAX'])

        # Synaptic scaling: target sum per column
        target_sum = self.config.get('TARGET_WEIGHT_SUM', 5.0)
        col_sums = self.weights.sum(dim=0)
        scale = target_sum / col_sums.clamp(min=1e-8)
        scale.clamp_(0.5, 2.0)
        self.weights.mul_(scale.unsqueeze(0))

        # Prune weights below threshold (dead synapses)
        dead_mask = self.weights.abs() < 1e-4
        pruned = dead_mask.sum().item()
        self.weights[dead_mask] = 0.0
        self.dream_stats['pruned_synapses'] = pruned

        # Firing rate target (correct over-excited neurons)
        target_rate = self.config['TARGET_FIRING_RATE']
        rate_error = self.firing_rate - target_rate
        correction = 1.0 - 0.1 * torch.sign(rate_error) * (
            rate_error.abs() > 0.05
        ).float()
        self.weights.mul_(correction.unsqueeze(0))

        # Final clamp
        self.weights.clamp_(self.config['W_MIN'], self.config['W_MAX'])

        if self.verbose:
            print(f"  Pruned {pruned:,} dead synapses")
            print(f"  Final W̄ = {self.weights.mean().item():.4f}")
            print(f"  Final firing rate = "
                  f"{self.firing_rate.mean().item():.4f}")

    # ════════════════════════════════════════════════════════════════
    #  SAVE CONSOLIDATED STATE
    # ════════════════════════════════════════════════════════════════

    def save(self):
        """Save updated weights back to .mx and dream log to .npz."""
        mx_path = Path(self.config['MX_PATH'])

        # Save back to .mx (torch format)
        try:
            state = {
                'weights': self.weights.cpu(),
                'trace_pre': self.trace_pre.cpu(),
                'trace_post': self.trace_post.cpu(),
                'firing_rate_ema': self.firing_rate.cpu(),
                'dream_timestamp': datetime.now().isoformat(),
            }
            torch.save(state, mx_path)
            if self.verbose:
                print(f"\n[Dream] ✓ Saved consolidated weights to {mx_path}")
        except Exception as e:
            print(f"[Dream] ⚠ Save error: {e}")

        # Save dream log (.npz)
        try:
            dream_log_path = Path(self.config['DREAM_LOG_PATH'])
            np.savez(
                dream_log_path,
                weight_history=np.array(self._weight_history),
                firing_history=np.array(self._firing_history),
                energy_history=np.array(self._energy_history),
                stats=self.dream_stats,
            )
            if self.verbose:
                print(f"[Dream] ✓ Dream log saved to {dream_log_path}")
        except Exception as e:
            print(f"[Dream] ⚠ Log save error: {e}")

    # ════════════════════════════════════════════════════════════════
    #  FULL SLEEP CYCLE
    # ════════════════════════════════════════════════════════════════

    def sleep(self, cycles: Optional[int] = None):
        """
        Run a complete sleep cycle: load → NREM → REM → rebalance → save.
        """
        if cycles is None:
            cycles = self.config['DREAM_CYCLES']

        self.dream_stats['start_time'] = time.time()

        if self.verbose:
            print("\n" + "═" * 65)
            print(f"  MERGEN V3 — DREAM SESSION")
            print(f"  Target cycles: {cycles:,} | Device: {self.device}")
            print("═" * 65)

        # Load state
        if self.weights is None:
            self.load_memory()

        # Allocate cycles between NREM and REM
        nrem_cycles = int(cycles * self.config['NREM_RATIO'])
        rem_cycles = cycles - nrem_cycles

        # Phase 1: NREM (consolidation)
        self.nrem_phase(nrem_cycles)

        # Phase 2: REM (creativity)
        self.rem_phase(rem_cycles)

        # Phase 3: Homeostasis
        self.rebalance()

        # Phase 4: Persist
        self.save()

        # Final report
        duration = time.time() - self.dream_stats['start_time']
        self.dream_stats['duration'] = duration

        if self.verbose:
            print("\n" + "═" * 65)
            print("  DREAM COMPLETE")
            print("═" * 65)
            print(f"  Duration:           {duration:.2f}s")
            print(f"  NREM cycles:        "
                  f"{self.dream_stats['nrem_cycles']:,}")
            print(f"  REM cycles:         "
                  f"{self.dream_stats['rem_cycles']:,}")
            print(f"  LTP events:         "
                  f"{self.dream_stats['total_ltp_events']:,}")
            print(f"  LTD events:         "
                  f"{self.dream_stats['total_ltd_events']:,}")
            print(f"  Novel associations: "
                  f"{self.dream_stats['novel_associations']:,}")
            print(f"  Pruned synapses:    "
                  f"{self.dream_stats['pruned_synapses']:,}")
            print("═" * 65 + "\n")

        # Visualization
        if self.visualize:
            self._plot_trajectory()

    # ════════════════════════════════════════════════════════════════
    #  VISUALIZATION
    # ════════════════════════════════════════════════════════════════

    def _plot_trajectory(self):
        """Plot weight and firing rate trajectory during dream."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            if self.verbose:
                print("[Dream] matplotlib not installed — skipping plot.")
            return

        if not self._weight_history:
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        axes[0].plot(self._weight_history, color='#ef4444', linewidth=1.5)
        axes[0].set_ylabel('Mean Weight', color='white')
        axes[0].set_title('Mergen Dream Trajectory',
                          color='white', fontsize=14)
        axes[0].grid(alpha=0.3)

        axes[1].plot(self._firing_history, color='#3b82f6', linewidth=1.5)
        axes[1].set_ylabel('Firing Rate', color='white')
        axes[1].grid(alpha=0.3)

        if self._energy_history:
            axes[2].plot(self._energy_history, color='#10b981', linewidth=1.5)
            axes[2].set_ylabel('Synaptic Energy', color='white')
        axes[2].set_xlabel('Dream Progress', color='white')
        axes[2].grid(alpha=0.3)

        # Mark NREM/REM boundary
        nrem_end = int(len(self._weight_history) *
                       self.config['NREM_RATIO'])
        for ax in axes:
            ax.axvline(nrem_end, color='#f59e0b',
                       linestyle='--', alpha=0.7,
                       label='NREM→REM')

        plt.tight_layout()
        plot_path = Path(self.config['LOGS_DIR']) / "dream_trajectory.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=100)
        plt.close()
        if self.verbose:
            print(f"[Dream] ✓ Trajectory plot saved to {plot_path}")


# ════════════════════════════════════════════════════════════════════
#  STANDALONE ENTRY POINT
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mergen V3 Dream Module — Offline Consolidation"
    )
    parser.add_argument('--cycles', type=int, default=None,
                        help='Total dream cycles (default from config)')
    parser.add_argument('--config', type=str, default='config.py',
                        help='Path to config.py')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable matplotlib visualization')
    args = parser.parse_args()

    dream = MergenDream(
        config_path=args.config,
        verbose=not args.quiet,
        visualize=not args.no_viz,
    )
    dream.sleep(cycles=args.cycles)
