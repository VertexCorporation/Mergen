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
import base64
import hashlib
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any


MX_MAGIC = b"MERGEN_MX_v1.0\n"
LEGACY_DREAM_MAGIC = b"MRGN"
LEGACY_DREAM_KEY = b"mergen_key_2026"


def _derive_mx_key(user_id: str = "default") -> bytes:
    """Match LimbicExecutiveLayer's deterministic .mx key derivation."""
    h = hashlib.sha256(f"mergen-mx-{user_id}-vertex".encode()).digest()
    return base64.b64encode(h)


def _xor_encrypt_mx(data: bytes, user_id: str = "default") -> bytes:
    key = _derive_mx_key(user_id)
    encrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
    return base64.b64encode(encrypted)


def _xor_decrypt_mx(data: bytes, user_id: str = "default") -> bytes:
    decoded = base64.b64decode(data)
    key = _derive_mx_key(user_id)
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(decoded))


def _xor_decrypt_legacy_dream(data: bytes) -> bytes:
    xored = base64.b64decode(data)
    return bytes(
        xored[i] ^ LEGACY_DREAM_KEY[i % len(LEGACY_DREAM_KEY)]
        for i in range(len(xored))
    )

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
        'MX_WEIGHTS_PATH': './mergen_weights.mx',
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
                            except (ValueError, TypeError) as e:
                                print(f"[Dream] Skipping config value {target.id}: {e}")
            print(f"[Dream] Loaded config from {config_path}")
        except Exception as e:
            print(f"[Dream] Config parse error: {e} - using defaults")
    else:
        print(f"[Dream] No {config_path} found - using defaults")

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

    @property
    def weights(self) -> Optional[torch.Tensor]:
        if getattr(self, 'is_column', False) and self.engine is not None:
            return self.engine.L5.weights.data
        return getattr(self, '_weights', None)

    @weights.setter
    def weights(self, value: Optional[torch.Tensor]) -> None:
        if getattr(self, 'is_column', False) and self.engine is not None:
            if value is not None:
                self.engine.L5.weights.data = value
        else:
            self._weights = value

    @property
    def trace_pre(self) -> Optional[torch.Tensor]:
        if getattr(self, 'is_column', False) and self.engine is not None:
            return self.engine.trace_pre
        return getattr(self, '_trace_pre', None)

    @trace_pre.setter
    def trace_pre(self, value: Optional[torch.Tensor]) -> None:
        if getattr(self, 'is_column', False) and self.engine is not None:
            if value is not None:
                self.engine.trace_pre = value
        else:
            self._trace_pre = value

    @property
    def trace_post(self) -> Optional[torch.Tensor]:
        if getattr(self, 'is_column', False) and self.engine is not None:
            return self.engine.trace_post
        return getattr(self, '_trace_post', None)

    @trace_post.setter
    def trace_post(self, value: Optional[torch.Tensor]) -> None:
        if getattr(self, 'is_column', False) and self.engine is not None:
            if value is not None:
                self.engine.trace_post = value
        else:
            self._trace_post = value

    @property
    def firing_rate(self) -> Optional[torch.Tensor]:
        if getattr(self, 'is_column', False) and self.engine is not None:
            return self.engine.firing_rate_ema
        return getattr(self, '_firing_rate', None)

    @firing_rate.setter
    def firing_rate(self, value: Optional[torch.Tensor]) -> None:
        if getattr(self, 'is_column', False) and self.engine is not None:
            if value is not None:
                self.engine.firing_rate_ema = value
        else:
            self._firing_rate = value

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
        self.user_id = self.config.get('USER_ID', 'default')

        # Multi-layer CorticalColumn states
        self.is_column = False
        self.engine = None

        # Backing variables for legacy mode
        self._weights: Optional[torch.Tensor] = None
        self._trace_pre: Optional[torch.Tensor] = None
        self._trace_post: Optional[torch.Tensor] = None
        self._firing_rate: Optional[torch.Tensor] = None

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
        mx_path = Path(self.config['MX_WEIGHTS_PATH'])
        logs_dir = Path(self.config['LOGS_DIR'])
        
        self.raw_state_dict = {}

        # ── Load .mx weights ──
        if mx_path.exists():
            try:
                with open(mx_path, 'rb') as f:
                    content = f.read()
                
                if content.startswith(MX_MAGIC):
                    encrypted = content[len(MX_MAGIC):]
                    json_bytes = _xor_decrypt_mx(encrypted, self.user_id)
                    state = json.loads(json_bytes.decode('utf-8'))
                    self.raw_state_dict = state

                    eng = state.get('engine', {})
                    if 'L4_weights' in eng and eng['L4_weights'] is not None:
                        self.is_column = True
                        from learning.cortical_column import CorticalColumn
                        n_post_col = len(eng['weights']) if 'weights' in eng else self.config['N_POST']
                        self.engine = CorticalColumn(
                            n_pre=768,
                            n_post=n_post_col,
                            n_hidden=1024,
                            device=self.device
                        )
                        # Set column weights
                        self.engine.L4.weights.data = torch.tensor(eng['L4_weights'], device=self.device).float()
                        if 'L23_weights' in eng and eng['L23_weights'] is not None:
                            self.engine.L23.weights.data = torch.tensor(eng['L23_weights'], device=self.device).float()
                        if 'weights' in eng and eng['weights'] is not None:
                            self.engine.L5.weights.data = torch.tensor(eng['weights'], device=self.device).float()
                        if 'L6_weights' in eng and eng['L6_weights'] is not None:
                            self.engine.L6.weights.data = torch.tensor(eng['L6_weights'], device=self.device).float()

                        # Load trace states directly
                        if 'trace_pre' in eng and eng['trace_pre'] is not None:
                            self.engine.trace_pre = torch.tensor(eng['trace_pre'], device=self.device).float()
                        if 'trace_post' in eng and eng['trace_post'] is not None:
                            self.engine.trace_post = torch.tensor(eng['trace_post'], device=self.device).float()
                        if 'firing_rate_ema' in eng and eng['firing_rate_ema'] is not None:
                            self.engine.firing_rate_ema = torch.tensor(eng['firing_rate_ema'], device=self.device).float()

                        if self.verbose:
                            print(f"[Dream] Loaded CorticalColumn .mx weights from {mx_path}")
                            print(f"          L4:  {tuple(self.engine.L4.weights.shape)}")
                            print(f"          L23: {tuple(self.engine.L23.weights.shape)}")
                            print(f"          L5:  {tuple(self.engine.L5.weights.shape)}")
                            print(f"          L6:  {tuple(self.engine.L6.weights.shape)}")
                    else:
                        self.is_column = False
                        self.engine = None
                        if 'weights' in eng:
                            self.weights = torch.tensor(eng['weights'], device=self.device).float()
                        if 'trace_pre' in eng:
                            self.trace_pre = torch.tensor(eng['trace_pre'], device=self.device).float()
                        if 'trace_post' in eng:
                            self.trace_post = torch.tensor(eng['trace_post'], device=self.device).float()
                        if 'firing_rate_ema' in eng:
                            self.firing_rate = torch.tensor(eng['firing_rate_ema'], device=self.device).float()

                        # Shape guard: CorticalColumn geçişinde trace boyutları
                        if self.weights is not None:
                            n_pre_w = self.weights.shape[0]
                            n_post_w = self.weights.shape[1]
                            if self.trace_pre is None or self.trace_pre.shape[0] != n_pre_w:
                                self.trace_pre = torch.zeros(n_pre_w, device=self.device)
                            if self.trace_post is None or self.trace_post.shape[0] != n_post_w:
                                self.trace_post = torch.zeros(n_post_w, device=self.device)
                            if self.firing_rate is None or self.firing_rate.shape[0] != n_post_w:
                                self.firing_rate = torch.zeros(n_post_w, device=self.device)

                        if self.verbose:
                            print(f"[Dream] Loaded Legacy Hebbian .mx weights from {mx_path}")
                            if self.weights is not None:
                                print(f"          Shape: {tuple(self.weights.shape)}")

                elif content.startswith(LEGACY_DREAM_MAGIC):
                    encrypted = content[len(LEGACY_DREAM_MAGIC):]
                    json_bytes = _xor_decrypt_legacy_dream(encrypted)
                    state = json.loads(json_bytes.decode('utf-8'))
                    self.raw_state_dict = state
                    
                    eng = state.get('engine', {})
                    if 'weights' in eng:
                        self.weights = torch.tensor(eng['weights'], device=self.device).float()
                    if 'trace_pre' in eng:
                        self.trace_pre = torch.tensor(eng['trace_pre'], device=self.device).float()
                    if 'trace_post' in eng:
                        self.trace_post = torch.tensor(eng['trace_post'], device=self.device).float()
                    if 'firing_rate_ema' in eng:
                        self.firing_rate = torch.tensor(eng['firing_rate_ema'], device=self.device).float()

                    # Shape guard (aynı CorticalColumn uyum fix'i)
                    if self.weights is not None:
                        n_pre_w = self.weights.shape[0]
                        n_post_w = self.weights.shape[1]
                        if self.trace_pre is None or self.trace_pre.shape[0] != n_pre_w:
                            self.trace_pre = torch.zeros(n_pre_w, device=self.device)
                        if self.trace_post is None or self.trace_post.shape[0] != n_post_w:
                            self.trace_post = torch.zeros(n_post_w, device=self.device)
                        if self.firing_rate is None or self.firing_rate.shape[0] != n_post_w:
                            self.firing_rate = torch.zeros(n_post_w, device=self.device)

                    if self.verbose:
                        print(f"[Dream] Loaded legacy Dream .mx weights from {mx_path}")
                        if self.weights is not None:
                            print(f"          Shape: {tuple(self.weights.shape)}")

                else:
                    # Fallback to unencrypted
                    try:
                        state = torch.load(mx_path, map_location=self.device)
                    except Exception as e:
                        if self.verbose:
                            print(f"[Dream] torch.load fallback failed: {e}")
                        state = json.loads(mx_path.read_text())
                        
                    if isinstance(state, dict):
                        self.raw_state_dict = state
                        
                        # Check root level
                        if 'weights' in state:
                            w = state['weights']
                            if isinstance(w, list): w = torch.tensor(w)
                            self.weights = w.to(self.device).float()
                        # Check engine level
                        elif 'engine' in state and 'weights' in state['engine']:
                            w = state['engine']['weights']
                            if isinstance(w, list): w = torch.tensor(w)
                            self.weights = w.to(self.device).float()
                            
                        for key_name, attr_name in [('trace_pre', 'trace_pre'), ('trace_post', 'trace_post'), ('firing_rate_ema', 'firing_rate')]:
                            val = None
                            if key_name in state:
                                val = state[key_name]
                            elif 'engine' in state and key_name in state['engine']:
                                val = state['engine'][key_name]
                            if val is not None:
                                if isinstance(val, list): val = torch.tensor(val)
                                setattr(self, attr_name, val.to(self.device).float())
                                
                    if self.verbose:
                        print(f"[Dream] Loaded unencrypted weights from {mx_path}")
            except Exception as e:
                print(f"[Dream] Could not load .mx: {e}")

        # ── Initialize if no .mx was loaded ──
        if self.weights is None:
            # Check vocab size
            vocab_size = 689
            vocab_path = Path('./mergen_vocab.json')
            if vocab_path.exists():
                try:
                    vocab_data = json.loads(vocab_path.read_text(encoding='utf-8'))
                    if 'all_words' in vocab_data:
                        vocab_size = len(vocab_data['all_words'])
                    elif 'words' in vocab_data:
                        vocab_size = len(vocab_data['words'])
                except Exception as e:
                    if self.verbose:
                        print(f"[Dream] Vocab size detection failed: {e}")
            # BUG-04 FIX: n_pre artık hardcoded 768 değil.
            # Öncelik: 1) config['N_PRE'], 2) vocab_size (simetrik başlangıç).
            # Bu sayede vocab büyüdüğünde Dream .mx'i bozulmuyor.
            n_pre = self.config.get('N_PRE', None)
            if n_pre is None:
                n_pre = vocab_size
                if self.verbose:
                    print(
                        f"[Dream] N_PRE config'de bulunamadı → vocab_size={vocab_size} "
                        f"kullanılıyor (simetrik başlangıç). "
                        f"config.py'ye N_PRE = {vocab_size} eklemek önerilir."
                    )
            n_post = vocab_size
            if self.verbose:
                print(f"[Dream] Initializing fresh weights ({n_pre}x{n_post} dream-scale block)")
            self.weights = torch.rand(
                n_pre, n_post, device=self.device
            ) * 0.05
            self.trace_pre = torch.zeros(n_pre, device=self.device)
            self.trace_post = torch.zeros(n_post, device=self.device)
            self.firing_rate = torch.zeros(n_post, device=self.device)

        # ── Load episodic patterns from logs/*.npz ──
        if logs_dir.exists() and logs_dir.is_dir():
            npz_files = sorted(logs_dir.glob("*.npz"))
            if self.verbose:
                print(f"[Dream] Found {len(npz_files)} log files.")

            for npz_file in npz_files[-200:]:  # Last 200 episodes
                try:
                    with np.load(npz_file, allow_pickle=True) as data:
                        if 'spikes' in data:
                            pattern = np.array(data['spikes'])
                            self.episodic_patterns.append(pattern)
                            # Confidence: if stored, else default 0.5
                            conf = float(data.get('confidence', 0.5))
                            self.episode_confidences.append(conf)
                        elif 'pattern' in data:
                            self.episodic_patterns.append(np.array(data['pattern']))
                            self.episode_confidences.append(0.5)
                except Exception as e:
                    if self.verbose:
                        print(f"[Dream] Skipping episodic memory item: {e}")
                    continue

            if self.verbose:
                print(f"[Dream] Loaded {len(self.episodic_patterns)} "
                      f"episodic memories.")
        else:
            if self.verbose:
                print(f"[Dream] No logs dir at {logs_dir} - "
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
    def nrem_phase(self, cycles: int, stop_event: Optional[Any] = None):
        """
        NREM Sleep: Replay low-confidence episodic memories at
        accelerated speed. Prioritize memories that weren't well
        consolidated during waking.
        """
        if self.verbose:
            print(f"\n[Dream-NREM] Starting NREM phase ({cycles} cycles)")

        # CorticalColumn için girdi katmanı boyutu (768), legacy için flat weights satır sayısı (768)
        n_in_neurons = self.engine.L4.weights.shape[0] if self.is_column else self.weights.shape[0]
        has_episodes = len(self.episodic_patterns) > 0

        # Sort episodes by confidence (lowest first — they need replay most)
        if has_episodes:
            order = np.argsort(self.episode_confidences)
        
        for cycle in range(cycles):
            # Check interrupt
            if stop_event is not None and stop_event.is_set():
                raise KeyboardInterrupt("Dream interrupted by active user interaction.")

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
                    pattern.flatten()[:n_in_neurons], device=self.device
                ).float()
                if p_tensor.shape[0] < n_in_neurons:
                    pad = torch.zeros(n_in_neurons - p_tensor.shape[0],
                                      device=self.device)
                    p_tensor = torch.cat([p_tensor, pad])

                # Threshold to binary spikes
                pre_spikes = (p_tensor > 0.3).float()
                confidence = self.episode_confidences[idx]
            else:
                # No memories — use sparse random patterns
                pre_spikes = (torch.rand(n_in_neurons, device=self.device)
                              < 0.08).float()
                confidence = 0.5

            if self.is_column:
                # CorticalColumn: L4 -> L23 -> L5 forward pass
                post_spikes = self.engine.forward(pre_spikes, spiking=True)
                
                # Çok katmanlı STDP güncellemesi ve dopamine modülasyonu
                self.engine.update_traces(pre_spikes, post_spikes)
                self.engine.apply_dopamine(reward=reward_mod)
                
                # Firing rate EMA güncellemesi
                self.engine.firing_rate_ema.mul_(0.99).add_(0.01 * post_spikes)
                self.dream_stats['total_ltp_events'] += int((self.engine.L5.trace_pre > 0.1).sum().item())
            else:
                # Forward pass: generate post-spikes
                membrane = torch.matmul(pre_spikes, self.weights)
                # Threshold with adaptive value
                threshold = membrane.mean() + membrane.std()
                post_spikes = (membrane > threshold).float()

                # STDP update with reward modulation based on replay priority
                # Low-confidence memories get HIGHER learning rate
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
    def rem_phase(self, cycles: int, stop_event: Optional[Any] = None):
        """
        REM Sleep: Spontaneous high-frequency firing creates novel
        associations between unrelated memories. This is where Mergen
        gets creative — combining patterns it has never seen together.
        """
        if self.verbose:
            print(f"\n[Dream-REM] Starting REM phase ({cycles} cycles)")

        n_in_neurons = self.engine.L4.weights.shape[0] if self.is_column else self.weights.shape[0]

        for cycle in range(cycles):
            # Check interrupt
            if stop_event is not None and stop_event.is_set():
                raise KeyboardInterrupt("Dream interrupted by active user interaction.")

            # REM fires more densely (~15-20% neurons active)
            noise_spikes = (torch.rand(n_in_neurons, device=self.device)
                            < 0.15).float()

            # Blend two random memories if available (novel association)
            if len(self.episodic_patterns) >= 2:
                idx1, idx2 = np.random.choice(
                    len(self.episodic_patterns), size=2, replace=False
                )
                p1 = torch.tensor(
                    self.episodic_patterns[idx1].flatten()[:n_in_neurons],
                    device=self.device
                ).float()
                p2 = torch.tensor(
                    self.episodic_patterns[idx2].flatten()[:n_in_neurons],
                    device=self.device
                ).float()
                # Pad if needed
                p1 = torch.nn.functional.pad(p1, (0, max(0, n_in_neurons - p1.shape[0])))[:n_in_neurons]
                p2 = torch.nn.functional.pad(p2, (0, max(0, n_in_neurons - p2.shape[0])))[:n_in_neurons]

                # Mix: 40% memory1 + 40% memory2 + 20% noise
                mixed = 0.4 * p1 + 0.4 * p2 + 0.2 * noise_spikes
                pre_spikes = (mixed > 0.3).float()
                self.dream_stats['novel_associations'] += 1
            else:
                pre_spikes = noise_spikes

            if self.is_column:
                # REM özelliğidir: uyarılabilirliği artırmak için girdiyi 1.3 ile ölçeklendir
                post_spikes = self.engine.forward(pre_spikes * 1.3, spiking=True)
                
                # REM evresi: STDP güncellemelerini ve dopamine (bias=1.2) uyguluyoruz
                self.engine.update_traces(pre_spikes, post_spikes)
                self.engine.apply_dopamine(reward=1.2)
                
                # Firing rate EMA güncellemesi
                self.engine.firing_rate_ema.mul_(0.99).add_(0.01 * post_spikes)
            else:
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

    @torch.no_grad()
    def consolidate_text_memory(self, text: str, wernicke: Any) -> None:
        """
        Targeted NREM consolidation: metin bilgisini algılayıp, Hebbian
        ağırlıklarında STDP ile konsolide eder (Metin -> Ağırlık).
        """
        if self.verbose:
            print(f"[Dream-NREM] Consolidating text memory: '{text}'")

        # 1. Convert text to spike train (time_steps, n_neurons)
        spike_train = wernicke.perceive(text)
        n_in_neurons = self.engine.L4.weights.shape[0] if self.is_column else self.weights.shape[0]

        for t in range(spike_train.shape[0]):
            p_tensor = spike_train[t]
            pre_spikes = (p_tensor > 0.3).float().to(self.device)

            if self.is_column:
                # CorticalColumn: L4 -> L23 -> L5 forward pass
                post_spikes = self.engine.forward(pre_spikes, spiking=True)
                
                # Çok katmanlı STDP güncellemesi ve dopamine modülasyonu
                self.engine.update_traces(pre_spikes, post_spikes)
                self.engine.apply_dopamine(reward=1.5)
            else:
                # Forward pass: generate post-spikes
                membrane = torch.matmul(pre_spikes, self.weights)
                threshold = membrane.mean() + membrane.std()
                post_spikes = (membrane > threshold).float()

                # Run STDP step with higher reward to emphasize consolidation
                self._stdp_step(pre_spikes, post_spikes, reward_modulation=1.5)

    def get_active_dream_concepts(self, vocabulary: List[str], top_n: int = 5) -> List[str]:
        """
        REM evresi sonrasında rüyada en aktif/co-active olan kavramları döndürür.
        Öncelikle `self.firing_rate` (EMA) üzerindeki en yüksek değerli nöronları seçer,
        eğer hepsi sıfır ise ağırlık matrisindeki (self.weights) en yüksek değerli sütunları kullanır.
        """
        scores = None
        if self.firing_rate is not None and self.firing_rate.sum().item() > 0:
            scores = self.firing_rate
        else:
            # Fallback: weights'in her bir post-synaptic kelime için maksimum ağırlık değeri
            scores = self.weights.max(dim=0)[0]

        top_indices = torch.argsort(scores, descending=True)[:top_n].tolist()

        active_concepts = []
        for idx in top_indices:
            if idx < len(vocabulary):
                active_concepts.append(vocabulary[idx])
            else:
                active_concepts.append(f"concept_{idx}")
        return active_concepts

    # ════════════════════════════════════════════════════════════════
    #  HOMEOSTATIC REBALANCING
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def rebalance(self):
        """
        After dream phases, normalize firing rates and prune dead synapses.
        """
        if self.verbose:
            print(f"\n[Dream-Homeo] Homeostatic rebalancing")

        if self.is_column:
            pruned = 0
            for layer in [self.engine.L4, self.engine.L23, self.engine.L5, self.engine.L6]:
                # Ağırlıkları sınırla
                layer.weights.data.clamp_(self.config['W_MIN'], self.config['W_MAX'])
                
                # Synaptic scaling: sadece plastik katmanlar için (scaling_speed > 0)
                if getattr(layer, 'scaling_speed', 0.0) > 0.0:
                    layer._homeostatic_normalization()
                    
                # Zayıf sinapsları buda (dead synapses)
                dead_mask = layer.weights.data.abs() < 1e-4
                pruned += dead_mask.sum().item()
                layer.weights.data[dead_mask] = 0.0
                layer.weights.data.clamp_(self.config['W_MIN'], self.config['W_MAX'])
                
            # Firing rate target: Çıkış katmanı (L5) için aktivasyon düzeltmesi
            target_rate = self.config['TARGET_FIRING_RATE']
            rate_error = self.firing_rate - target_rate
            correction = 1.0 - 0.1 * torch.sign(rate_error) * (
                rate_error.abs() > 0.05
            ).float()
            self.engine.L5.weights.data.mul_(correction.unsqueeze(0))
            self.engine.L5.weights.data.clamp_(self.config['W_MIN'], self.config['W_MAX'])
            
            self.dream_stats['pruned_synapses'] = pruned
            
            if self.verbose:
                print(f"  Pruned {pruned:,} dead synapses across layers")
                print(f"  Final L5 W̄ = {self.weights.mean().item():.4f}")
                print(f"  Final firing rate = {self.firing_rate.mean().item():.4f}")
        else:
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
        mx_path = Path(self.config['MX_WEIGHTS_PATH'])
        # ARCH-02 FIX: .mx.lock advisory lock dosyası — Limbic ve Dream'in
        # aynı anda aynı .mx dosyasına yazmasını önler.
        lock_path = mx_path.with_suffix('.mx.lock')
        _lock_acquired = False
        _lock_deadline = time.time() + 30.0  # 30 saniye timeout
        while time.time() < _lock_deadline:
            try:
                # Exclusive create: başka process lock tutuyorsa FileExistsError atar
                lock_fd = open(lock_path, 'x')
                lock_fd.write(f"dream:{os.getpid()}:{time.time()}")
                lock_fd.close()
                _lock_acquired = True
                break
            except FileExistsError:
                time.sleep(0.1)  # Bekle ve tekrar dene

        if not _lock_acquired:
            print(f"[Dream] ARCH-02 WARNING: .mx lock alınamadı (30s timeout). Yazma atlanıyor!")
            return

        try:
            # Save back to .mx (encrypted json format matching limbic_executive_layer.py)
            try:
                # We must preserve the existing raw_state_dict keys (limbic, broca, version, etc.)
                # if they exist, and only update the 'engine' part.
                state = getattr(self, 'raw_state_dict', {})
                if not isinstance(state, dict):
                    state = {}
                    
                state['version'] = state.get('version', '2.0') if self.is_column else state.get('version', '1.0')
                state['timestamp'] = datetime.now().isoformat()
                
                # Setup engine dict
                if 'engine' not in state:
                    state['engine'] = {}
                    
                state['engine']['weights'] = self.weights.cpu().tolist()
                if self.trace_pre is not None:
                    state['engine']['trace_pre'] = self.trace_pre.cpu().tolist()
                if self.trace_post is not None:
                    state['engine']['trace_post'] = self.trace_post.cpu().tolist()
                if self.firing_rate is not None:
                    state['engine']['firing_rate_ema'] = self.firing_rate.cpu().tolist()

                if self.is_column:
                    state['engine']['L4_weights'] = self.engine.L4.weights.data.cpu().tolist()
                    state['engine']['L23_weights'] = self.engine.L23.weights.data.cpu().tolist()
                    state['engine']['L6_weights'] = self.engine.L6.weights.data.cpu().tolist()
                    
                # Filter out any tensor or non-serializable objects from root level to avoid serialization errors
                for k in list(state.keys()):
                    if isinstance(state[k], torch.Tensor):
                        del state[k]
                    
                # Serialize → encrypt → write
                json_bytes = json.dumps(state).encode('utf-8')
                encrypted = _xor_encrypt_mx(json_bytes, self.user_id)
                
                # Atomic write
                tmp_path = mx_path.with_suffix('.mx.tmp')
                with open(tmp_path, 'wb') as f:
                    f.write(MX_MAGIC)
                    f.write(encrypted)
                tmp_path.replace(mx_path)
                
                if self.verbose:
                    print(f"\n[Dream] Saved consolidated weights to {mx_path}")
            except Exception as e:
                print(f"[Dream] Save error: {e}")
        finally:
            # Her durumda lock'u serbest bırak (finally garantisi)
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass

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
                print(f"[Dream] Dream log saved to {dream_log_path}")
        except Exception as e:
            print(f"[Dream] Log save error: {e}")

    # ════════════════════════════════════════════════════════════════
    #  FULL SLEEP CYCLE
    # ════════════════════════════════════════════════════════════════

    def sleep(self, cycles: Optional[int] = None, stop_event: Optional[Any] = None):
        """
        Run a complete sleep cycle: load → NREM → REM → rebalance → save.
        """
        if cycles is None:
            cycles = self.config['DREAM_CYCLES']

        self.dream_stats['start_time'] = time.time()

        if self.verbose:
            print("\n" + "=" * 65)
            print(f"  MERGEN V3 - DREAM SESSION")
            print(f"  Target cycles: {cycles:,} | Device: {self.device}")
            print("=" * 65)

        # Load state
        if self.weights is None:
            self.load_memory()

        # Allocate cycles between NREM and REM
        nrem_cycles = int(cycles * self.config['NREM_RATIO'])
        rem_cycles = cycles - nrem_cycles

        try:
            # Phase 1: NREM (consolidation)
            self.nrem_phase(nrem_cycles, stop_event=stop_event)

            # Phase 2: REM (creativity)
            self.rem_phase(rem_cycles, stop_event=stop_event)

            # Phase 3: Homeostasis
            self.rebalance()

            # Phase 4: Persist
            self.save()
        except KeyboardInterrupt:
            if self.verbose:
                print("\n[Dream] Sleep process was interrupted by user activity. Synaptic weights were NOT saved to disk.")
            raise

        # Final report
        duration = time.time() - self.dream_stats['start_time']
        self.dream_stats['duration'] = duration

        if self.verbose:
            print("\n" + "=" * 65)
            print("  DREAM COMPLETE")
            print("=" * 65)
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
            print("=" * 65 + "\n")

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
                print("[Dream] matplotlib not installed - skipping plot.")
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
            print(f"[Dream] Trajectory plot saved to {plot_path}")


# ════════════════════════════════════════════════════════════════════
#  STANDALONE ENTRY POINT
# ════════════════════════════════════════════════════════════════════

    def extract_semantic_truth(self, episodic_text: str) -> str:
        import re
        text = episodic_text.lower()
        context_words = [
            r'kullanıcı\s+bana\s+', r'kullanıcı\s+', r'bana\s+', r'benim\s+',
            r'senin\s+', r'onun\s+', r'saat\s+\d{1,2}:\d{2}\'?t?e?\s+',
            r'bugün\s+', r'dün\s+', r'yarın\s+', r'şimdi\s+',
            r'dedi\s*', r'söyledi\s*', r'belirtti\s*', r'ifade\s+etti\s*',
            r'olduğunu\s+', r'olduğunu\s+söyledi\s*'
        ]
        for cw in context_words:
            text = re.sub(cw, '', text).strip()
        if len(text.split()) < 2:
            return episodic_text
        return text

    def consolidate_episodes(self, brain) -> None:
        if not hasattr(brain, 'episodic') or not hasattr(brain, 'semantic'):
            return
        if self.verbose:
            print("[Dream] Episodik bellek Semantik belleğe konsolide ediliyor...")
        consolidated_count = 0
        for event in brain.episodic.events:
            if event.get('weight', 0.0) >= 0.8 or event.get('access_count', 0) >= 1:
                raw_text = event.get('text', '')
                semantic_fact = self.extract_semantic_truth(raw_text)
                if self.verbose:
                    print(f"[Dream-Consolidation] Episodik: '{raw_text}' -> Semantik: '{semantic_fact}'")
                brain.semantic.add_fact(
                    text=semantic_fact, 
                    concept_ids=event.get('concept_ids', []),
                    weight=event.get('weight', 1.0) * 1.5
                )
                consolidated_count += 1
        if self.verbose:
            print(f"[Dream] {consolidated_count} anı semantik gerçeğe dönüştürüldü.")
        brain.episodic.clear()

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
