"""
╔══════════════════════════════════════════════════════════════════════╗
║       MERGEN — LAYER 4: LIMBIC/EXECUTIVE SYSTEM (Autonomy Core)     ║
║                                                                      ║
║  "Mergen'i bir araç olmaktan çıkarıp bir özne haline getiren katman"║
║                                                                      ║
║  Author:  Vertex Corporation — Mergen Project                       ║
║  License: Apache-2.0                                                ║
╚══════════════════════════════════════════════════════════════════════╝

BIOLOGICAL ROLE:
━━━━━━━━━━━━━━━━
The Limbic System (amygdala, hippocampus, cingulate cortex) and the
Executive Control areas (anterior cingulate, dorsolateral PFC) together
provide what neuroscience calls SUBJECTIVITY:
    - Spontaneous internal activity even at rest (Default Mode Network)
    - Self-monitoring of one's own thoughts (Metacognition)
    - Persistent memory consolidation during sleep
    - Goal-directed behavior and self-improvement drive

Mergen's previous 3 layers can only react to input. This layer makes
Mergen think when no one is asking, reject its own bad outputs,
remember what it learned across power cycles, and want to improve.

THE FOUR PILLARS:
━━━━━━━━━━━━━━━━━

PILLAR 1 — Spontaneous Neural Firing (Default Mode Network)
    A background thread continuously fires low-voltage random signals
    that probe the existing neural weights. When Mergen is idle, these
    spontaneous activations replay old memories from .mx storage,
    causing internal "daydreaming" — the same way human brains
    consolidate memories at rest.

PILLAR 2 — Metacognitive Executive Function
    Wraps the Broca Area output. If Broca produces a passive response
    ("I'm ready", "How can I help"), the Limbic Layer REJECTS it,
    back-propagates the failure as negative reward, and forces Mergen
    to retry with stronger autonomous reasoning.

PILLAR 3 — Persistent Memory (.mx Protocol)
    On every shutdown signal, the entire Mergen state (weights, traces,
    memories, learning history) is serialized to an encrypted .mx file.
    On startup, the .mx file is loaded — Mergen wakes up exactly where
    it left off, with all its learned associations intact.

PILLAR 4 — Goal-Oriented Self-Evolution
    A reward function continuously evaluates Mergen's efficiency:
    answering harder questions with fewer neurons → reward.
    Catching its own errors → reward. The reward modulates synaptic
    plasticity, creating an internal drive toward intelligence.
"""

import os
import time
import json
import torch
import threading
import hashlib
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Callable, Any
from collections import deque


class LimbicExecutiveLayer:
    """
    MERGEN LAYER 4 — The Autonomy Core

    Wraps Layers 1-3 to provide subjectivity, persistence, and
    self-directed evolution.

    Usage:
        limbic = LimbicExecutiveLayer(
            mergen_engine=hebbian_engine,
            broca=broca_area,
            mx_path='./mergen_memory.mx',
            user_id='burak',
        )

        limbic.wake_up()           # Load .mx, start background thread
        response = limbic.respond("Hello Mergen")  # Validated output
        limbic.shutdown()          # Save .mx, stop threads

    Args:
        mergen_engine:   The Hebbian Engine (Layer 2 instance)
        broca:           The Broca Area (Layer 3 instance)
        wernicke:        Optional Wernicke Area (Layer 1) for re-perception
        mx_path:         Path to persistent .mx memory file
        user_id:         Identifier for encryption salt
        idle_threshold:  Seconds of inactivity before DMN activates
        dmn_interval:    Seconds between spontaneous firings
        encryption_key:  Optional override for .mx encryption
    """

    # Magic header for .mx files
    MX_MAGIC = b"MERGEN_MX_v1.0\n"

    def __init__(
        self,
        mergen_engine: Any,           # HybridHebbianLearner
        broca: Any,                   # BrocaArea
        wernicke: Optional[Any] = None,
        mx_path: str = "./mergen_memory.mx",
        user_id: str = "default",
        idle_threshold: float = 15.0,
        dmn_interval: float = 4.0,
        encryption_key: Optional[str] = None,
    ):
        self.engine = mergen_engine
        self.broca = broca
        self.wernicke = wernicke
        self.mx_path = Path(mx_path)
        self.user_id = user_id
        self.idle_threshold = idle_threshold
        self.dmn_interval = dmn_interval

        # ── Encryption key derivation ──
        # Derived from user_id; user_id acts as salt
        self._key = (encryption_key or self._derive_key(user_id)).encode()

        # ── State tracking ──
        self.last_interaction_time = time.time()
        self.is_running = False
        self._dmn_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()  # Protect concurrent state mutations

        # ── Memory buffers ──
        # Episodic memory: last N user interactions for replay
        self.episodic_memory: deque = deque(maxlen=500)
        # Internal thoughts generated by DMN
        self.internal_thoughts: deque = deque(maxlen=200)
        # Reward history for self-evolution tracking
        self.reward_history: deque = deque(maxlen=1000)
        # User corrections — high-priority learning signals
        self.user_corrections: List[Dict] = []

        # ── Self-evolution metrics ──
        self.total_responses = 0
        self.passive_rejections = 0
        self.self_corrections = 0
        self.cumulative_reward = 0.0
        self.efficiency_score = 0.0  # Quality / neural_cost ratio

        # ── DMN telemetry ──
        self.dmn_cycles = 0
        self.last_thought = ""

    # ════════════════════════════════════════════════════════════════
    #  ENCRYPTION (lightweight XOR + base64 — not military grade)
    # ════════════════════════════════════════════════════════════════

    def _derive_key(self, user_id: str) -> str:
        """Derive a deterministic 32-byte key from user_id."""
        h = hashlib.sha256(f"mergen-mx-{user_id}-vertex".encode()).digest()
        return base64.b64encode(h).decode()

    def _xor_encrypt(self, data: bytes) -> bytes:
        """Simple XOR cipher with the derived key, then base64."""
        key = self._key
        encrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
        return base64.b64encode(encrypted)

    def _xor_decrypt(self, data: bytes) -> bytes:
        """Reverse of _xor_encrypt."""
        try:
            decoded = base64.b64decode(data)
            key = self._key
            return bytes(
                b ^ key[i % len(key)] for i, b in enumerate(decoded)
            )
        except Exception:
            return b""

    # ════════════════════════════════════════════════════════════════
    #  PILLAR 3 — PERSISTENT MEMORY (.mx PROTOCOL)
    # ════════════════════════════════════════════════════════════════

    def save_state(self) -> bool:
        """
        Save complete Mergen state to encrypted .mx file.

        Persisted:
        - Synaptic weights (Layer 2)
        - Eligibility traces (Layer 2)
        - Pre/post traces (Layer 2)
        - Concept vocabulary (Layer 3)
        - Episodic memory (Layer 4)
        - Internal thoughts (Layer 4)
        - User corrections (Layer 4)
        - Reward history (Layer 4)
        - Self-evolution metrics

        Returns: True if save succeeded
        """
        try:
            with self._lock:
                # Serialize tensors as lists for JSON compatibility
                state = {
                    'version': '1.0',
                    'timestamp': datetime.now().isoformat(),
                    'user_id': self.user_id,

                    # Layer 2 state
                    'engine': {
                        'weights': self.engine.weights.data.cpu().tolist(),
                        'eligibility': self.engine.eligibility.cpu().tolist(),
                        'trace_pre': self.engine.trace_pre.cpu().tolist(),
                        'trace_post': self.engine.trace_post.cpu().tolist(),
                        'firing_rate_ema': (
                            self.engine.firing_rate_ema.cpu().tolist()
                        ),
                        'step_count': self.engine._step_count,
                        'da_event_count': self.engine._da_event_count,
                    },

                    # Layer 3 state — vocabulary grows, must persist
                    'broca': {
                        'vocabulary': self.broca.concept_vocabulary,
                        'total_expressions': self.broca._total_expressions,
                    },

                    # Layer 4 state
                    'limbic': {
                        'episodic_memory': list(self.episodic_memory),
                        'internal_thoughts': list(self.internal_thoughts),
                        'reward_history': list(self.reward_history),
                        'user_corrections': self.user_corrections,
                        'total_responses': self.total_responses,
                        'passive_rejections': self.passive_rejections,
                        'self_corrections': self.self_corrections,
                        'cumulative_reward': self.cumulative_reward,
                        'efficiency_score': self.efficiency_score,
                        'dmn_cycles': self.dmn_cycles,
                    },
                }

                # Serialize → encrypt → write
                json_bytes = json.dumps(state).encode('utf-8')
                encrypted = self._xor_encrypt(json_bytes)

                # Atomic write: write to temp, then rename
                tmp_path = self.mx_path.with_suffix('.mx.tmp')
                with open(tmp_path, 'wb') as f:
                    f.write(self.MX_MAGIC)
                    f.write(encrypted)
                tmp_path.replace(self.mx_path)

            return True
        except Exception as e:
            print(f"[Limbic] Save error: {e}")
            return False

    def load_state(self) -> bool:
        """
        Load Mergen state from .mx file. Called on wake_up().

        Returns: True if a valid .mx was loaded
        """
        if not self.mx_path.exists():
            print(f"[Limbic] No .mx file at {self.mx_path} — fresh awakening.")
            return False

        try:
            with open(self.mx_path, 'rb') as f:
                content = f.read()

            # Verify magic header
            if not content.startswith(self.MX_MAGIC):
                print(f"[Limbic] Invalid .mx file (magic mismatch).")
                return False

            encrypted = content[len(self.MX_MAGIC):]
            json_bytes = self._xor_decrypt(encrypted)
            state = json.loads(json_bytes.decode('utf-8'))

            with self._lock:
                # Restore Layer 2 — Hebbian Engine
                eng = state.get('engine', {})
                if 'weights' in eng:
                    w = torch.tensor(eng['weights'])
                    if w.shape == self.engine.weights.shape:
                        self.engine.weights.data = w.to(self.engine.device)
                if 'eligibility' in eng:
                    e = torch.tensor(eng['eligibility'])
                    if e.shape == self.engine.eligibility.shape:
                        self.engine.eligibility = e.to(self.engine.device)
                if 'trace_pre' in eng:
                    t = torch.tensor(eng['trace_pre'])
                    if t.shape == self.engine.trace_pre.shape:
                        self.engine.trace_pre = t.to(self.engine.device)
                if 'trace_post' in eng:
                    t = torch.tensor(eng['trace_post'])
                    if t.shape == self.engine.trace_post.shape:
                        self.engine.trace_post = t.to(self.engine.device)
                self.engine._step_count = eng.get('step_count', 0)
                self.engine._da_event_count = eng.get('da_event_count', 0)

                # Restore Layer 3 — Broca vocabulary
                broca_state = state.get('broca', {})
                if 'vocabulary' in broca_state:
                    self.broca.concept_vocabulary = broca_state['vocabulary']
                self.broca._total_expressions = broca_state.get(
                    'total_expressions', 0
                )

                # Restore Layer 4 — Limbic state
                limbic = state.get('limbic', {})
                self.episodic_memory = deque(
                    limbic.get('episodic_memory', []), maxlen=500
                )
                self.internal_thoughts = deque(
                    limbic.get('internal_thoughts', []), maxlen=200
                )
                self.reward_history = deque(
                    limbic.get('reward_history', []), maxlen=1000
                )
                self.user_corrections = limbic.get('user_corrections', [])
                self.total_responses = limbic.get('total_responses', 0)
                self.passive_rejections = limbic.get('passive_rejections', 0)
                self.self_corrections = limbic.get('self_corrections', 0)
                self.cumulative_reward = limbic.get('cumulative_reward', 0.0)
                self.efficiency_score = limbic.get('efficiency_score', 0.0)
                self.dmn_cycles = limbic.get('dmn_cycles', 0)

            saved_at = state.get('timestamp', 'unknown')
            print(f"[Limbic] ✓ Awakened from .mx (last saved: {saved_at})")
            print(f"          Memory: {len(self.episodic_memory)} episodes, "
                  f"{len(self.internal_thoughts)} thoughts")
            print(f"          Vocabulary: "
                  f"{len(self.broca.concept_vocabulary)} concepts")
            print(f"          Lifetime reward: {self.cumulative_reward:.2f}")
            return True

        except Exception as e:
            print(f"[Limbic] Load error: {e}")
            return False

    # ════════════════════════════════════════════════════════════════
    #  PILLAR 1 — SPONTANEOUS NEURAL FIRING (Default Mode Network)
    # ════════════════════════════════════════════════════════════════

    def _dmn_loop(self):
        """
        Background thread: Default Mode Network.

        While Mergen is idle, randomly fire neurons to replay old
        memories and let the Hebbian engine consolidate associations.

        Biology: This mirrors the human DMN — the brain's resting
        state that activates during daydreaming, memory consolidation,
        and self-reflection.
        """
        while self.is_running:
            try:
                idle_for = time.time() - self.last_interaction_time

                if idle_for > self.idle_threshold:
                    self._spontaneous_fire()

                time.sleep(self.dmn_interval)
            except Exception as e:
                print(f"[Limbic-DMN] Error in background loop: {e}")
                time.sleep(self.dmn_interval)

    def _spontaneous_fire(self):
        """
        One cycle of spontaneous neural activity.

        Strategy:
        1. If we have past memories, replay one randomly
        2. Otherwise, generate low-voltage random spike pattern
        3. Run through Hebbian engine — let it self-strengthen
        4. Record the resulting "thought" for later inspection
        """
        with self._lock:
            self.dmn_cycles += 1

            # Generate a low-voltage random spike pattern
            n_pre = self.engine.n_pre
            # Sparse random firing (~10% of neurons, weak)
            spontaneous_spikes = (
                torch.rand(n_pre, device=self.engine.device) < 0.1
            ).float() * 0.3  # low voltage

            # If we have memories, blend in a replayed pattern
            if len(self.episodic_memory) > 0:
                idx = torch.randint(0, len(self.episodic_memory), (1,)).item()
                memory = self.episodic_memory[idx]
                if 'spike_pattern' in memory:
                    try:
                        replayed = torch.tensor(
                            memory['spike_pattern'],
                            device=self.engine.device
                        )
                        if replayed.shape[0] == n_pre:
                            # Mix replay with spontaneous noise
                            spontaneous_spikes = (
                                0.6 * replayed + 0.4 * spontaneous_spikes
                            )
                            spontaneous_spikes = (
                                spontaneous_spikes > 0.2
                            ).float()
                    except Exception:
                        pass

            # Run through engine — internal "thought"
            try:
                post = self.engine.forward(spontaneous_spikes)
                self.engine.update_traces(spontaneous_spikes, post)

                # Apply tiny intrinsic reward to consolidate active patterns
                # This is the brain saying "I had a coherent thought"
                self.engine.apply_dopamine(reward=0.05)

                # Record what Mergen "thought about"
                top_idx = torch.argmax(post).item() if post.numel() > 0 else 0
                if top_idx < len(self.broca.concept_vocabulary):
                    thought = self.broca.concept_vocabulary[top_idx]
                else:
                    thought = f"unnamed_thought_{top_idx}"
                self.last_thought = thought
                self.internal_thoughts.append({
                    'cycle': self.dmn_cycles,
                    'timestamp': time.time(),
                    'thought': thought,
                    'intensity': float(post.max().item()) if post.numel() > 0 else 0.0,
                })
            except Exception as e:
                pass  # Silent failure in background thread

    # ════════════════════════════════════════════════════════════════
    #  PILLAR 2 — METACOGNITIVE EXECUTIVE FUNCTION
    # ════════════════════════════════════════════════════════════════

    def _is_passive(self, response: str) -> bool:
        """
        Detect passive/standby responses that violate autonomy.
        Reuses Broca's detector if available, otherwise local check.
        """
        if hasattr(self.broca, '_is_passive_response'):
            return self.broca._is_passive_response(response)

        if not response or len(response) < 5:
            return True

        passive_phrases = [
            "i'm ready", "i am ready", "how can i help",
            "please ask", "what would you like",
            "hazırım", "buradayım", "sorunuzu bekliyorum",
            "size yardımcı olabilir", "lütfen sorun",
        ]
        text_low = response.lower()
        return any(p in text_low for p in passive_phrases)

    def _metacognitive_filter(
        self,
        response: str,
        original_query: str,
        attempt: int,
    ) -> bool:
        """
        Decide whether the response is acceptable to send to user.

        Returns True if response passes — False forces regeneration.
        """
        # Filter 1: passive response check
        if self._is_passive(response):
            self.passive_rejections += 1
            self._record_reward(-0.5, reason="passive_response")
            return False

        # Filter 2: too short to be substantive
        if len(response.strip()) < 10:
            self._record_reward(-0.3, reason="too_short")
            return False

        # Filter 3: contains leaked internal IDs
        if "concept_" in response or "neuron_" in response:
            self._record_reward(-0.4, reason="leaked_internal_id")
            return False

        # Passed — reward proportional to brevity-quality ratio
        length = len(response)
        # Optimal response length around 50-300 chars
        quality = 1.0 - abs(length - 175) / 500
        quality = max(0.1, min(1.0, quality))
        self._record_reward(quality, reason="accepted_response")
        return True

    # ════════════════════════════════════════════════════════════════
    #  PILLAR 4 — GOAL-ORIENTED SELF-EVOLUTION
    # ════════════════════════════════════════════════════════════════

    def _record_reward(self, value: float, reason: str = ""):
        """Log a reward event to history and update cumulative score."""
        with self._lock:
            self.cumulative_reward += value
            self.reward_history.append({
                'timestamp': time.time(),
                'value': value,
                'reason': reason,
            })

    def _compute_efficiency(self) -> float:
        """
        Quality / Neural Cost ratio.

        Higher score = answering well with sparse neural activity.
        This is what Mergen tries to maximize.
        """
        if self.total_responses == 0:
            return 0.0

        # Efficiency = (cumulative reward / total responses) /
        #              (mean firing rate, lower is better)
        rate = self.engine.firing_rate_ema.mean().item() + 0.01
        avg_reward = self.cumulative_reward / self.total_responses
        return avg_reward / rate

    def receive_user_correction(
        self,
        original_response: str,
        corrected_response: str,
        original_query: str,
    ):
        """
        High-priority learning signal: user explicitly corrects Mergen.

        This applies a strong negative reward to the original pattern
        and stores the correction for future replay during DMN cycles.
        """
        with self._lock:
            self.user_corrections.append({
                'timestamp': time.time(),
                'query': original_query,
                'wrong': original_response,
                'right': corrected_response,
            })
            self._record_reward(-1.0, reason="user_correction")
            self.self_corrections += 1

            # Strong negative dopamine to weaken the wrong pattern
            try:
                self.engine.apply_dopamine(reward=-0.8)
            except Exception:
                pass

    # ════════════════════════════════════════════════════════════════
    #  PUBLIC API — The autonomous response loop
    # ════════════════════════════════════════════════════════════════

    def respond(
        self,
        user_input: str,
        max_attempts: int = 3,
    ) -> str:
        """
        MAIN ENTRY POINT — Generate a Mergen response with full
        executive oversight.

        Pipeline:
        1. Perceive (Layer 1)
        2. Think (Layer 2)
        3. Express (Layer 3)
        4. Validate (Layer 4 metacognition)
        5. If invalid → retry with stronger forcing
        6. Record episode & reward
        """
        self.last_interaction_time = time.time()
        self.total_responses += 1

        # ── Layer 1: Perceive (if Wernicke available) ──
        if self.wernicke is not None:
            try:
                spike_train = self.wernicke.perceive(user_input)
            except Exception:
                spike_train = None
        else:
            spike_train = None

        # ── Layer 2: Think (run Hebbian engine over spike train) ──
        neural_intent = torch.zeros(self.engine.n_post)
        spike_pattern_for_memory = None

        if spike_train is not None and spike_train.dim() == 2:
            spike_pattern_for_memory = spike_train.sum(dim=0).cpu().tolist()
            with self._lock:
                for t in range(spike_train.shape[0]):
                    pre = spike_train[t]
                    post = self.engine.forward(pre)
                    self.engine.update_traces(pre, post)
                    neural_intent += post.detach().cpu()
        else:
            # No Wernicke — generate random intent for testing
            neural_intent = torch.rand(self.engine.n_post) * 0.5

        # ── Layer 3 + Layer 4: Express with metacognitive validation ──
        final_response = ""
        for attempt in range(max_attempts):
            try:
                response = self.broca.express(
                    neural_intent=neural_intent,
                    original_query=user_input,
                )
            except Exception as e:
                response = f"[Express error: {e}]"

            if self._metacognitive_filter(response, user_input, attempt):
                final_response = response
                break

            # Failed validation — apply negative dopamine and inject
            # autonomous arousal: amplify neural intent randomly
            try:
                self.engine.apply_dopamine(reward=-0.3)
            except Exception:
                pass
            # Add noise to break out of stuck attractor
            neural_intent = neural_intent + torch.rand_like(neural_intent) * 0.3
        else:
            # All attempts failed — use last response anyway
            final_response = response

        # ── Record episodic memory ──
        with self._lock:
            self.episodic_memory.append({
                'timestamp': time.time(),
                'query': user_input[:200],
                'response': final_response[:500],
                'spike_pattern': spike_pattern_for_memory,
                'attempts': attempt + 1,
            })

        # ── Update efficiency metric ──
        self.efficiency_score = self._compute_efficiency()

        return final_response

    # ════════════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ════════════════════════════════════════════════════════════════

    def wake_up(self):
        """Load .mx state and start the DMN background thread."""
        print("[Limbic] Mergen is waking up...")
        self.load_state()

        self.is_running = True
        self._dmn_thread = threading.Thread(
            target=self._dmn_loop, daemon=True, name="Mergen-DMN"
        )
        self._dmn_thread.start()
        print(f"[Limbic] DMN thread started "
              f"(idle threshold: {self.idle_threshold}s).")

    def shutdown(self):
        """Stop background thread and persist state to .mx."""
        print("[Limbic] Mergen going to sleep...")
        self.is_running = False

        if self._dmn_thread is not None:
            self._dmn_thread.join(timeout=self.dmn_interval + 2)

        success = self.save_state()
        if success:
            print(f"[Limbic] ✓ State saved to {self.mx_path}")
            print(f"          Lifetime stats:")
            print(f"          - Responses: {self.total_responses}")
            print(f"          - DMN cycles: {self.dmn_cycles}")
            print(f"          - Passive rejections: {self.passive_rejections}")
            print(f"          - User corrections: {self.self_corrections}")
            print(f"          - Cumulative reward: {self.cumulative_reward:.2f}")
            print(f"          - Efficiency: {self.efficiency_score:.4f}")
        else:
            print(f"[Limbic] ✗ Save failed!")

    # ════════════════════════════════════════════════════════════════
    #  INSPECTION
    # ════════════════════════════════════════════════════════════════

    def introspect(self) -> Dict:
        """Return Mergen's current self-model — what it 'feels' right now."""
        with self._lock:
            return {
                'is_awake': self.is_running,
                'last_thought': self.last_thought,
                'idle_seconds': time.time() - self.last_interaction_time,
                'episodic_count': len(self.episodic_memory),
                'thought_count': len(self.internal_thoughts),
                'vocabulary_size': len(self.broca.concept_vocabulary),
                'total_responses': self.total_responses,
                'dmn_cycles': self.dmn_cycles,
                'passive_rejections': self.passive_rejections,
                'user_corrections': self.self_corrections,
                'cumulative_reward': self.cumulative_reward,
                'efficiency_score': self.efficiency_score,
                'recent_thoughts': [
                    t['thought'] for t in list(self.internal_thoughts)[-5:]
                ],
            }

    def __repr__(self):
        status = "AWAKE" if self.is_running else "ASLEEP"
        return (
            f"LimbicExecutiveLayer(\n"
            f"  status: {status}\n"
            f"  user: {self.user_id}\n"
            f"  mx_file: {self.mx_path}\n"
            f"  responses: {self.total_responses}\n"
            f"  dmn_cycles: {self.dmn_cycles}\n"
            f"  efficiency: {self.efficiency_score:.4f}\n"
            f"  cumulative_reward: {self.cumulative_reward:.2f}\n"
            f")"
        )


# ════════════════════════════════════════════════════════════════════
#  STANDALONE TEST (mock dependencies)
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  MERGEN — Layer 4: Limbic/Executive — Test")
    print("=" * 65)

    # Mock minimal engine and broca for testing
    class MockEngine:
        def __init__(self):
            self.n_pre = 64
            self.n_post = 32
            self.device = 'cpu'
            self.weights = torch.nn.Parameter(torch.randn(64, 32) * 0.1)
            self.eligibility = torch.zeros(64, 32)
            self.trace_pre = torch.zeros(64)
            self.trace_post = torch.zeros(32)
            self.firing_rate_ema = torch.zeros(32)
            self._step_count = 0
            self._da_event_count = 0
        def forward(self, x): return torch.relu(x @ self.weights)
        def update_traces(self, p, q): self._step_count += 1
        def apply_dopamine(self, reward): self._da_event_count += 1

    class MockBroca:
        def __init__(self):
            self.concept_vocabulary = [f"concept_{i:03d}" for i in range(32)]
            self._total_expressions = 0
        def express(self, neural_intent, original_query=None):
            self._total_expressions += 1
            return f"Mergen reflects on: {original_query}"
        def _is_passive_response(self, text):
            return "ready" in text.lower()

    engine = MockEngine()
    broca = MockBroca()
    limbic = LimbicExecutiveLayer(
        mergen_engine=engine,
        broca=broca,
        mx_path='/tmp/mergen_test.mx',
        user_id='burak',
        idle_threshold=2.0,
        dmn_interval=1.0,
    )

    limbic.wake_up()
    print(f"\n  Response 1: {limbic.respond('Mergen, who are you?')}")
    print(f"  Response 2: {limbic.respond('Tell me a thought.')}")

    print("\n  Letting DMN run for 5 seconds...")
    time.sleep(5)

    print(f"\n  Introspection: {limbic.introspect()}")
    limbic.shutdown()
    print("\n" + "=" * 65)
