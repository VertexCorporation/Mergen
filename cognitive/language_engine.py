"""
╔══════════════════════════════════════════════════════════════════════╗
║       MERGEN V3 — LANGUAGE ENGINE (Pure SNN, No External LLM)       ║
║                                                                      ║
║  "Mergen now speaks with its own neurons."                          ║
║                                                                      ║
║  Author:  Vertex Corporation — Mergen V3                            ║
║  License: Apache-2.0                                                ║
║  Scale:   200k motor neurons → vocabulary                           ║
╚══════════════════════════════════════════════════════════════════════╝

BIOLOGICAL FOUNDATION:
━━━━━━━━━━━━━━━━━━━━━
In the human brain, Broca's area (BA 44/45) does not contain a
"dictionary." Instead, each learned word corresponds to a unique
pattern of coordinated neuron firing — a "spike signature."

When we speak:
    1. Motor cortex generates a spatiotemporal spike pattern
    2. Pattern matches a learned signature for a specific word
    3. Articulatory muscles execute the corresponding motor program

Mergen's Language Engine replicates this without any LLM:
    • Each word in the vocabulary = unique population spike signature
    • Speaking = pattern matching between motor layer & signatures
    • Learning = strengthening word↔signature bonds via Hebbian STDP
    • Dreaming = consolidating these bonds offline (via dream.py)

NO LLM. NO API. NO EXTERNAL MODEL.
Just 200,000 motor neurons learning to fire in meaningful patterns.

PIPELINE:
━━━━━━━━━
    ┌────────────────────────┐
    │  Motor Layer           │  (400×500 = 200k neurons)
    │  spike activations     │
    └──────┬─────────────────┘
           ▼
    ┌────────────────────────┐
    │  Spike-to-Signature    │  Compare against learned word
    │  Pattern Matching      │  signatures via cosine/dot
    └──────┬─────────────────┘
           ▼
    ┌────────────────────────┐
    │  Readout Logic         │  Softmax + temperature + top-k
    │  Word Selection        │  Probabilistic or greedy
    └──────┬─────────────────┘
           ▼
    ┌────────────────────────┐
    │  Sequence Builder      │  Multi-step decoding for sentences
    │  (Temporal Integration)│  Motor layer ticks per token
    └──────┬─────────────────┘
           ▼
    ┌────────────────────────┐
    │  Natural Text          │  Pure Mergen speech
    └────────────────────────┘
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from collections import defaultdict


class LanguageEngine(nn.Module):
    """
    Pure SNN Neural Decoder — Mergen's Voice.

    Takes motor layer spike activations and converts them to words
    using learned population-level spike signatures. No external LLM.

    Each word w has a learned signature S_w ∈ ℝ^(motor_layer_size)
    When the motor layer produces activation A, the most likely word is:

        word = argmax_w (cosine_similarity(A, S_w) * confidence_w)

    Signatures are LEARNED — not hand-crafted:
        • During supervised learning, motor activity for a known word
          strengthens that word's signature (Hebbian rule).
        • During dream.py, signatures are consolidated via STDP replay.
    """

    def __init__(
        self,
        motor_layer_size: int = 200_000,
        motor_rows: int = 400,
        motor_cols: int = 500,
        vocabulary: Optional[List[str]] = None,
        vocabulary_path: Optional[str] = None,
        signature_path: str = './signatures.pt',
        temperature: float = 0.9,
        top_k: int = 40,
        activation_threshold: float = 0.1,
        signature_lr: float = 0.01,
        device: str = 'cpu',
    ):
        super().__init__()

        self.motor_layer_size = motor_layer_size
        self.motor_rows = motor_rows
        self.motor_cols = motor_cols
        self.temperature = temperature
        self.top_k = top_k
        self.activation_threshold = activation_threshold
        self.signature_lr = signature_lr
        self.device = device
        self.signature_path = Path(signature_path)

        # ── Load vocabulary ──
        self.vocabulary = self._load_vocabulary(vocabulary, vocabulary_path)
        self.vocab_size = len(self.vocabulary)
        self.word_to_id = {w: i for i, w in enumerate(self.vocabulary)}

        # ── Initialize spike signatures ──
        # Each word gets a unique random signature, refined via learning
        # Shape: (vocab_size, motor_layer_size)
        if self.signature_path.exists():
            self._load_signatures()
        else:
            self._init_signatures()

        # ── Per-word confidence tracking ──
        # How well-learned is each word? Starts low, grows with exposure
        self.register_buffer(
            'confidences',
            torch.ones(self.vocab_size, device=device) * 0.1
        )
        # How often each word has been "spoken" (for normalization)
        self.register_buffer(
            'utterance_counts',
            torch.zeros(self.vocab_size, device=device)
        )

        # ── Readout weights (learnable fine-tuning layer) ──
        # This is a small trainable projection that can be optimized
        # during dream phase without changing the signatures directly.
        self.readout_bias = nn.Parameter(torch.zeros(self.vocab_size))

        # ── Special tokens ──
        self.BOS_ID = self.word_to_id.get('<bos>', 0)
        self.EOS_ID = self.word_to_id.get('<eos>', 1)
        self.PAD_ID = self.word_to_id.get('<pad>', 2)
        self.UNK_ID = self.word_to_id.get('<unk>', 3)

        # ── Telemetry ──
        self._total_words_spoken = 0
        self._total_training_updates = 0
        self._last_output = ""

    # ════════════════════════════════════════════════════════════════
    #  VOCABULARY LOADING
    # ════════════════════════════════════════════════════════════════

    def _load_vocabulary(
        self,
        vocabulary: Optional[List[str]],
        vocabulary_path: Optional[str],
    ) -> List[str]:
        """Load vocabulary from argument, file, or use minimal default."""
        if vocabulary is not None:
            vocab = list(vocabulary)
        elif vocabulary_path and Path(vocabulary_path).exists():
            with open(vocabulary_path, 'r', encoding='utf-8') as f:
                if vocabulary_path.endswith('.json'):
                    vocab = json.load(f)
                else:
                    vocab = [line.strip() for line in f if line.strip()]
        else:
            # Minimal bootstrap vocabulary
            # In production, load from config.py VOCABULARY constant
            vocab = (
                ['<bos>', '<eos>', '<pad>', '<unk>'] +
                list("abcçdefgğhıijklmnoöprsştuüvyz") +
                ['.', ',', '?', '!', ' '] +
                # Common Turkish & English stubs
                ['ben', 'sen', 'o', 'biz', 'siz', 'onlar',
                 'i', 'you', 'he', 'she', 'we', 'they',
                 'evet', 'hayır', 'yes', 'no',
                 'merhaba', 'hello', 'mergen', 'think',
                 'düşünüyorum', 'dir', 'dır', 'the', 'is']
            )
        return vocab

    # ════════════════════════════════════════════════════════════════
    #  SIGNATURE MANAGEMENT
    # ════════════════════════════════════════════════════════════════

    def _init_signatures(self):
        """Initialize unique random spike signatures for each word."""
        # Orthogonal initialization gives better separation between words
        # Shape: (vocab_size, motor_layer_size)
        sig = torch.randn(
            self.vocab_size, self.motor_layer_size, device=self.device
        )
        # L2 normalize each signature to unit vector
        sig = sig / (sig.norm(dim=1, keepdim=True) + 1e-8)
        self.register_buffer('signatures', sig)
        print(f"[Language] Initialized {self.vocab_size} random signatures "
              f"over {self.motor_layer_size} motor neurons.")

    def _load_signatures(self):
        """Load pre-trained signatures from disk."""
        try:
            sig = torch.load(self.signature_path, map_location=self.device)
            if isinstance(sig, dict):
                sig_tensor = sig.get('signatures')
                if 'confidences' in sig:
                    self.register_buffer('confidences',
                                         sig['confidences'].to(self.device))
            else:
                sig_tensor = sig

            # Check shape compatibility
            if sig_tensor.shape != (self.vocab_size, self.motor_layer_size):
                print(f"[Language] ⚠ Signature shape mismatch "
                      f"({sig_tensor.shape} vs "
                      f"({self.vocab_size}, {self.motor_layer_size})) "
                      f"— reinitializing.")
                self._init_signatures()
                return

            self.signatures = sig_tensor.to(self.device).float()
            print(f"[Language] ✓ Loaded signatures from "
                  f"{self.signature_path}")
        except Exception as e:
            print(f"[Language] ⚠ Signature load error: {e} — reinitializing.")
            self._init_signatures()

    def save_signatures(self):
        """Persist signatures and confidences."""
        try:
            state = {
                'signatures': self.signatures.cpu(),
                'confidences': self.confidences.cpu(),
                'utterance_counts': self.utterance_counts.cpu(),
                'vocabulary': self.vocabulary,
            }
            torch.save(state, self.signature_path)
            print(f"[Language] ✓ Signatures saved to {self.signature_path}")
            return True
        except Exception as e:
            print(f"[Language] ⚠ Save error: {e}")
            return False

    # ════════════════════════════════════════════════════════════════
    #  SPIKE-TO-WORD CONVERSION (CORE DECODER)
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def decode_spike_pattern(
        self,
        motor_activation: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, float, Dict]:
        """
        Convert a motor layer activation pattern to a single word.

        Args:
            motor_activation: (motor_layer_size,) spike pattern
            deterministic: If True, greedy argmax. If False, sample from
                          top-k distribution with temperature.

        Returns:
            word_id: Index of chosen word
            confidence: Match quality (cosine similarity)
            info: Dict with top candidates for inspection
        """
        # Flatten if 2D motor layer (400×500) was passed
        if motor_activation.dim() == 2:
            motor_activation = motor_activation.flatten()

        # Pad/truncate to motor_layer_size
        if motor_activation.shape[0] < self.motor_layer_size:
            pad = torch.zeros(
                self.motor_layer_size - motor_activation.shape[0],
                device=self.device,
            )
            motor_activation = torch.cat([motor_activation, pad])
        elif motor_activation.shape[0] > self.motor_layer_size:
            motor_activation = motor_activation[:self.motor_layer_size]

        # Normalize activation
        norm = motor_activation.norm() + 1e-8
        act_normalized = motor_activation / norm

        # Cosine similarity with all signatures: (vocab_size,)
        # signatures shape: (vocab_size, motor_layer_size)
        similarities = torch.matmul(self.signatures, act_normalized)

        # Ensure parameters are on the same device
        device_sim = similarities.device
        if self.confidences.device != device_sim:
            self.confidences = self.confidences.to(device_sim)
        if self.readout_bias.device != device_sim:
            self.readout_bias = nn.Parameter(self.readout_bias.to(device_sim))

        # Apply confidence weighting
        # Words we've spoken often get slight boost (like motor habit)
        logits = similarities + 0.1 * self.confidences + self.readout_bias

        # Apply temperature
        logits = logits / max(self.temperature, 0.01)

        # Top-k filtering
        if self.top_k > 0 and self.top_k < self.vocab_size:
            topk_vals, topk_idx = torch.topk(logits, self.top_k)
            mask = torch.full_like(logits, float('-inf'))
            mask[topk_idx] = topk_vals
            logits = mask

        # Softmax → probabilities
        probs = torch.softmax(logits, dim=0)

        if deterministic:
            word_id = int(torch.argmax(probs).item())
        else:
            word_id = int(torch.multinomial(probs, 1).item())

        confidence = float(similarities[word_id].item())

        # Top-3 for inspection
        top3_vals, top3_idx = torch.topk(similarities, min(3, self.vocab_size))
        info = {
            'top_candidates': [
                (self.vocabulary[int(i)], float(v))
                for i, v in zip(top3_idx, top3_vals)
            ],
            'activation_magnitude': float(norm.item()),
        }

        return word_id, confidence, info

    # ════════════════════════════════════════════════════════════════
    #  SEQUENCE GENERATION (MULTI-WORD OUTPUT)
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def speak(
        self,
        motor_sequence: torch.Tensor,
        max_length: int = 64,
        deterministic: bool = False,
    ) -> str:
        """
        MAIN API — Convert a motor layer sequence to natural text.

        Args:
            motor_sequence: Can be:
                - (motor_layer_size,) — single timestep → single word
                - (T, motor_layer_size) — T timesteps → sequence of T words
                - (motor_rows, motor_cols) — 2D → flattened → single word
            max_length: Maximum words to produce
            deterministic: Greedy vs sampled decoding

        Returns:
            Generated text string
        """
        # Handle 2D motor layer (400×500)
        if motor_sequence.dim() == 2 and motor_sequence.shape[0] == self.motor_rows:
            motor_sequence = motor_sequence.flatten().unsqueeze(0)

        # Handle single timestep
        if motor_sequence.dim() == 1:
            motor_sequence = motor_sequence.unsqueeze(0)

        words = []
        for t in range(min(motor_sequence.shape[0], max_length)):
            activation = motor_sequence[t]

            # Threshold: if motor layer is too quiet, skip this tick
            if activation.abs().max() < self.activation_threshold:
                continue

            word_id, confidence, info = self.decode_spike_pattern(
                activation, deterministic=deterministic
            )

            word = self.vocabulary[word_id]

            # Stop on EOS
            if word_id == self.EOS_ID:
                break

            # Skip padding/unknown
            if word_id in (self.PAD_ID, self.UNK_ID):
                continue

            words.append(word)
            self._total_words_spoken += 1

            # Update utterance count (for confidence growth)
            self.utterance_counts[word_id] += 1

        text = self._detokenize(words)
        self._last_output = text
        return text

    def _detokenize(self, words: List[str]) -> str:
        """Smart joining: no space before punctuation, handle subwords."""
        result = []
        for w in words:
            if not result:
                result.append(w)
            elif w in ('.', ',', '?', '!', ';', ':'):
                result[-1] += w
            elif w.startswith('##'):  # BERT-style subword
                result[-1] += w[2:]
            else:
                result.append(w)
        return ' '.join(result).strip()

    # ════════════════════════════════════════════════════════════════
    #  AUTONOMOUS TRAINING (NO-LLM LEARNING)
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def strengthen_association(
        self,
        word: Union[str, int],
        motor_activation: torch.Tensor,
        learning_rate: Optional[float] = None,
    ):
        """
        Strengthen the bond between a word and a motor activation pattern.

        This is called:
        • During supervised learning when user provides a correct word
        • During dream.py replay phase
        • When the engine speaks successfully (positive reinforcement)

        Hebbian rule: S_w ← S_w + η · (A - S_w)
        (moves signature toward the activation pattern)

        Args:
            word: Word string or vocabulary index
            motor_activation: Motor layer pattern that produced this word
            learning_rate: Override default signature_lr
        """
        # Resolve word → id
        if isinstance(word, str):
            if word not in self.word_to_id:
                # New word — add to vocabulary dynamically
                self.word_to_id[word] = len(self.vocabulary)
                self.vocabulary.append(word)
                # Extend signatures with random initialization
                new_sig = torch.randn(
                    1, self.motor_layer_size, device=self.device
                )
                new_sig = new_sig / (new_sig.norm() + 1e-8)
                self.signatures = torch.cat([self.signatures, new_sig], dim=0)
                # Extend confidences
                self.confidences = torch.cat([
                    self.confidences,
                    torch.tensor([0.1], device=self.device)
                ])
                self.utterance_counts = torch.cat([
                    self.utterance_counts,
                    torch.tensor([0.0], device=self.device)
                ])
                # Extend readout bias
                with torch.no_grad():
                    new_bias = torch.zeros(1, device=self.readout_bias.device)
                    self.readout_bias = nn.Parameter(
                        torch.cat([self.readout_bias.data, new_bias])
                    )
                self.vocab_size = len(self.vocabulary)
            word_id = self.word_to_id[word]
        else:
            word_id = int(word)

        # Prepare activation
        if motor_activation.dim() == 2:
            motor_activation = motor_activation.flatten()
        if motor_activation.shape[0] != self.motor_layer_size:
            # Resize
            if motor_activation.shape[0] < self.motor_layer_size:
                pad = torch.zeros(
                    self.motor_layer_size - motor_activation.shape[0],
                    device=self.device,
                )
                motor_activation = torch.cat([motor_activation, pad])
            else:
                motor_activation = motor_activation[:self.motor_layer_size]

        # Normalize
        act_norm = motor_activation / (motor_activation.norm() + 1e-8)

        # Hebbian update: move signature toward activation
        lr = learning_rate or self.signature_lr
        self.signatures[word_id] = (
            self.signatures[word_id] + lr * (act_norm - self.signatures[word_id])
        )
        # Re-normalize signature to unit sphere
        self.signatures[word_id] = (
            self.signatures[word_id] /
            (self.signatures[word_id].norm() + 1e-8)
        )

        # Increase confidence
        self.confidences[word_id] = min(
            1.0, self.confidences[word_id].item() + lr * 0.5
        )

        self._total_training_updates += 1

    @torch.no_grad()
    def weaken_association(
        self,
        word: Union[str, int],
        motor_activation: torch.Tensor,
        learning_rate: Optional[float] = None,
    ):
        """
        Weaken a word-pattern bond. Called on user corrections.

        Anti-Hebbian: signature moves AWAY from the activation.
        """
        if isinstance(word, str):
            if word not in self.word_to_id:
                return
            word_id = self.word_to_id[word]
        else:
            word_id = int(word)

        if motor_activation.dim() == 2:
            motor_activation = motor_activation.flatten()
        if motor_activation.shape[0] != self.motor_layer_size:
            motor_activation = motor_activation[:self.motor_layer_size]

        act_norm = motor_activation / (motor_activation.norm() + 1e-8)
        lr = learning_rate or self.signature_lr

        # Move signature AWAY from wrong activation
        self.signatures[word_id] = (
            self.signatures[word_id] - lr * (act_norm - self.signatures[word_id])
        )
        self.signatures[word_id] = (
            self.signatures[word_id] /
            (self.signatures[word_id].norm() + 1e-8)
        )

        # Decrease confidence slightly
        self.confidences[word_id] = max(
            0.01, self.confidences[word_id].item() - lr * 0.3
        )

    # ════════════════════════════════════════════════════════════════
    #  DREAM INTEGRATION (called by dream.py)
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def dream_consolidate(
        self,
        replay_episodes: List[Dict],
        cycles: int = 1000,
    ):
        """
        Offline signature consolidation during sleep.

        Called by dream.py during NREM phase with episodic memories
        containing (word, motor_activation) pairs.

        Each replay slightly strengthens the learned signatures,
        especially for low-confidence words.

        Args:
            replay_episodes: List of {'word': str, 'motor_activation': tensor}
            cycles: Number of replay iterations
        """
        if not replay_episodes:
            return

        for cycle in range(cycles):
            episode = replay_episodes[cycle % len(replay_episodes)]
            word = episode.get('word')
            activation = episode.get('motor_activation')

            if word is None or activation is None:
                continue

            # Lower learning rate during dream (gentle consolidation)
            # But higher for low-confidence words
            if isinstance(word, str) and word in self.word_to_id:
                word_id = self.word_to_id[word]
                confidence = self.confidences[word_id].item()
                dream_lr = self.signature_lr * (2.0 - confidence) * 0.5
            else:
                dream_lr = self.signature_lr * 0.5

            self.strengthen_association(word, activation, learning_rate=dream_lr)

    # ════════════════════════════════════════════════════════════════
    #  TELEMETRY & INTROSPECTION
    # ════════════════════════════════════════════════════════════════

    def get_telemetry(self) -> Dict:
        return {
            'vocab_size': self.vocab_size,
            'motor_layer_size': self.motor_layer_size,
            'total_words_spoken': self._total_words_spoken,
            'total_training_updates': self._total_training_updates,
            'last_output': self._last_output[:100],
            'avg_confidence': self.confidences.mean().item(),
            'max_confidence': self.confidences.max().item(),
            'min_confidence': self.confidences.min().item(),
            'most_confident_words': [
                self.vocabulary[i] for i in
                torch.topk(self.confidences, min(5, self.vocab_size))[1].tolist()
            ],
            'signature_separation': self._compute_signature_separation(),
        }

    @torch.no_grad()
    def _compute_signature_separation(self) -> float:
        """
        Measure how distinct the learned signatures are.
        Higher = words more clearly separated = better decoding.
        """
        if self.vocab_size < 2:
            return 0.0
        # Sample up to 100 pairs for efficiency
        n_sample = min(100, self.vocab_size)
        idx = torch.randperm(self.vocab_size)[:n_sample]
        sample = self.signatures[idx]
        # Mean pairwise cosine similarity
        sim_matrix = torch.matmul(sample, sample.T)
        # Mask diagonal (self-similarity = 1)
        mask = 1.0 - torch.eye(n_sample, device=self.device)
        mean_sim = (sim_matrix * mask).sum() / mask.sum()
        # Separation = 1 - mean_similarity (higher is better)
        return float(1.0 - mean_sim.item())

    def __repr__(self):
        return (
            f"LanguageEngine(\n"
            f"  motor_layer: {self.motor_rows}×{self.motor_cols} "
            f"= {self.motor_layer_size:,} neurons\n"
            f"  vocabulary: {self.vocab_size:,} words\n"
            f"  signatures: shape "
            f"{tuple(self.signatures.shape)}\n"
            f"  avg_confidence: "
            f"{self.confidences.mean().item():.3f}\n"
            f"  signature_separation: "
            f"{self._compute_signature_separation():.3f}\n"
            f"  words_spoken: {self._total_words_spoken:,}\n"
            f"  training_updates: {self._total_training_updates:,}\n"
            f")"
        )


# ════════════════════════════════════════════════════════════════════
#  STANDALONE DEMO
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  MERGEN V3 — Language Engine (Pure SNN Decoder)")
    print("  No Ollama. No LLM. Just neurons.")
    print("=" * 65)

    vocab = [
        '<bos>', '<eos>', '<pad>', '<unk>',
        'merhaba', 'ben', 'mergen', 'düşünüyorum',
        'evet', 'hayır', 'bilmiyorum', 'öğreniyorum',
        '.', ',', '?',
        'hello', 'i', 'am', 'think', 'yes', 'no', 'learn',
    ]

    engine = LanguageEngine(
        motor_layer_size=2000,  # scaled for demo
        motor_rows=40,
        motor_cols=50,
        vocabulary=vocab,
        temperature=0.9,
        top_k=10,
        device='cpu',
    )
    print(f"\n{engine}\n")

    # Simulate motor layer activity for "merhaba"
    print("  ── Phase 1: Learning ──")
    target_word = 'merhaba'
    motor_pattern = torch.randn(2000) * 0.1
    motor_pattern[100:200] = 1.0  # Characteristic firing

    # Train: strengthen association (repeated exposure)
    for _ in range(20):
        engine.strengthen_association(target_word, motor_pattern)

    print(f"  Trained '{target_word}' with characteristic pattern.")

    # Test: does the same pattern produce the word?
    print("\n  ── Phase 2: Speaking ──")
    word_id, conf, info = engine.decode_spike_pattern(motor_pattern)
    print(f"  Decoded word: '{engine.vocabulary[word_id]}' "
          f"(confidence: {conf:.3f})")
    print(f"  Top candidates: {info['top_candidates']}")

    # Test multi-word sequence
    print("\n  ── Phase 3: Sentence Generation ──")
    sequence = torch.randn(5, 2000) * 0.1
    for t in range(5):
        sequence[t, 100 + t*50:200 + t*50] = 1.0
    text = engine.speak(sequence, max_length=10)
    print(f"  Generated: '{text}'")

    # Save
    engine.signature_path = Path('/tmp/mergen_signatures_test.pt')
    engine.save_signatures()

    print(f"\n{engine}")
    print("=" * 65)
