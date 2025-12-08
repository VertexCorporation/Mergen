# utils/encoder.py
import torch
import numpy as np


class SpikeEncoder:
    """
    Converts symbolic data (e.g., text like "2 + 2") into
    spatiotemporal input currents for the cortical sheet.

    Design goals:
    - Each character gets a high-dimensional positive pattern.
    - Patterns are roughly normalized so no neuron dominates too much.
    - Time axis is divided into contiguous windows per character.
    """

    def __init__(self, n_neurons: int, device: str = "cpu") -> None:
        self.n_neurons = n_neurons
        self.device = device

        # Map from character -> index in embedding table
        self.char_map = {}
        self.embeddings = None  # (vocab_size, n_neurons)

        # Deterministic RNG for reproducibility
        self.rng = np.random.default_rng(42)

    # --------------------------------------------------------- #
    # INTERNAL: EMBEDDING TABLE
    # --------------------------------------------------------- #

    def _init_embeddings(self, text_corpus: str) -> None:
        """
        Lazily builds / extends a table of character embeddings.

        Each character is assigned a random, positive high-dimensional
        pattern. Values are in [0, 1], then rescaled to have controlled
        mean & variance.
        """
        # 1) Make sure every character in text_corpus has an index
        for ch in sorted(set(text_corpus)):
            if ch not in self.char_map:
                self.char_map[ch] = len(self.char_map)

        vocab_size = len(self.char_map) + 4  # small buffer

        # If embeddings already exist and are large enough, keep them
        if self.embeddings is not None and self.embeddings.shape[0] >= vocab_size:
            return

        # 2) Create / resize embedding matrix
        # New matrix: (vocab_size, n_neurons)
        # Values ~ Uniform[0, 1], then normalized per row.
        emb_np = self.rng.uniform(0.0, 1.0, size=(vocab_size, self.n_neurons))

        # Row-wise normalization: each char has similar energy
        # Avoid zeros by adding small epsilon.
        emb_norm = np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-8
        emb_np = emb_np / emb_norm

        # 3) Convert to torch
        self.embeddings = torch.tensor(
            emb_np,
            dtype=torch.float32,
            device=self.device,
        )

    # --------------------------------------------------------- #
    # PUBLIC: TEXT → (T, N) CURRENT
    # --------------------------------------------------------- #

    def encode_text(self, text: str, duration_steps: int) -> torch.Tensor:
        """
        Converts a string like "2 + 2" into a tensor of input currents.

        Args:
            text: string to encode.
            duration_steps: total number of time steps in the episode.

        Returns:
            Tensor of shape (T, N_NEURONS) with continuous input currents.
        """
        text = str(text)
        if len(text) == 0:
            raise ValueError("encode_text() received empty text.")

        # Ensure embeddings exist for every character in this text
        self._init_embeddings(text)

        # Allocate signal (Time, Neurons)
        signal = torch.zeros(
            (duration_steps, self.n_neurons),
            device=self.device,
            dtype=torch.float32,
        )

        # Divide time axis evenly across characters
        steps_per_char = max(duration_steps // len(text), 1)

        # Base gain: how strongly each character drives the cortex
        base_gain = 3.0  # was ~1.5; bump it up for guaranteed spikes

        for i, ch in enumerate(text):
            char_idx = self.char_map[ch]
            pattern = self.embeddings[char_idx]  # (N_NEURONS,)

            start = i * steps_per_char
            end = min((i + 1) * steps_per_char, duration_steps)

            if start >= duration_steps:
                break  # text longer than time; ignore late chars

            # Inject current: positive pattern * gain
            # Broadcasting over time slice: (end-start, N_NEURONS)
            signal[start:end, :] += pattern * base_gain

        # Optional: add tiny uniform noise so not all neurons are identical
        noise_level = 0.05
        if noise_level > 0.0:
            noise = (torch.rand_like(signal) - 0.5) * (2.0 * noise_level)
            # Keep everything non-negative
            signal = torch.clamp(signal + noise, min=0.0)

        return signal