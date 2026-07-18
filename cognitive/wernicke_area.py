"""
╔══════════════════════════════════════════════════════════════════════╗
║         MERGEN — LAYER 1: WERNICKE AREA (Sensory Perception)        ║
║                                                                      ║
║  "Mergen'in duyma ve anlamlandırma katmanı."                        ║
║                                                                      ║
║  Author:  Vertex Corporation — Mergen Project                       ║
║  License: Apache-2.0                                                ║
╚══════════════════════════════════════════════════════════════════════╝

BIOLOGICAL ROLE:
━━━━━━━━━━━━━━━━
The Wernicke Area in the human brain is responsible for language
comprehension — converting raw auditory/visual symbols into semantic
meaning that can be processed by deeper cortical regions.

In Mergen, this module performs the equivalent function:
    Raw Text → Semantic Vectors → Neural Spike Trains

It is the FIRST contact point between the external world and Mergen's
biological cognitive engine. It must produce spike patterns that
Layer 2 (Mergen Engine with Hebbian/STDP learning) can process.

PIPELINE:
━━━━━━━━━
    ┌─────────────┐
    │  Raw Text   │  "What is consciousness?"
    └──────┬──────┘
           ▼
    ┌─────────────────────────┐
    │  Tokenization &          │   Local sentence-transformers
    │  Semantic Embedding      │   (no API calls — fully local)
    └──────┬───────────────────┘
           ▼  
    ┌─────────────────────────┐
    │  Normalization           │   L2-normalize to unit sphere
    │  & Quantization          │   for stable spike encoding
    └──────┬───────────────────┘
           ▼
    ┌─────────────────────────┐
    │  Spike Encoding          │   Three encoding modes:
    │  (Rate / Temporal /      │   - Rate (Poisson)
    │   Population)            │   - Temporal (TTFS)
    └──────┬───────────────────┘   - Population (Gaussian)
           ▼
    ┌─────────────────────────┐
    │  Spike Train Tensor     │   shape: (T_steps, N_neurons)
    │  → Layer 2 (Mergen)     │   binary {0, 1}
    └─────────────────────────┘

ENCODING STRATEGIES:
━━━━━━━━━━━━━━━━━━━
1. RATE CODING (Poisson)
   - Each embedding dimension → firing rate of one neuron
   - Higher value = higher rate of stochastic spikes
   - Biology: Mimics primary sensory cortex rate coding
   - Pros: Robust to noise, simple interpretation
   - Cons: Loses temporal precision

2. TEMPORAL CODING (Time-To-First-Spike)
   - Higher value = earlier spike in time window
   - Biology: Mimics rapid sensory processing in retina/cochlea
   - Pros: Energy efficient, fast information transfer
   - Cons: Sensitive to timing noise

3. POPULATION CODING (Gaussian Tuning)
   - Each value mapped across multiple neurons via Gaussian curves
   - Biology: Mimics cortical tuning curves (orientation, frequency)
   - Pros: High precision, biologically realistic
   - Cons: Requires more neurons per dimension
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import Optional, Union, List, Dict


class WernickeArea(nn.Module):
    """
    MERGEN LAYER 1 — Sensory Input & Semantic Perception

    Converts raw text into biologically-plausible spike trains
    that can drive the Hebbian/STDP learning engine in Layer 2.

    Usage:
        wernicke = WernickeArea(
            embedding_model='sentence-transformers/all-MiniLM-L6-v2',
            n_neurons=384,
            time_window=50,
            encoding='rate'
        )

        spikes = wernicke.perceive("What is consciousness?")
        # spikes.shape == (50, 384) — feed to MergenEngine

    Args:
        embedding_model: HuggingFace sentence-transformers model name.
                        Must run locally — no API calls.
        n_neurons:      Number of input neurons in Layer 2.
                        Should match embedding dimensionality for
                        rate/temporal encoding, or be a multiple of it
                        for population encoding.
        time_window:    Number of simulation time steps per text input.
                        Larger = more spike resolution.
        encoding:       'rate', 'temporal', or 'population'
        max_rate:       Maximum firing rate (spikes per time step) for
                        rate coding. Should be < 1.0 to leave room for
                        sparse activity.
        device:         'cpu' or 'cuda'
    """

    def __init__(
        self,
        embedding_model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        n_neurons: int = 768,
        time_window: int = 50,
        encoding: str = 'rate',
        max_rate: float = 0.4,
        population_size: int = 4,
        device: str = 'cpu',
        embed_cache_size: int = 256,
    ):
        super().__init__()

        self.embedding_model_name = embedding_model
        self.n_neurons = n_neurons
        self.time_window = time_window
        self.encoding = encoding.lower()
        self.max_rate = max_rate
        self.population_size = population_size
        self.device = device

        # LRU embedding cache — tekrar eden girdilerde encoder.encode() atlanır.
        # embed_cache_size=0 → önbellek devre dışı.
        self._embed_cache_size: int = max(0, embed_cache_size)
        self._embed_cache: OrderedDict = OrderedDict()   # key: str → value: Tensor (embedding_dim,)
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # Validate encoding
        if self.encoding not in ('rate', 'temporal', 'population'):
            raise ValueError(
                f"Encoding must be 'rate', 'temporal', or 'population', "
                f"got '{encoding}'"
            )

        # Load local sentence-transformer model
        # Lazy import to avoid forcing dependency at module load
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(embedding_model, device=device)
            self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers required. "
                "Install: pip install sentence-transformers"
            )

        # For population coding: precompute Gaussian centers
        if self.encoding == 'population':
            # Each embedding dim is encoded by `population_size` neurons
            # Centers spread across [-1, 1] (normalized embedding range)
            self.register_buffer(
                'pop_centers',
                torch.linspace(-1.0, 1.0, population_size, device=device)
            )
            # Width of each Gaussian tuning curve
            self.pop_sigma = 2.0 / population_size

            # Total neurons needed for population coding
            required = self.embedding_dim * population_size
            if n_neurons < required:
                print(
                    f"⚠ Warning: n_neurons ({n_neurons}) < required "
                    f"({required}) for population coding. Truncating."
                )

        # Telemetry
        self._total_perceptions = 0
        self._last_input_text = ""
        self._last_spike_count = 0
        self._last_active_neurons = 0


    # ════════════════════════════════════════════════════════════════
    #  EMBEDDING CACHE — LRU Önbellekli Tekli Gömme
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _embed_single_cached(self, text: str) -> torch.Tensor:
        """
        Tek bir metin için LRU önbellekli gömme.

        Önbellekte bulunursa encoder.encode() çağrılmaz (CPU tasarrufu).
        Önbellekte yoksa encode edilir, önbelleğe eklenir ve en eski
        giriş çıkarılır (LRU eviction).

        Args:
            text: Tek bir girdi metni

        Returns:
            Embedding tensoru, shape (embedding_dim,) — L2-normalized.
        """
        if self._embed_cache_size > 0 and text in self._embed_cache:
            # Önbellekte bulundu → en sona taşı (most-recently-used)
            self._embed_cache.move_to_end(text)
            self._cache_hits += 1
            return self._embed_cache[text]

        # Önbellekte yok → gerçekten encode et
        embedding = self.encoder.encode(
            [text],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)
        embedding = embedding[0].to(self.device)   # (embedding_dim,)

        self._cache_misses += 1

        if self._embed_cache_size > 0:
            self._embed_cache[text] = embedding
            self._embed_cache.move_to_end(text)
            # Kapasite aşıldıysa en eski girişi çıkar (LRU eviction)
            while len(self._embed_cache) > self._embed_cache_size:
                self._embed_cache.popitem(last=False)

        return embedding

    # ════════════════════════════════════════════════════════════════
    #  EMBEDDING — Text to Vector
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def embed(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Convert raw text into semantic embedding vector(s).

        Uses local sentence-transformers — NO API calls, NO internet.

        Args:
            text: Single string or list of strings

        Returns:
            Embeddings tensor, shape (batch, embedding_dim)
            L2-normalized to unit sphere for stable encoding.
        """
        if isinstance(text, str):
            text = [text]

        # Encode via local model
        embeddings = self.encoder.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,  # L2 norm for stability
            show_progress_bar=False,
        )

        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)

        return embeddings.to(self.device)

    # ════════════════════════════════════════════════════════════════
    #  ENCODING STRATEGIES — Vector to Spikes
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _rate_encode(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        RATE CODING: Map vector values to Poisson firing rates.

        Each embedding dimension drives one neuron. The value's
        magnitude determines spike probability per time step.

        Args:
            embedding: (embedding_dim,) — normalized vector

        Returns:
            Spike train (time_window, n_neurons) — binary
        """
        # Dual-rail encoding: [relu(x), relu(-x)]
        pos = torch.relu(embedding)
        neg = torch.relu(-embedding)
        dual_embedding = torch.cat([pos, neg], dim=-1)
        rates = dual_embedding * self.max_rate

        # Pad/truncate to n_neurons
        if rates.shape[0] < self.n_neurons:
            padding = torch.zeros(
                self.n_neurons - rates.shape[0], device=self.device
            )
            rates = torch.cat([rates, padding])
        else:
            rates = rates[:self.n_neurons]

        # Generate Poisson spikes for each time step
        # spike[t, n] = 1 if random() < rate[n] else 0
        random_values = torch.rand(
            self.time_window, self.n_neurons, device=self.device
        )
        spikes = (random_values < rates.unsqueeze(0)).float()

        return spikes

    @torch.no_grad()
    def _temporal_encode(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        TEMPORAL CODING (Time-To-First-Spike):

        Higher values fire EARLIER in the time window.
        Lower values fire LATER (or not at all).

        Biology: Retinal ganglion cells use this for fast processing.

        Args:
            embedding: (embedding_dim,) — normalized vector

        Returns:
            Spike train (time_window, n_neurons) — binary
        """
        # Dual-rail encoding
        pos = torch.relu(embedding)
        neg = torch.relu(-embedding)
        dual_embedding = torch.cat([pos, neg], dim=-1)
        magnitudes = dual_embedding

        # Pad/truncate
        if magnitudes.shape[0] < self.n_neurons:
            padding = torch.zeros(
                self.n_neurons - magnitudes.shape[0], device=self.device
            )
            magnitudes = torch.cat([magnitudes, padding])
        else:
            magnitudes = magnitudes[:self.n_neurons]

        # Compute spike time: higher value → earlier spike
        # spike_time = (1 - magnitude) * time_window
        spike_times = ((1.0 - magnitudes) * self.time_window).long()
        spike_times.clamp_(0, self.time_window - 1)

        # Build spike train
        spikes = torch.zeros(
            self.time_window, self.n_neurons, device=self.device
        )

        # Only neurons with sufficient magnitude actually fire
        threshold = 0.05
        active = magnitudes > threshold
        # SORUN-08 FIX: Python for-dongusu vektorize edildi.
        # nonzero() aktif noronlarin indekslerini dondurur;
        # gelismis indeksleme scatter mantigiyla spikes matrisini doldurur.
        active_idx = active.nonzero(as_tuple=True)[0]          # (k,)  k = aktif noron sayisi
        if active_idx.numel() > 0:
            times_for_active = spike_times[active_idx]          # (k,)  her aktif noronun spike zamani
            spikes[times_for_active, active_idx] = 1.0          # tek seferde yerlesir

        return spikes

    @torch.no_grad()
    def _population_encode(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        POPULATION CODING: Each value spreads across multiple neurons
        via Gaussian tuning curves.

        For each embedding dimension, `population_size` neurons compete.
        The neuron whose center is closest to the value fires most.

        Biology: Mimics cortical orientation columns, frequency tuning.

        Args:
            embedding: (embedding_dim,) — normalized vector

        Returns:
            Spike train (time_window, n_neurons) — binary
        """
        # Compute Gaussian response for each (dim, center) pair
        # responses[d, c] = exp(-(value[d] - center[c])^2 / (2*sigma^2))
        values = embedding.unsqueeze(1)  # (dim, 1)
        centers = self.pop_centers.unsqueeze(0)  # (1, pop_size)

        diff = values - centers  # (dim, pop_size)
        responses = torch.exp(
            -(diff ** 2) / (2 * self.pop_sigma ** 2)
        )  # (dim, pop_size)

        # Flatten to (dim * pop_size,) — one neuron per (dim, center)
        rates = responses.flatten() * self.max_rate

        # Pad/truncate
        if rates.shape[0] < self.n_neurons:
            padding = torch.zeros(
                self.n_neurons - rates.shape[0], device=self.device
            )
            rates = torch.cat([rates, padding])
        else:
            rates = rates[:self.n_neurons]

        # Generate Poisson spikes
        random_values = torch.rand(
            self.time_window, self.n_neurons, device=self.device
        )
        spikes = (random_values < rates.unsqueeze(0)).float()

        return spikes

    # ════════════════════════════════════════════════════════════════
    #  PERCEPTION — The main entry point
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def perceive(self, text: str) -> torch.Tensor:
        """
        MAIN ENTRY POINT — Convert text to spike train.

        This is the function Layer 2 (MergenEngine) calls to
        receive sensory input. The output spike train can be
        directly fed to HybridHebbianLearner.update_traces().

        Args:
            text: Raw input text from user

        Returns:
            spikes: (time_window, n_neurons) binary spike tensor

        Example:
            wernicke = WernickeArea()
            spikes = wernicke.perceive("Hello, Mergen.")

            # Feed to Layer 2
            for t in range(spikes.shape[0]):
                pre_spikes = spikes[t]
                post_spikes = mergen_engine.forward(pre_spikes)
                mergen_engine.update_traces(pre_spikes, post_spikes)
        """
        self._total_perceptions += 1
        self._last_input_text = text

        # Step 1: Text → Embedding (önbellekli tekil yol)
        embedding = self._embed_single_cached(text)

        # Step 2: Embedding → Spike Train
        if self.encoding == 'rate':
            spikes = self._rate_encode(embedding)
        elif self.encoding == 'temporal':
            spikes = self._temporal_encode(embedding)
        elif self.encoding == 'population':
            spikes = self._population_encode(embedding)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

        # Telemetry
        self._last_spike_count = int(spikes.sum().item())
        self._last_active_neurons = int((spikes.sum(dim=0) > 0).sum().item())

        return spikes

    # ════════════════════════════════════════════════════════════════
    #  SEMANTIC SIMILARITY — Helper for Layer 4 (Prefrontal)
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def semantic_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute cosine similarity between two texts in embedding space.

        Useful for Layer 4 (Prefrontal Cortex) to decide whether
        a new query is similar to something Mergen already knows.

        Returns: Similarity score in [-1, 1] (1 = identical meaning)
        """
        emb_a = self.embed(text_a)[0]
        emb_b = self.embed(text_b)[0]
        return torch.dot(emb_a, emb_b).item()

    # ════════════════════════════════════════════════════════════════
    #  TELEMETRY
    # ════════════════════════════════════════════════════════════════

    def get_telemetry(self) -> Dict:
        """Return perception statistics for monitoring."""
        total_lookups = self._cache_hits + self._cache_misses
        return {
            'total_perceptions': self._total_perceptions,
            'last_input': self._last_input_text[:50],
            'last_spike_count': self._last_spike_count,
            'last_active_neurons': self._last_active_neurons,
            'sparsity': 1.0 - (
                self._last_spike_count /
                max(1, self.time_window * self.n_neurons)
            ),
            'encoding': self.encoding,
            'embedding_dim': self.embedding_dim,
            'n_neurons': self.n_neurons,
            'time_window': self.time_window,
            # Önbellek istatistikleri
            'cache_size': len(self._embed_cache),
            'cache_capacity': self._embed_cache_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': round(
                self._cache_hits / total_lookups, 4
            ) if total_lookups > 0 else 0.0,
        }

    def clear_cache(self) -> None:
        """LRU embedding önbelleğini temizle (bellek baskısı veya model güncelleme sonrası)."""
        self._embed_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def __repr__(self):
        total_lookups = self._cache_hits + self._cache_misses
        hit_rate = (
            f"{self._cache_hits / total_lookups:.1%}"
            if total_lookups > 0 else "n/a"
        )
        return (
            f"WernickeArea(\n"
            f"  model: {self.embedding_model_name}\n"
            f"  encoding: {self.encoding}\n"
            f"  topology: {self.embedding_dim}D → {self.n_neurons} neurons "
            f"× {self.time_window} steps\n"
            f"  perceptions: {self._total_perceptions}\n"
            f"  embed_cache: {len(self._embed_cache)}/{self._embed_cache_size} "
            f"(hit_rate={hit_rate})\n"
            f")"
        )


# ════════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  MERGEN — Layer 1: Wernicke Area — Test Run")
    print("=" * 65)

    # Initialize with rate coding (default)
    wernicke = WernickeArea(
        embedding_model='sentence-transformers/all-MiniLM-L6-v2',
        n_neurons=384,
        time_window=50,
        encoding='rate',
        max_rate=0.4,
    )

    print(f"\n{wernicke}\n")

    # Test perceptions
    test_inputs = [
        "What is consciousness?",
        "Mergen is a digital brain.",
        "Hello world.",
    ]

    for text in test_inputs:
        spikes = wernicke.perceive(text)
        tele = wernicke.get_telemetry()
        print(f"  Input: '{text}'")
        print(f"    Spike shape: {tuple(spikes.shape)}")
        print(f"    Total spikes: {tele['last_spike_count']}")
        print(f"    Active neurons: {tele['last_active_neurons']}/384")
        print(f"    Sparsity: {tele['sparsity']:.3f}")
        print()

    # Test semantic similarity
    sim = wernicke.semantic_similarity(
        "I love programming",
        "Coding is my passion"
    )
    print(f"  Semantic similarity test: {sim:.4f} (should be high)")

    print("\n" + "=" * 65)
    print("  Wernicke Area ready to feed Mergen Engine (Layer 2)")
    print("=" * 65)
