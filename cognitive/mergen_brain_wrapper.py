"""
╔══════════════════════════════════════════════════════════════════════╗
║         MERGEN — ENHANCED BRAIN (Wernicke + MergenBrain)             ║
║                                                                      ║
║  "The brain that perceives, learns, and remembers."                 ║
║                                                                      ║
║  Author:  Vertex Corporation — Mergen Project                       ║
║  License: Apache-2.0                                                ║
╚══════════════════════════════════════════════════════════════════════╝

This module integrates:
  • Wernicke Area (text → semantic embedding → spike trains)
  • MergenBrain (neural processing + Hebbian learning)
  • Enhanced knowledge recall with semantic similarity
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any

from cognitive.mergen_brain import MergenBrain, MergenConfig


class EnhancedMergenBrain:
    """
    Enhanced brain that wraps MergenBrain and adds:
    • Wernicke Area for semantic perception
    • Semantic similarity-based recall
    • Text content hashing into neural processing
    • Spike train generation from text input
    """

    def __init__(
        self,
        brain: MergenBrain,
        config: Any = None,
        use_wernicke: bool = True,
        device: str = 'cpu',
    ):
        self.brain = brain
        self.config = config or MergenConfig()
        self.device = device
        self.use_wernicke = use_wernicke
        self.wernicke = None

        # Initialize Wernicke Area if available
        if use_wernicke:
            try:
                from cognitive.wernicke_area import WernickeArea
                self.wernicke = WernickeArea(
                    embedding_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                    n_neurons=768,
                    time_window=50,
                    encoding='rate',
                    device=device,
                )
                print("[EnhancedBrain] Wernicke Area loaded")
            except ImportError:
                print("[EnhancedBrain] Wernicke Area not available (sentence-transformers required)")
                self.use_wernicke = False

        # Projection layer: Wernicke 384 → Brain input 256
        self._wernicke_to_brain_proj = None

    def _project_wernicke_to_brain(self, wernicke_output: torch.Tensor) -> torch.Tensor:
        """Project Wernicke output to match brain input dimensions."""
        if self._wernicke_to_brain_proj is None:
            w_dim = wernicke_output.shape[0]
            b_dim = self.brain.input_dim
            if w_dim == b_dim:
                return wernicke_output
            # Create random projection (fixed seed for consistency)
            gen = torch.Generator(device=self.device).manual_seed(42)
            self._wernicke_to_brain_proj = torch.randn(
                b_dim, w_dim, device=self.device, generator=gen
            ) * 0.1

        if wernicke_output.dim() == 1:
            return self._wernicke_to_brain_proj @ wernicke_output
        return self._wernicke_to_brain_proj @ wernicke_output.flatten()

    def perceive(self, text: str) -> Dict:
        """
        Full perception pipeline:
        Text → Wernicke (spike train) → MergenBrain (neural activation)
        """
        result = {
            'spike_train': None,
            'neural_intent': None,
            'hidden_state': None,
        }

        # Step 1: Wernicke perception (text → spikes)
        if self.use_wernicke and self.wernicke:
            try:
                spike_train = self.wernicke.perceive(text)
                result['spike_train'] = spike_train
            except Exception as e:
                print(f"[EnhancedBrain] Wernicke error: {e}")
                self.use_wernicke = False

        # Step 2: MergenBrain processing
        intent_report = {
            'primary_intent': 'UNKNOWN',
            'confidence_score': 0.5,
            'sentiment': {'sentiment_score': 0.0, 'excitement': 0.0},
        }

        try:
            brain_output = self.brain.process(intent_report)
            result['neural_intent'] = brain_output.get('neural_intent')
            result['hidden_state'] = brain_output.get('hidden_state')
        except Exception as e:
            print(f"[EnhancedBrain] Brain processing error: {e}")

        return result

    def process_with_intent(
        self,
        text: str,
        intent_report: Dict,
    ) -> Dict:
        """
        Process text with intent information.

        Unlike the old version, this actually feeds text content
        into the brain, not just the sparse intent report.
        """
        result = {
            'spike_train': None,
            'neural_intent': None,
            'hidden_state': None,
        }

        # Wernicke perception
        wernicke_spike = None
        if self.use_wernicke and self.wernicke:
            try:
                wernicke_spike = self.wernicke.perceive(text)
                result['spike_train'] = wernicke_spike
            except Exception as e:
                print(f"[EnhancedBrain] Wernicke perception skipped: {e}")

        # Brain processing: use intent_report BUT also inject text content
        # by modifying the input encoding to include text hashes
        try:
            # Create an enriched intent report that includes text content
            enriched_report = dict(intent_report)

            # Inject text content hashes into the report so the brain
            # processes actual text meaning, not just metadata
            if text:
                enriched_report['_text_content'] = text

            brain_output = self.brain.process(enriched_report)
            neural_intent = brain_output.get('neural_intent')

            # If we have Wernicke spikes, blend them with neural intent
            # This gives the brain access to semantic embeddings
            if wernicke_spike is not None and neural_intent is not None:
                try:
                    # Project Wernicke output to vocab dimension and blend
                    projected = self._project_wernicke_to_brain(wernicke_spike)
                    # Project to vocab size
                    if projected.shape[0] != neural_intent.shape[0]:
                        # Simple pooling: take top-N and spread
                        n = neural_intent.shape[0]
                        if projected.shape[0] > n:
                            projected = projected[:n]
                        else:
                            pad = torch.zeros(n - projected.shape[0], device=self.device)
                            projected = torch.cat([projected, pad])

                    # Normalize both to [0,1] range
                    p_min, p_max = projected.min(), projected.max()
                    if p_max > p_min:
                        projected = (projected - p_min) / (p_max - p_min)

                    n_min, n_max = neural_intent.min(), neural_intent.max()
                    if n_max > n_min:
                        neural_normalized = (neural_intent - n_min) / (n_max - n_min)
                    else:
                        neural_normalized = neural_intent

                    # Blend: 30% Wernicke semantic + 70% brain activation
                    neural_intent = 0.3 * projected + 0.7 * neural_normalized

                except Exception as e:
                    print(f"[EnhancedBrain] Semantic blend skipped: {e}")

            result['neural_intent'] = neural_intent
            result['hidden_state'] = brain_output.get('hidden_state')
        except Exception as e:
            print(f"[EnhancedBrain] Error: {e}")

        return result

    def learn_from_text(self, text: str, vocab: Any, intent_report: Optional[Dict] = None,
                        learning_rate: float = 0.01, reward: float = 1.0) -> Dict:
        """Learn from text using the underlying MergenBrain."""
        return self.brain.learn_from_text(text, vocab, intent_report, learning_rate, reward)

    def recall(self, query: str, vocab: Any, top_k: int = 3) -> List[Dict]:
        """Recall from knowledge base using vocab-based matching."""
        return self.brain.recall(query, vocab, top_k)

    def recall_raw(self, query: str, top_k: int = 3) -> List[Dict]:
        """Recall using raw text token overlap (Turkish morphology aware)."""
        return self.brain.recall_raw(query, top_k)

    def recall_all_about(self, subject: str, top_k: int = 10) -> List[Dict]:
        """Recall all facts about a subject."""
        return self.brain.recall_all_about(subject, top_k)

    def recall_semantic(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Semantic similarity-based recall using Wernicke embeddings.
        Finds facts whose embeddings are closest to the query embedding.
        """
        if not self.use_wernicke or not self.wernicke or (not self.brain.semantic.knowledge_base and not self.brain.episodic.events):
            return self.recall_raw(query, top_k)

        try:
            # Get query embedding
            query_emb = self.wernicke.embed(query)[0]

            # Score each fact by semantic similarity
            scored_facts = []
            all_facts = self.brain.semantic.knowledge_base + self.brain.episodic.events
            for kb_idx, fact in enumerate(all_facts):
                fact_text = fact.get('text', '')
                if len(fact_text) < 10:
                    continue

                try:
                    fact_emb = self.wernicke.embed(fact_text)[0]
                    similarity = torch.dot(query_emb, fact_emb).item()

                    # Combine semantic similarity with fact weight
                    relevance = similarity * fact.get('weight', 1.0)

                    scored_facts.append({
                        'text': fact_text,
                        'relevance': relevance,
                        'semantic_similarity': similarity,
                        'kb_idx': kb_idx,
                    })
                except Exception as e:
                    print(f"[EnhancedBrain] Semantic scoring skipped for fact {kb_idx}: {e}")
                    continue

            # Sort by relevance
            scored_facts.sort(key=lambda x: -x['relevance'])

            # Update access counts
            for f in scored_facts[:top_k]:
                idx = f['kb_idx']
                if idx < len(self.brain.semantic.knowledge_base):
                    self.brain.semantic.knowledge_base[idx]['access_count'] = self.brain.semantic.knowledge_base[idx].get('access_count', 0) + 1
                else:
                    ep_idx = idx - len(self.brain.semantic.knowledge_base)
                    if ep_idx < len(self.brain.episodic.events):
                        self.brain.episodic.events[ep_idx]['access_count'] = self.brain.episodic.events[ep_idx].get('access_count', 0) + 1

            return scored_facts[:top_k]

        except Exception as e:
            print(f"[EnhancedBrain] Semantic recall error: {e}")
            return self.recall_raw(query, top_k)

    def knowledge_size(self) -> int:
        """Return number of facts in knowledge base."""
        return self.brain.knowledge_size()

    def save(self, path: str) -> bool:
        """Save brain weights and knowledge base."""
        return self.brain.save(path)

    def load(self, path: str) -> bool:
        """Load brain weights and knowledge base."""
        return self.brain.load(path)

    def reinforce(self, neural_intent: torch.Tensor, reward: float = 1.0,
                  learning_rate: float = 0.005):
        """Reinforce recent activation pattern."""
        self.brain.reinforce(neural_intent, reward, learning_rate)

    @property
    def hebbian_trace(self):
        return self.brain.hebbian_trace

    @property
    def knowledge_base(self):
        return self.brain.semantic.knowledge_base + self.brain.episodic.events

    @property
    def mx1(self):
        return self.brain.mx1

    @property
    def mx2(self):
        return self.brain.mx2
