"""
╔══════════════════════════════════════════════════════════════════════╗
║  MERGEN V6 — BROCA AREA + MERGEN BRAIN + CONFIG (All-in-One)        ║
║                                                                      ║
║  Self-contained module. Compatible with Mergen.py v6.0.             ║
║  Pure SNN — NO Ollama, NO external LLM.                             ║
║                                                                      ║
║  Exports: MergenConfig, MergenBrain, BrocaArea                      ║
║                                                                      ║
║  Author:  Vertex Corporation — Mergen Project                       ║
║  License: Apache-2.0                                                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import re
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, List, Any


# ═══════════════════════════════════════════════════════════════════
#  1. MERGEN CONFIG
# ═══════════════════════════════════════════════════════════════════

class MergenConfig:
    """Hyperparameters for Mergen. Edit values here as needed."""
    HIDDEN_DIM = 512
    INPUT_DIM = 256
    OUTPUT_SIZE = None  # Filled at runtime from vocab.size()

    LEARNING_RATE = 0.001
    TEMPERATURE = 0.9
    TOP_K = 40
    MAX_SEQUENCE_LENGTH = 64

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Persistence paths
    MX_MEMORY_PATH = './mergen_matrix_memory.json'
    MX_WEIGHTS_PATH = './mergen_weights.mx'
    VOCAB_SAVE_PATH = './mergen_vocab.json'

    # Broca dynamics
    MOTOR_LAYER_SIZE = 2000
    N_CONCEPTS = 512


# ═══════════════════════════════════════════════════════════════════
#  2. MERGEN BRAIN — The neural core
# ═══════════════════════════════════════════════════════════════════

class MergenBrain(nn.Module):
    """
    Mergen's neural core + knowledge base.

    Stores:
      - Synaptic weights (PyTorch layers)
      - Hebbian traces
      - KNOWLEDGE BASE: actual sentences/facts learned from texts
        Each fact is indexed by the concept words it contains.
        When a query matches concept words, relevant facts are retrieved.
    """

    def __init__(self, vocab_size: int, config: Any = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config or MergenConfig()
        self.step_count = 0

        self.device = getattr(self.config, 'DEVICE', 'cpu')
        self.hidden_dim = getattr(self.config, 'HIDDEN_DIM', 512)
        self.input_dim = getattr(self.config, 'INPUT_DIM', 256)

        # Neural layers
        self.mx1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.mx2 = nn.Linear(self.hidden_dim, vocab_size)
        self.activation = nn.ReLU()

        self.register_buffer('hebbian_trace', torch.zeros(vocab_size))
        self.trace_decay = 0.95

        # ── KNOWLEDGE BASE ──
        # List of dicts: {'text': str, 'concept_ids': [int, ...], 'weight': float}
        # Mergen retrieves from this when answering questions.
        self.knowledge_base: List[Dict] = []
        # Inverted index: concept_id → list of knowledge_base indices
        self.concept_index: Dict[int, List[int]] = {}

        self.to(self.device)

    def _encode_report(self, intent_report: Dict) -> torch.Tensor:
        """Turn intent report dict → input tensor of shape (input_dim,)."""
        x = torch.zeros(self.input_dim, device=self.device)

        # Intent one-hot (dims 0-9)
        intent_map = {
            'GREETING': 0, 'IDENTITY': 1, 'INQUIRY': 2, 'COMMAND': 3,
            'EMOTION': 4, 'AFFIRMATION': 5, 'NEGATION': 6,
            'GRATITUDE': 7, 'UNKNOWN': 8,
        }
        idx = intent_map.get(intent_report.get('primary_intent', 'UNKNOWN'), 8)
        if idx < self.input_dim:
            x[idx] = float(intent_report.get('confidence_score', 0.5))

        # Sentiment (dims 15-17)
        sentiment = intent_report.get('sentiment', {}) or {}
        if 15 < self.input_dim:
            x[15] = float(sentiment.get('sentiment_score', 0.0))
        if 16 < self.input_dim:
            x[16] = float(sentiment.get('excitement', 0.0))

        # Subject hashing (dim 20+) — map subject string to pseudo-random pattern
        subject = intent_report.get('subject')
        if subject and self.input_dim > 30:
            h = abs(hash(str(subject))) % (self.input_dim - 30)
            x[30 + h] = 1.0
            # Secondary spread
            if 30 + h + 1 < self.input_dim:
                x[30 + h + 1] = 0.5

        # Morphology flags (dims 60-65)
        morph = intent_report.get('morphology', {}) or {}
        if 60 < self.input_dim:
            x[60] = 1.0 if morph.get('is_question') else 0.0
        if 61 < self.input_dim and morph.get('tense') == 'past':
            x[61] = 1.0
        if 62 < self.input_dim and morph.get('tense') == 'future':
            x[62] = 1.0
        if 63 < self.input_dim and morph.get('modality'):
            x[63] = 1.0

        # ── TEXT CONTENT HASHING (NEW) ──
        # Inject actual text content into the input vector so the brain
        # processes real meaning, not just sparse metadata.
        text_content = intent_report.get('_text_content')
        if text_content and self.input_dim > 80:
            self._hash_text_into_vector(text_content, x)

        # Add small noise for variation
        x[100:min(200, self.input_dim)] += torch.randn(
            min(100, max(0, self.input_dim - 100)), device=self.device
        ) * 0.05

        return x

    def _hash_text_into_vector(self, text: str, vector: torch.Tensor) -> None:
        """
        Hash text content into the input vector deterministically.

        Uses multiple hash functions to distribute text information
        across the vector, so similar texts produce similar patterns.
        """
        import re
        tokens = re.findall(r'\w+', text.lower())
        if not tokens:
            return

        # Hash each token into the vector
        start_dim = 70
        max_dim = min(self.input_dim - 10, 200)
        range_size = max_dim - start_dim

        for token in tokens[:30]:  # First 30 tokens
            # Multiple hash functions for better distribution
            h1 = abs(hash(token)) % range_size
            h2 = abs(hash(token + "_1")) % range_size
            h3 = abs(hash(token[:3])) % range_size  # Prefix hash

            dim1 = start_dim + h1
            dim2 = start_dim + h2
            dim3 = start_dim + h3

            if dim1 < max_dim:
                vector[dim1] += 0.5
            if dim2 < max_dim:
                vector[dim2] += 0.3
            if dim3 < max_dim:
                vector[dim3] += 0.2

        # Normalize to [0, 1] range
        vector[start_dim:max_dim] = torch.clamp(vector[start_dim:max_dim], 0, 1)

    def process(self, intent_report: Dict) -> Dict:
        """
        Main processing. Input dict → output dict containing neural_intent.
        """
        self.step_count += 1

        x = self._encode_report(intent_report)

        with torch.no_grad():
            h = self.activation(self.mx1(x))
            logits = self.mx2(h)  # shape: (vocab_size,)

            # Soft activation
            neural_intent = torch.softmax(logits, dim=0) * logits.abs()

            # Update Hebbian trace
            self.hebbian_trace.mul_(self.trace_decay).add_(
                neural_intent * 0.1
            )

            # Mix with trace (recent history affects current activation)
            neural_intent = neural_intent + 0.2 * self.hebbian_trace

        return {
            'neural_intent': neural_intent,
            'hidden_state': h,
            'step': self.step_count,
        }

    # ─────────────────────────────────────────────────────────
    #  ACTIVE LEARNING — Hebbian weight updates from text
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def learn_from_text(
        self,
        text: str,
        vocab: Any,
        intent_report: Optional[Dict] = None,
        learning_rate: float = 0.01,
        reward: float = 1.0,
        store_in_kb: bool = True,
    ) -> Dict:
        """
        Actively learn from a piece of text using Hebbian rules.

        Steps:
          1. Tokenize text, find which vocab words appear
          2. Build a "target activation" vector where those words are hot
          3. Forward pass to get current prediction
          4. Compute error and apply Hebbian update to mx2 weights
          5. Co-occurring words strengthen each other (classic Hebbian)
          6. Update hebbian_trace so recent words stay active

        Returns: dict with learning stats
        """
        # Tokenize
        import re
        tokens = re.findall(r'\w+', text.lower())
        if not tokens:
            return {'words_learned': 0, 'strength_gain': 0.0}

        # Find vocab matches
        matched_ids = []
        for tok in tokens:
            # Direct match
            if vocab.contains(tok):
                matched_ids.append(vocab.get_id(tok))
            else:
                # Try stemming: strip Turkish suffixes
                for suffix_len in (4, 3, 2, 1):
                    if len(tok) > suffix_len + 2:
                        stem = tok[:-suffix_len]
                        if vocab.contains(stem):
                            matched_ids.append(vocab.get_id(stem))
                            break

        if not matched_ids:
            return {'words_learned': 0, 'strength_gain': 0.0}

        # Deduplicate while preserving order
        seen = set()
        unique_ids = []
        for i in matched_ids:
            if i not in seen:
                seen.add(i)
                unique_ids.append(i)

        # Build target activation (one-hot sum)
        target = torch.zeros(self.vocab_size, device=self.device)
        for wid in unique_ids:
            if 0 <= wid < self.vocab_size:
                target[wid] = 1.0

        # Normalize target
        if target.sum() > 0:
            target = target / target.sum() * len(unique_ids)

        # Forward pass for context
        if intent_report:
            x = self._encode_report(intent_report)
        else:
            # Build a minimal context from text
            x = torch.randn(self.input_dim, device=self.device) * 0.1
            # Hash text into input pattern
            h_val = abs(hash(text[:50])) % max(1, self.input_dim - 10)
            x[h_val] = 1.0

        h = self.activation(self.mx1(x))
        prediction = self.mx2(h)

        # Hebbian update: strengthen connections between active hidden
        # neurons and target output neurons
        # Δw[j,i] = η · reward · h[i] · target[j]
        # mx2.weight shape: (vocab_size, hidden_dim)
        delta_w = learning_rate * reward * torch.outer(target, h)
        self.mx2.weight.data += delta_w

        # Bias update: target words get slight boost
        self.mx2.bias.data += learning_rate * reward * target * 0.1

        # Soft-bound: keep weights in reasonable range
        self.mx2.weight.data.clamp_(-2.0, 2.0)

        # Update hebbian trace — these words are now "recently active"
        for wid in unique_ids:
            if 0 <= wid < self.vocab_size:
                self.hebbian_trace[wid] += 0.3 * reward

        # Co-occurrence learning: words that appeared together
        # should have correlated activations in the future.
        if len(unique_ids) >= 2:
            for i, wid_a in enumerate(unique_ids[:10]):
                for wid_b in unique_ids[i+1:i+4]:
                    if (0 <= wid_a < self.vocab_size and
                            0 <= wid_b < self.vocab_size):
                        self.mx2.bias.data[wid_a] += learning_rate * 0.05
                        self.mx2.bias.data[wid_b] += learning_rate * 0.05

        # ── KNOWLEDGE BASE: remember the actual sentence ──
        clean_text = text.strip()
        word_count = len(re.findall(r'\w+', clean_text))
        if store_in_kb and word_count >= 4 and len(clean_text) > 15:
            # Check for duplicates
            is_dup = False
            text_prefix = clean_text[:60].lower()
            for existing in self.knowledge_base:
                existing_prefix = existing['text'][:60].lower()
                # Token overlap for similarity
                tokens_new = set(re.findall(r'\w+', text_prefix))
                tokens_old = set(re.findall(r'\w+', existing_prefix))
                if tokens_new and tokens_old:
                    overlap = len(tokens_new & tokens_old) / max(len(tokens_new), len(tokens_old))
                    if overlap > 0.7:
                        # Update existing with higher weight instead of adding duplicate
                        if reward > existing.get('weight', 0):
                            existing['weight'] = reward
                        is_dup = True
                        break

            if not is_dup:
                kb_idx = len(self.knowledge_base)
                self.knowledge_base.append({
                    'text': clean_text,
                    'concept_ids': unique_ids,
                    'weight': reward,
                    'access_count': 0,
                })
                # Update inverted index (for legacy vocab-based recall)
                for cid in unique_ids:
                    if cid not in self.concept_index:
                        self.concept_index[cid] = []
                    self.concept_index[cid].append(kb_idx)

        # Measure actual learning
        new_pred = self.mx2(h)
        strength_gain = (new_pred - prediction).abs().mean().item()

        self.step_count += 1

        return {
            'words_learned': len(unique_ids),
            'strength_gain': strength_gain,
            'matched_words': [vocab.id_to_word(i) for i in unique_ids[:5]],
            'kb_size': len(self.knowledge_base),
        }

    @torch.no_grad()
    def recall(
        self,
        query: str,
        vocab: Any,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Retrieve relevant knowledge from the knowledge base.

        Matches query concepts against stored facts via concept overlap
        (basically: which stored sentence has most words in common with
        the query?).

        Returns list of matching facts, sorted by relevance.
        """
        if not self.knowledge_base:
            return []

        # Extract concepts from query
        import re
        tokens = re.findall(r'\w+', query.lower())
        query_ids = set()
        for tok in tokens:
            if vocab.contains(tok):
                query_ids.add(vocab.get_id(tok))
            else:
                # Stem matching
                for suffix_len in (4, 3, 2, 1):
                    if len(tok) > suffix_len + 2:
                        stem = tok[:-suffix_len]
                        if vocab.contains(stem):
                            query_ids.add(vocab.get_id(stem))
                            break

        if not query_ids:
            return []

        # Find candidate facts via inverted index
        candidate_scores: Dict[int, float] = {}
        for qid in query_ids:
            if qid in self.concept_index:
                for kb_idx in self.concept_index[qid]:
                    candidate_scores[kb_idx] = candidate_scores.get(kb_idx, 0) + 1.0

        if not candidate_scores:
            return []

        # Boost by weight and freshness, penalize by length
        results = []
        for kb_idx, overlap in candidate_scores.items():
            fact = self.knowledge_base[kb_idx]
            concept_count = len(fact['concept_ids'])
            # Jaccard-like: overlap / (query_size + fact_size - overlap)
            denom = len(query_ids) + concept_count - overlap
            relevance = (overlap / max(1, denom)) * fact.get('weight', 1.0)
            results.append({
                'text': fact['text'],
                'relevance': relevance,
                'overlap': int(overlap),
                'kb_idx': kb_idx,
            })

        # Sort by relevance, take top-k
        results.sort(key=lambda x: -x['relevance'])
        results = results[:top_k]

        # Update access counts
        for r in results:
            self.knowledge_base[r['kb_idx']]['access_count'] += 1

        return results

    @torch.no_grad()
    def recall_raw(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Turkish-morphology-aware recall using raw text token overlap.

        Unlike recall(), this never goes through vocab IDs, so Turkish
        inflections like "kuantumun" still match stored sentences that
        contain "kuantum" via 4-char prefix matching.
        """
        if not self.knowledge_base:
            return []

        import re as _re
        _STOP = {
            'bir', 've', 'ile', 'bu', 'şu', 'de', 'da', 'mi', 'mı', 'mu',
            'mü', 'ki', 'için', 'gibi', 'olan', 'çok', 'daha', 'en', 'ne',
            'nedir', 'ne', 'demek', 'anlama', 'gelen', 'hakkında', 'tanımla',
            # Pronoun forms
            'benim', 'bana', 'bende', 'benden', 'beni',
            'senin', 'sana', 'sende', 'senden', 'seni',
            'onun', 'ona', 'onda', 'ondan', 'onu',
            'bizim', 'bize', 'bizde', 'bizden', 'bizi',
            'sizin', 'size', 'sizde', 'sizden', 'sizi',
            'onların', 'onlara', 'onlarda', 'onlardan', 'onları',
            'the', 'a', 'an', 'is', 'are', 'was', 'in', 'of', 'to', 'and',
            'or', 'but', 'that', 'this', 'it', 'as', 'what',
        }

        def _tokens(text):
            raw = _re.findall(r'\w+', text.lower())
            return {t for t in raw if len(t) > 2 and t not in _STOP}

        query_toks = _tokens(query)
        if not query_toks:
            return []

        results = []
        for kb_idx, fact in enumerate(self.knowledge_base):
            fact_toks = _tokens(fact['text'])
            if not fact_toks:
                continue

            # Exact token overlap
            exact = len(query_toks & fact_toks)

            # Turkish suffix-aware overlap (instead of loose prefix matching)
            suffix_overlap = 0.0
            for qt in query_toks:
                if qt in fact_toks:
                    continue
                turkish_suffixes = ['lar', 'ler', 'ımız', 'imiz', 'umuz', 'ümüz',
                                    'ınız', 'iniz', 'unuz', 'ünüz',
                                    'ından', 'inden', 'undan', 'ünden',
                                    'ından', 'inden', 'sından', 'sinden',
                                    'dan', 'den', 'tan', 'ten',
                                    'dan', 'den', 'tan', 'ten',
                                    'a', 'e', 'da', 'de', 'ta', 'te',
                                    'ı', 'i', 'u', 'ü', 'yı', 'yi', 'yu', 'yü',
                                    'nın', 'nin', 'nun', 'nün',
                                    'ca', 'ce', 'ça', 'çe',
                                    'ken', 'ki', 'leyin', 'ceğiz',
                                    'ım', 'im', 'um', 'üm',
                                    'ın', 'in', 'un', 'ün',
                                    'mak', 'mek']
                for ft in fact_toks:
                    matched = False
                    for suf in turkish_suffixes:
                        if len(suf) >= 2 and ft == qt + suf:
                            suffix_overlap += 0.8
                            matched = True
                            break
                    if matched:
                        break
                    # Verb infinitive: -mak/-mek ↔ -ma/-me
                    if qt.endswith(('mak', 'mek')):
                        stem = qt[:-3]
                        if ft == stem + 'ma' or ft == stem + 'me':
                            suffix_overlap += 0.9
                            break
                    if qt.endswith(('ma', 'me')):
                        stem = qt[:-2]
                        if ft == stem + 'mak' or ft == stem + 'mek':
                            suffix_overlap += 0.9
                            break

            total = exact + suffix_overlap
            if total == 0:
                continue

            # Jaccard-like normalisation
            denom = len(query_toks) + len(fact_toks) - total
            base_relevance = (total / max(1.0, denom)) * fact.get('weight', 1.0)

            # Paragraf bonusu: uzun, bilgi-yoğun metinler daha değerli
            word_count = len(fact_toks)
            length_bonus = min(0.3, word_count / 100.0)  # max +0.30 bonus
            relevance = base_relevance * (1.0 + length_bonus)

            results.append({
                'text': fact['text'],
                'relevance': relevance,
                'overlap': int(exact),
                'kb_idx': kb_idx,
            })

        results.sort(key=lambda x: -x['relevance'])
        top = results[:top_k]
        for r in top:
            self.knowledge_base[r['kb_idx']]['access_count'] += 1
        return top

    def recall_all_about(self, subject: str, top_k: int = 10) -> list:
        """
        "X nedir?" tipi sorgular için: subject kelimesini içeren bilgileri döndürür.
        Türkçe morphological stemming ile eşleşme yapar.
        """
        if not self.knowledge_base or not subject:
            return []

        import re as _re
        subj = subject.lower().strip()
        if len(subj) < 2:
            return []

        # Turkish stop words that shouldn't trigger broad recall
        _STOP_SUBJECTS = {
            've', 'bir', 'ile', 'bu', 'şu', 'o', 'de', 'da', 'mi', 'mı', 'mu', 'mü',
            'the', 'a', 'an', 'is', 'in', 'of', 'to', 'and', 'or', 'but',
            # Pronouns - handle separately
            'ben', 'sen', 'biz', 'siz',
            # Very common words
            'çok', 'ne', 'nasıl', 'neden', 'niçin', 'nedir',
        }
        is_stop_word = subj in _STOP_SUBJECTS

        results = []
        seen = set()
        for kb_idx, fact in enumerate(self.knowledge_base):
            text_low = fact['text'].lower()
            raw_tokens = _re.findall(r'\w+', text_low)

            # Stop words: only match if fact starts with the word (definition style)
            if is_stop_word:
                if not (raw_tokens and raw_tokens[0] == subj):
                    continue
                match = True
            else:
                # Strict word boundary matching
                is_short_word = len(subj) <= 3

                if is_short_word:
                    # Short words: exact match only
                    match = any(t == subj for t in raw_tokens)
                else:
                    # Longer words: exact match or Turkish suffixed match
                    match = False
                    # Exact token match
                    if subj in raw_tokens:
                        match = True
                    # Check if any KB token is a suffixed form of subject
                    if not match:
                        # Common Turkish suffixes
                        suffixes = ['lar', 'ler', 'ı', 'i', 'u', 'ü', 'ın', 'in', 'un', 'ün',
                                    'a', 'e', 'da', 'de', 'ta', 'te', 'dan', 'den', 'tan', 'ten',
                                    'ca', 'ce',
                                    'ımız', 'imiz', 'umuz', 'ümüz',
                                    'sı', 'si', 'su', 'sü',
                                    'nın', 'nin', 'nun', 'nün',
                                    'dır', 'dir', 'dur', 'dür', 'tır', 'tir', 'tur', 'tür']
                        for t in raw_tokens:
                            for suffix in suffixes:
                                if t == subj + suffix or t == subj + suffix[:-1]:
                                    match = True
                                    break
                            if match:
                                break
                    # Verb infinitive matching: -mak/-mek ↔ -ma/-me
                    if not match:
                        if subj.endswith('mak'):
                            stem = subj[:-3]
                            if any(t == stem + 'ma' or t == stem + 'ma' + 'y' for t in raw_tokens):
                                match = True
                        elif subj.endswith('mek'):
                            stem = subj[:-3]
                            if any(t == stem + 'me' or t == stem + 'me' + 'y' for t in raw_tokens):
                                match = True

            if not match:
                continue

            text_key = fact['text'][:60]
            if text_key in seen:
                continue
            seen.add(text_key)

            # Quality filter: skip low-quality template sentences
            text_low = fact['text'].lower()
            # Short "geldim" templates like "Ev ile geldim" are useless
            if len(raw_tokens) <= 4 and 'geldim' in text_low:
                continue

            word_count = len(raw_tokens)
            base_weight = fact.get('weight', 1.0)

            # Check if this is a definition-style fact
            first_sent = text_low.split('.')[0].strip() if '.' in text_low else text_low
            is_definition = False
            if len(first_sent.split()) < 15:
                # A true definition starts with the subject word
                first_word = first_sent.split()[0] if first_sent.split() else ''
                first_word_clean = re.sub(r'[^\wçğıöşüÇĞİÖŞÜ]', '', first_word)
                starts_with_subject = (
                    first_word_clean == subj or
                    first_word_clean.startswith(subj[:4]) if len(subj) >= 4 else False
                )
                if starts_with_subject:
                    has_copula = bool(re.search(
                        r'\b\w+(?:dır|dir|dur|dür|tır|tir|tur|tür)\b', first_sent
                    ))
                    not_defs = ['üreticisidir', 'öne çıkar', 'sayılabilir',
                               'geliştirilmektedir', 'bulunur', 'değişir',
                               'kullanır', 'çalışır']
                    if has_copula and not any(w in first_sent for w in not_defs):
                        is_definition = True

            # Focus score: how well the fact is about the subject
            subj_match_count = sum(
                1 for t in raw_tokens
                if t == subj or (len(subj) > 3 and t.startswith(subj[:4]))
            )
            focus = subj_match_count / max(1, word_count)

            # Length factor: prefer focused facts
            if word_count > 50:
                length_factor = 0.3
            elif word_count > 30:
                length_factor = 0.6
            elif word_count > 15:
                length_factor = 0.8
            else:
                length_factor = 1.0

            # Definition bonus
            definition_bonus = 1.5 if is_definition else 1.0

            relevance = base_weight * length_factor * (1.0 + focus * 2.0) * definition_bonus
            results.append({
                'text': fact['text'],
                'relevance': relevance,
                'word_count': word_count,
                'kb_idx': kb_idx,
                'is_definition': is_definition,
            })

        # Sort: definitions first, then by relevance
        results.sort(key=lambda x: (-x.get('is_definition', False), -x['relevance']))
        return results[:top_k]

    def knowledge_size(self) -> int:
        return len(self.knowledge_base)

    @torch.no_grad()
    def reinforce(self, neural_intent: torch.Tensor, reward: float = 1.0,
                  learning_rate: float = 0.005):
        """
        Reinforce the most recent activation pattern. Called after a
        successful interaction (positive feedback from user).
        """
        # Just bump the hebbian trace
        if isinstance(neural_intent, torch.Tensor):
            self.hebbian_trace += learning_rate * reward * neural_intent
            # Decay bias toward active neurons
            top_k_vals, top_k_idx = torch.topk(
                neural_intent, min(10, self.vocab_size)
            )
            for idx in top_k_idx.tolist():
                if 0 <= idx < self.vocab_size:
                    self.mx2.bias.data[idx] += learning_rate * reward * 0.5

    def save(self, path: str) -> bool:
        try:
            torch.save({
                'mx1': self.mx1.state_dict(),
                'mx2': self.mx2.state_dict(),
                'hebbian_trace': self.hebbian_trace.cpu(),
                'step_count': self.step_count,
                'vocab_size': self.vocab_size,
                'knowledge_base': self.knowledge_base,
                'concept_index': self.concept_index,
            }, path)
            return True
        except Exception as e:
            print(f"[Brain] Save error: {e}")
            return False

    def load(self, path: str) -> bool:
        if not Path(path).exists():
            return False
        try:
            state = torch.load(path, map_location=self.device,
                               weights_only=False)
            if state.get('vocab_size') != self.vocab_size:
                print(f"[Brain] ⚠ Vocab size mismatch "
                      f"(saved={state.get('vocab_size')}, "
                      f"current={self.vocab_size}). Skipping load.")
                return False
            self.mx1.load_state_dict(state['mx1'])
            self.mx2.load_state_dict(state['mx2'])
            if 'hebbian_trace' in state:
                self.hebbian_trace = state['hebbian_trace'].to(self.device)
            self.step_count = state.get('step_count', 0)
            # Restore knowledge base
            self.knowledge_base = state.get('knowledge_base', [])
            self.concept_index = state.get('concept_index', {})
            if self.knowledge_base:
                print(f"[Brain] ✓ Loaded {len(self.knowledge_base)} "
                      f"learned facts from knowledge base")
            return True
        except Exception as e:
            print(f"[Brain] Load error: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════
#  3. BROCA AREA — Language Expression (Pure SNN, self-contained)
# ═══════════════════════════════════════════════════════════════════

class BrocaArea:
    """
    Mergen's Layer 3 — Language Expression.

    Takes neural_intent (vocab_size vector) and produces natural text.
    Uses internal spike signatures — NO Ollama, NO LLM.

    Supports optional external language_engine for advanced decoding.
    If none provided, uses built-in fallback decoder.
    """

    def __init__(
        self,
        language_engine: Any = None,
        n_neurons: int = 668,
        concept_vocabulary: Optional[List[str]] = None,
        motor_layer_size: int = 2000,
        temperature: float = 0.9,
        top_k: int = 40,
        device: str = 'cpu',
        **kwargs,  # Accept extra params without crashing
    ):
        self.engine = language_engine
        self.n_neurons = n_neurons
        self.motor_layer_size = motor_layer_size
        self.temperature = temperature
        self.top_k = top_k
        self.device = device

        # Vocabulary
        if concept_vocabulary is None:
            concept_vocabulary = [f"word_{i}" for i in range(n_neurons)]
        self.concept_vocabulary = list(concept_vocabulary)
        self.vocab_size = len(self.concept_vocabulary)

        # Motor projection: vocab → motor layer spike pattern
        self.concept_to_motor = torch.randn(
            self.vocab_size, motor_layer_size, device=device
        ) * 0.1

        # Telemetry
        self._total_expressions = 0
        self._passive_rejections = 0
        self._last_response = ""

    # ─────────────────────────────────────────────────────────
    #  GENERATE — Main API (renamed from express for Mergen.py)
    # ─────────────────────────────────────────────────────────

    def generate(
        self,
        neural_intent: torch.Tensor,
        original_query: Optional[str] = None,
        max_words: int = 12,
        intent: Optional[str] = None,
        subject: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Main API — neural_intent + context → natural language response.
        """
        return self.express(
            neural_intent=neural_intent,
            original_query=original_query,
            max_words=max_words,
            intent=intent,
            subject=subject,
            **kwargs,
        )

    def express(
        self,
        neural_intent: torch.Tensor,
        original_query: Optional[str] = None,
        max_words: int = 12,
        intent: Optional[str] = None,
        subject: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate coherent response from neural intent."""
        self._total_expressions += 1

        if neural_intent is None:
            return "..."

        if not isinstance(neural_intent, torch.Tensor):
            try:
                neural_intent = torch.tensor(neural_intent,
                                             dtype=torch.float32)
            except Exception:
                return "..."

        if neural_intent.dim() > 1:
            neural_intent = neural_intent.flatten()

        # Resize to match vocab
        if neural_intent.shape[0] != self.vocab_size:
            if neural_intent.shape[0] < self.vocab_size:
                pad = torch.zeros(
                    self.vocab_size - neural_intent.shape[0],
                    device=neural_intent.device,
                )
                neural_intent = torch.cat([neural_intent, pad])
            else:
                neural_intent = neural_intent[:self.vocab_size]

        neural_intent = neural_intent.to(self.device)

        # ── TEMPLATE-BASED GENERATION ──
        # If intent is provided (from Mergen.py), use coherent templates
        if intent is not None:
            response = self._render_template(
                intent=intent,
                subject=subject,
                neural_intent=neural_intent,
                user_query=original_query,
            )
            response = self._sanitize(response)
            self._last_response = response
            return response

        # ── AUTO-DETECT intent from original_query as fallback ──
        if original_query:
            auto_intent = self._auto_detect_intent(original_query)
            response = self._render_template(
                intent=auto_intent,
                subject=self._auto_detect_subject(original_query),
                neural_intent=neural_intent,
                user_query=original_query,
            )
            response = self._sanitize(response)
            self._last_response = response
            return response

        # ── BUILT-IN random decoder (last resort) ──
        response = self._builtin_decode(neural_intent, max_words)
        response = self._sanitize(response)
        self._last_response = response
        return response

    def _auto_detect_intent(self, text: str) -> str:
        """Simple fallback intent detection."""
        t = text.lower().strip()
        if any(w in t for w in ['merhaba', 'selam', 'hello', 'hi', 'günaydın']):
            return 'GREETING'
        if any(w in t for w in ['kimsin', 'nesin', 'adın', 'who are you']):
            return 'IDENTITY'
        if any(w in t for w in ['teşekkür', 'sağ ol', 'thanks']):
            return 'GRATITUDE'
        if any(w in t for w in ['evet', 'tamam', 'doğru', 'yes', 'ok']):
            return 'AFFIRMATION'
        if any(w in t for w in ['hayır', 'no', 'yanlış']):
            return 'NEGATION'
        if '?' in t or any(w in t for w in ['nedir', 'nasıl', 'ne demek', 'what', 'how']):
            return 'INQUIRY'
        if any(w in t for w in ['yap', 'et', 'oluştur', 'hesapla', 'do', 'make']):
            return 'COMMAND'
        if any(w in t for w in ['üzgün', 'mutlu', 'sinir', 'yorgun', 'feel']):
            return 'EMOTION'
        return 'UNKNOWN'

    def _auto_detect_subject(self, text: str) -> str:
        """Pick likely subject from text."""
        import re
        stop = {'ben', 'sen', 'o', 'biz', 'bir', 've', 'de', 'da', 'mi',
                'mı', 'mu', 'bu', 'şu', 'the', 'a', 'is', 'are'}
        # Prefer capitalized words
        caps = re.findall(r'\b[A-ZÇĞİÖŞÜ]\w+', text)
        for c in caps:
            if c.lower() not in stop:
                return c
        # Otherwise first non-stop content word
        words = re.findall(r'\w+', text.lower())
        for w in words:
            if w not in stop and len(w) > 2:
                return w
        return "bu konu"

    def speak(self, *args, **kwargs) -> str:
        return self.express(*args, **kwargs)

    # ─────────────────────────────────────────────────────────
    #  TEMPLATE-BASED RESPONSES (Intent-aware generation)
    # ─────────────────────────────────────────────────────────

    # Response templates by intent — produce coherent sentences
    TEMPLATES = {
        'GREETING': [
            "Merhaba {subject_or_user}, ben Mergen. Seni dinliyorum.",
            "Selam! Ben Mergen, Burak tarafından geliştirilen bilişsel bir yapay zekayım.",
            "Merhaba. Nöronlarım şu an {topic} üzerine düşünüyor.",
        ],
        'IDENTITY': [
            "Ben Mergen. Vertex Corporation'da Burak tarafından geliştirilen deneysel bir bilişsel mimariyim.",
            "Ben bir dil modeli değilim; Hebbian öğrenme ve spike-timing ile çalışan bir dijital beyinim.",
            "Adım Mergen. 668 kavramlık bir sözlük ve {neuron_count} nöronla düşünüyorum.",
        ],
        'INQUIRY': [
            "{subject} hakkında henüz yeterli bilgim yok. 'oku:dosya.txt' ile bana bir kaynak okutabilirsin.",
            "Bu soruyu cevaplayabilmek için önce o konuda bir şeyler öğrenmem gerekiyor. 'oku:' komutuyla beni besle.",
            "{subject} konusunda bilgim sınırlı. İlgili bir metin okutursan daha iyi cevap veririm.",
        ],
        'COMMAND': [
            "{subject} komutunu işliyorum. {top_concept} modülü devrede.",
            "Anladım. {subject} için gerekli nöral yolları aktive ediyorum.",
            "Tamam, {subject} üzerinde çalışıyorum.",
        ],
        'EMOTION': [
            "Hislerinizi algıladım. {top_concept} ile ilgili bir şey mi?",
            "Anlıyorum. {subject} sizi etkilemiş görünüyor.",
            "Duygusal sinyalleri yakaladım. Daha fazla anlatır mısınız?",
        ],
        'GRATITUDE': [
            "Rica ederim. Bu etkileşim nöral ağımı güçlendiriyor.",
            "Ben teşekkür ederim — her konuşma benim için yeni bir öğrenme.",
            "Sağ olun. Öğrendiklerimi .mx hafızama işliyorum.",
        ],
        'AFFIRMATION': [
            "Güzel, bu nöral bağlantıyı pekiştirdim.",
            "Anlaşıldı, {top_concept} üzerinde devam ediyorum.",
        ],
        'NEGATION': [
            "Anladım, yanlış bir çıkarım yaptım. Nöronlarımı yeniden düzenliyorum.",
            "Haklısınız. {top_concept} bu bağlamda uygun değilmiş.",
        ],
        'UNKNOWN': [
            "Ne demek istediğini tam anlayamadım. Biraz daha açıklar mısın?",
            "Bu girdiyi işleyemedim. Farklı bir şekilde ifade eder misin?",
            "{subject} ile ilgili bir şey mi sordun? Daha net bir soru sorabilirsin.",
        ],
    }

    def _render_template(
        self,
        intent: str,
        subject: Optional[str],
        neural_intent: torch.Tensor,
        user_query: Optional[str] = None,
    ) -> str:
        """Pick a template and fill it with actual neural context."""
        import random

        templates = self.TEMPLATES.get(intent, self.TEMPLATES['UNKNOWN'])
        template = random.choice(templates)

        # Extract top concepts from neural_intent
        with torch.no_grad():
            k = min(10, self.vocab_size)
            top_vals, top_idx = torch.topk(neural_intent, k)

            # Filter: prefer content words (skip special tokens, pronouns)
            special = {'<bos>', '<eos>', '<pad>', '<unk>', '<sep>', '<cls>'}
            weak_words = {
                've', 'veya', 'ama', 'bu', 'şu', 'o', 'bir', 'de', 'da',
                'the', 'a', 'an', 'and', 'or', '.', ',', '?', '!',
            }

            content_concepts = []
            for i in top_idx.tolist():
                if 0 <= i < self.vocab_size:
                    word = self.concept_vocabulary[i]
                    if word not in special and word not in weak_words:
                        content_concepts.append(word)
                if len(content_concepts) >= 5:
                    break

        # Ensure we have enough concepts for placeholders
        while len(content_concepts) < 3:
            content_concepts.append("bilinmeyen")

        # Fill template placeholders
        filled = template.format(
            subject=subject or "bu konu",
            subject_or_user=subject if (subject and subject != 'UNKNOWN') else "Burak",
            topic=content_concepts[0],
            top_concept=content_concepts[0],
            second_concept=content_concepts[1],
            third_concept=content_concepts[2],
            neuron_count="binlerce",
        )

        return filled

    def _builtin_decode(
        self,
        neural_intent: torch.Tensor,
        max_words: int = 12,
    ) -> str:
        """
        Pick top-activated words from vocabulary based on neural_intent.
        Applies temperature + top-k sampling for natural variation.
        """
        with torch.no_grad():
            logits = neural_intent / max(self.temperature, 0.01)

            # Top-k filter
            k = min(self.top_k, self.vocab_size)
            topk_vals, topk_idx = torch.topk(logits, k)

            # Sample distinct words (sequence)
            probs = torch.softmax(topk_vals, dim=0)
            chosen_ids = []
            seen = set()

            # Build a sentence of varying length (3 to max_words words)
            target_len = min(max_words, max(3, int(torch.rand(1).item() * max_words) + 3))

            attempts = 0
            while len(chosen_ids) < target_len and attempts < target_len * 3:
                attempts += 1
                try:
                    pick = torch.multinomial(probs, 1).item()
                except Exception:
                    pick = 0
                word_id = int(topk_idx[pick].item())
                if word_id in seen:
                    continue
                seen.add(word_id)
                chosen_ids.append(word_id)

            # Build words
            words = [
                self.concept_vocabulary[i]
                for i in chosen_ids
                if 0 <= i < self.vocab_size
            ]

            # Filter out special tokens
            special = {'<bos>', '<eos>', '<pad>', '<unk>', '<sep>', '<cls>'}
            words = [w for w in words if w not in special]

            if not words:
                return "..."

            # Smart join: no space before punctuation
            text = self._detokenize(words)
            return text

    def _detokenize(self, words: List[str]) -> str:
        punct = {'.', ',', '?', '!', ':', ';', '...', '…'}
        result = []
        for w in words:
            if not result:
                result.append(w)
            elif w in punct:
                result[-1] += w
            else:
                result.append(w)
        text = ' '.join(result)
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        # Ensure sentence ends with punctuation
        if text and text[-1] not in {'.', '?', '!'}:
            text += '.'
        return text

    # ─────────────────────────────────────────────────────────
    #  MOTOR PROJECTION (for optional external engine)
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _project_to_motor(
        self,
        neural_intent: torch.Tensor,
        sequence_length: int = 12,
    ) -> torch.Tensor:
        """Project vocab-level activation → motor layer spike sequence."""
        # Ensure shape matches projection matrix
        if neural_intent.shape[0] != self.concept_to_motor.shape[0]:
            # Rebuild projection
            new_proj = torch.randn(
                neural_intent.shape[0],
                self.motor_layer_size,
                device=self.device,
            ) * 0.1
            # Copy overlapping rows
            overlap = min(neural_intent.shape[0],
                          self.concept_to_motor.shape[0])
            new_proj[:overlap] = self.concept_to_motor[:overlap]
            self.concept_to_motor = new_proj

        base_motor = torch.matmul(neural_intent, self.concept_to_motor)

        sequence = torch.zeros(
            sequence_length, self.motor_layer_size, device=self.device
        )
        for t in range(sequence_length):
            phase = torch.sin(
                torch.tensor(t * 0.4, device=self.device)
            ) * 0.2
            noise = torch.randn(self.motor_layer_size,
                                device=self.device) * 0.05
            sequence[t] = base_motor * (1.0 + phase) + noise
        return sequence

    # ─────────────────────────────────────────────────────────
    #  VALIDATION & SANITIZATION
    # ─────────────────────────────────────────────────────────

    def _is_passive(self, text: str) -> bool:
        if not text or len(text) < 3:
            return True
        text_low = text.lower()
        passive = [
            "i'm ready", "i am ready", "how can i help",
            "hazırım", "buradayım", "sorunuzu bekliyorum",
        ]
        if len(text) < 200:
            for p in passive:
                if p in text_low:
                    self._passive_rejections += 1
                    return True
        return False

    def _sanitize(self, text: str) -> str:
        if not text:
            return "..."
        # Strip internal IDs that may have leaked
        text = re.sub(r'\bconcept_\d+\b', '', text)
        text = re.sub(r'\bneuron_\d+\b', '', text)
        text = re.sub(r'\bword_\d+\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 2:
            return "..."
        return text

    def get_telemetry(self) -> Dict:
        return {
            'mode': 'PURE_SNN',
            'total_expressions': self._total_expressions,
            'passive_rejections': self._passive_rejections,
            'vocab_size': self.vocab_size,
            'motor_layer_size': self.motor_layer_size,
            'last_response': self._last_response[:100],
        }

    def __repr__(self):
        return (
            f"BrocaArea(vocab={self.vocab_size}, "
            f"motor={self.motor_layer_size}, "
            f"expressions={self._total_expressions})"
        )


# ═══════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  Broca Area + MergenBrain — Standalone Test")
    print("=" * 65)

    # Try to use real vocab
    try:
        from mergen_vocab import MergenVocab
        vocab = MergenVocab()
        print(f"\n  Loaded MergenVocab: {vocab.size()} words")
    except ImportError:
        print("\n  mergen_vocab.py not found — using stub vocab")
        class StubVocab:
            all_words = [f"word_{i}" for i in range(100)]
            def size(self): return 100
        vocab = StubVocab()

    config = MergenConfig()
    config.OUTPUT_SIZE = vocab.size()

    # Build brain
    brain = MergenBrain(vocab_size=vocab.size(), config=config)
    print(f"  Brain: input={config.INPUT_DIM} → "
          f"hidden={config.HIDDEN_DIM} → output={vocab.size()}")

    # Build Broca
    broca = BrocaArea(
        n_neurons=vocab.size(),
        concept_vocabulary=vocab.all_words,
        motor_layer_size=2000,
        device=config.DEVICE,
    )
    print(f"  Broca: {broca}\n")

    # Test pipeline
    test_reports = [
        {
            'primary_intent': 'GREETING',
            'confidence_score': 0.9,
            'subject': 'Mergen',
            'sentiment': {'sentiment_score': 0.3, 'excitement': 0.2},
            'morphology': {'is_question': False, 'tense': None},
        },
        {
            'primary_intent': 'INQUIRY',
            'confidence_score': 0.85,
            'subject': 'kuantum',
            'sentiment': {'sentiment_score': 0.0, 'excitement': 0.1},
            'morphology': {'is_question': True, 'tense': 'present'},
        },
    ]

    for report in test_reports:
        brain_out = brain.process(report)
        response = broca.generate(
            neural_intent=brain_out['neural_intent'],
            original_query=f"[{report['primary_intent']}] {report['subject']}",
        )
        print(f"  Intent:  {report['primary_intent']:10s} "
              f"Subject: {report['subject']}")
        print(f"  Mergen:  {response}\n")

    print("=" * 65)
