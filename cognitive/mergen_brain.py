# mergen_brain.py
import re
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, List, Any

# Yeni bellek modülleri
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory

class MergenConfig:
    INPUT_DIM: int = 768
    HIDDEN_DIM: int = 256
    OUTPUT_SIZE: int = 689
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MX_WEIGHTS_PATH: str = "./mergen_weights.mx"
    VOCAB_SAVE_PATH: str = "./mergen_vocab.json"
    MX_MEMORY_PATH: str = "./mergen_matrix_memory.json"
    MX_KNOWLEDGE_PATH: str = "./mergen_knowledge.mx" # Isolated path for knowledge base

class MergenBrain(nn.Module):
    """
    Mergen's neural core + knowledge base.
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

        # ── KNOWLEDGE BASE (Faz 4 Ayrışımı) ──
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.concept_index: Dict[int, List[int]] = {}

        self.to(self.device)

    def _encode_report(self, intent_report: Dict) -> torch.Tensor:
        x = torch.zeros(self.input_dim, device=self.device)
        intent_map = {
            'GREETING': 0, 'IDENTITY': 1, 'INQUIRY': 2, 'COMMAND': 3,
            'EMOTION': 4, 'AFFIRMATION': 5, 'NEGATION': 6,
            'GRATITUDE': 7, 'UNKNOWN': 8,
        }
        idx = intent_map.get(intent_report.get('primary_intent', 'UNKNOWN'), 8)
        if idx < self.input_dim:
            x[idx] = float(intent_report.get('confidence_score', 0.5))

        sentiment = intent_report.get('sentiment', {}) or {}
        if 15 < self.input_dim:
            x[15] = float(sentiment.get('sentiment_score', 0.0))
        if 16 < self.input_dim:
            x[16] = float(sentiment.get('excitement', 0.0))

        subject = intent_report.get('subject')
        if subject and self.input_dim > 30:
            h = abs(hash(str(subject))) % (self.input_dim - 30)
            x[30 + h] = 1.0
            if 30 + h + 1 < self.input_dim:
                x[30 + h + 1] = 0.5

        morph = intent_report.get('morphology', {}) or {}
        if 60 < self.input_dim:
            x[60] = 1.0 if morph.get('is_question') else 0.0
        if 61 < self.input_dim and morph.get('tense') == 'past':
            x[61] = 1.0
        if 62 < self.input_dim and morph.get('tense') == 'future':
            x[62] = 1.0
        if 63 < self.input_dim and morph.get('modality'):
            x[63] = 1.0

        text_content = intent_report.get('_text_content')
        if text_content and self.input_dim > 80:
            self._hash_text_into_vector(text_content, x)

        return x

    def _hash_text_into_vector(self, text: str, vector: torch.Tensor) -> None:
        tokens = re.findall(r'\w+', text.lower())
        if not tokens:
            return

        start_dim = 70
        max_dim = min(self.input_dim - 10, 200)
        range_size = max_dim - start_dim

        for token in tokens[:30]:
            h1 = abs(hash(token)) % range_size
            h2 = abs(hash(token + "_1")) % range_size
            h3 = abs(hash(token[:3])) % range_size

            dim1 = start_dim + h1
            dim2 = start_dim + h2
            dim3 = start_dim + h3

            if dim1 < max_dim:
                vector[dim1] += 0.5
            if dim2 < max_dim:
                vector[dim2] += 0.3
            if dim3 < max_dim:
                vector[dim3] += 0.2

        vector[start_dim:max_dim] = torch.clamp(vector[start_dim:max_dim], 0, 1)

    # ── TD-13 FIX: Türkçe karakter normalizasyonu — tek yetkili yer ──
    # Önceden recall_raw(), recall_all_about() ve learn_from_text() içinde
    # aynı tablo 3 kez kopyalanmıştı. DRY prensibine aykırıydı.
    _FOLD_TABLE = str.maketrans({
        'ç': 'c', 'ğ': 'g', 'ı': 'i', 'İ': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
        'Ç': 'c', 'Ğ': 'g', 'I': 'i', 'Ö': 'o', 'Ş': 's', 'Ü': 'u',
    })

    @staticmethod
    def _fold_text(text: str) -> str:
        """Türkçe özgün karakterleri ASCII karşılıklarına dönüştürür ve küçük harf yapar.
        Sınıf düzeyinde tanımlı tek normalizasyon noktası (TD-13 DRY fix).
        """
        return (text or '').translate(MergenBrain._FOLD_TABLE).lower()

    def process(self, intent_report: Dict) -> Dict:
        self.step_count += 1
        x = self._encode_report(intent_report)
        with torch.no_grad():
            h = self.activation(self.mx1(x))
            logits = self.mx2(h)
            neural_intent = torch.softmax(logits, dim=0) * logits.abs()
            self.hebbian_trace.mul_(self.trace_decay).add_(neural_intent * 0.1)
            neural_intent = neural_intent + 0.2 * self.hebbian_trace

        return {
            'neural_intent': neural_intent,
            'hidden_state': h,
            'step': self.step_count,
        }

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
        tokens = re.findall(r'\w+', text.lower())
        if not tokens:
            return {'words_learned': 0, 'strength_gain': 0.0}

        matched_ids = []
        for tok in tokens:
            if vocab.contains(tok):
                matched_ids.append(vocab.get_id(tok))
            else:
                for suffix_len in (4, 3, 2, 1):
                    if len(tok) > suffix_len + 2:
                        stem = tok[:-suffix_len]
                        if vocab.contains(stem):
                            matched_ids.append(vocab.get_id(stem))
                            break

        if not matched_ids:
            return {'words_learned': 0, 'strength_gain': 0.0}

        seen = set()
        unique_ids = []
        for i in matched_ids:
            if i not in seen:
                seen.add(i)
                unique_ids.append(i)

        target = torch.zeros(self.vocab_size, device=self.device)
        for wid in unique_ids:
            if 0 <= wid < self.vocab_size:
                target[wid] = 1.0

        if target.sum() > 0:
            target = target / target.sum() * len(unique_ids)

        if intent_report:
            x = self._encode_report(intent_report)
        else:
            x = torch.randn(self.input_dim, device=self.device) * 0.1
            h_val = abs(hash(text[:50])) % max(1, self.input_dim - 10)
            x[h_val] = 1.0

        h = self.activation(self.mx1(x))
        prediction = self.mx2(h)

        # mx2.weight is shape [vocab_size, hidden_dim] = [out, in]
        # outer(target, h) produces [vocab_size, hidden_dim] which matches weight shape
        delta_w = learning_rate * reward * torch.outer(target, h)
        self.mx2.weight.data += delta_w
        self.mx2.bias.data += learning_rate * reward * target * 0.1
        self.mx2.weight.data.clamp_(-2.0, 2.0)


        for wid in unique_ids:
            if 0 <= wid < self.vocab_size:
                self.hebbian_trace[wid] += 0.3 * reward

        if len(unique_ids) >= 2:
            for i, wid_a in enumerate(unique_ids[:10]):
                for wid_b in unique_ids[i+1:i+4]:
                    if (0 <= wid_a < self.vocab_size and
                            0 <= wid_b < self.vocab_size):
                        self.mx2.bias.data[wid_a] += learning_rate * 0.05
                        self.mx2.bias.data[wid_b] += learning_rate * 0.05

        clean_text = text.strip()
        word_count = len(re.findall(r'\w+', clean_text))
        if store_in_kb and word_count >= 4 and len(clean_text) > 15:
            # BUG-02 FIX: Ekleme öncesinde birleşik listenin toplam boyutunu kaydet.
            # Bu değer, yeni girişin recall() tarafından kullanılan
            # (semantic.knowledge_base + episodic.events) birleşik listedeki
            # mutlak (0-tabanlı) indeksidir.
            all_facts_before = len(self.semantic) + len(self.episodic)

            # episodic vs semantic ayrımı
            is_episodic = (intent_report is not None)
            if is_episodic:
                # add_event() içindeki deduplication yoktur; dışarıdan kontrol gerekiyorsa
                # burada yapılabilir. Şimdilik doğrudan ekle.
                self.episodic.add_event(clean_text, unique_ids, reward)
            else:
                # add_fact() kendi içinde tam-eşleşme deduplication yapar.
                # Eğer duplicate bulunursa mevcut öğeyi güncelleyip o öğenin
                # indeksini döndürür — all_facts_before zaten o noktayı göstermez,
                # ancak concept_index için yeni bir giriş eklenmemesi sağlanır.
                result_idx = self.semantic.add_fact(clean_text, unique_ids, reward)
                # Semantic add_fact duplicate bulduysa len değişmedi, o zaman
                # kb_idx'i birleşik listede gerçek konuma çevir.
                all_facts_before = result_idx  # semantic'teki 0-tabanlı indeks = birleşik listede de aynı konum

            kb_idx = all_facts_before

            for cid in unique_ids:
                if cid not in self.concept_index:
                    self.concept_index[cid] = []
                # Aynı kb_idx'i tekrar eklememek için kontrol
                if kb_idx not in self.concept_index[cid]:
                    self.concept_index[cid].append(kb_idx)

        new_pred = self.mx2(h)
        strength_gain = (new_pred - prediction).abs().mean().item()
        self.step_count += 1

        return {
            'words_learned': len(unique_ids),
            'strength_gain': strength_gain,
            'matched_words': [vocab.id_to_word(i) for i in unique_ids[:5]],
            'kb_size': len(self.semantic) + len(self.episodic),
        }

    @torch.no_grad()
    def recall(
        self,
        query: str,
        vocab: Any,
        top_k: int = 3,
    ) -> List[Dict]:
        if not self.semantic.knowledge_base and not self.episodic.events:
            return []

        tokens = re.findall(r'\w+', query.lower())
        query_ids = set()
        for tok in tokens:
            if vocab.contains(tok):
                query_ids.add(vocab.get_id(tok))
            else:
                for suffix_len in (4, 3, 2, 1):
                    if len(tok) > suffix_len + 2:
                        stem = tok[:-suffix_len]
                        if vocab.contains(stem):
                            query_ids.add(vocab.get_id(stem))
                            break

        if not query_ids:
            return []

        candidate_scores: Dict[int, float] = {}
        for qid in query_ids:
            if qid in self.concept_index:
                for kb_idx in self.concept_index[qid]:
                    candidate_scores[kb_idx] = candidate_scores.get(kb_idx, 0) + 1.0

        if not candidate_scores:
            return []

        results = []
        # İki bellekte birden arama
        all_facts = self.semantic.knowledge_base + self.episodic.events
        for kb_idx, overlap in candidate_scores.items():
            if kb_idx < len(all_facts):
                fact = all_facts[kb_idx]
                concept_count = len(fact.get('concept_ids', []))
                denom = len(query_ids) + concept_count - overlap
                relevance = (overlap / max(1, denom)) * fact.get('weight', 1.0)
                results.append({
                    'text': fact.get('text', ''),
                    'relevance': relevance,
                    'overlap': int(overlap),
                    'kb_idx': kb_idx,
                })

        results.sort(key=lambda x: -x['relevance'])
        results = results[:top_k]

        for r in results:
            idx = r['kb_idx']
            if idx < len(self.semantic.knowledge_base):
                self.semantic.knowledge_base[idx]['access_count'] += 1
            else:
                ep_idx = idx - len(self.semantic.knowledge_base)
                if ep_idx < len(self.episodic.events):
                    self.episodic.events[ep_idx]['access_count'] = self.episodic.events[ep_idx].get('access_count', 0) + 1

        return results

    @torch.no_grad()
    def recall_raw(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.semantic.knowledge_base and not self.episodic.events:
            return []

        _STOP = {
            'bir', 've', 'ile', 'bu', 'şu', 'de', 'da', 'mi', 'mı', 'mu',
            'mü', 'ki', 'için', 'gibi', 'olan', 'çok', 'daha', 'en', 'ne',
            'nedir', 'ne', 'demek', 'anlama', 'gelen', 'hakkında', 'tanımla',
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
            # TD-13 FIX: yerel _fold() kaldirildi, sinif metodu kullaniliyor
            raw = re.findall(r'\w+', self._fold_text(text))
            res = set()
            for t in raw:
                if t.isdigit() or t in {'on', 'uc', 'uc'}:
                    res.add(t)
                elif len(t) > 2 and t not in _STOP:
                    res.add(t)
            return res

        query_toks = _tokens(query)
        if not query_toks:
            return []

        results = []
        for kb_idx, fact in enumerate(self.semantic.knowledge_base + self.episodic.events):
            fact_toks = _tokens(fact['text'])
            if not fact_toks:
                continue

            exact = len(query_toks & fact_toks)
            suffix_overlap = 0.0
            for qt in query_toks:
                if qt in fact_toks:
                    continue
                # Suffix listesi: _fold_text() ASCII'ye donusturdugundan
                # Turkce karakterler zaten normalize; dogrudan ASCII yaziyoruz.
                turkish_suffixes = [
                    'lar', 'ler', 'imiz', 'imiz', 'umuz', 'umuz',
                    'iniz', 'iniz', 'unuz', 'unuz',
                    'indan', 'inden', 'undan', 'unden',
                    'indan', 'inden', 'sindan', 'sinden',
                    'dan', 'den', 'tan', 'ten',
                    'a', 'e', 'da', 'de', 'ta', 'te',
                    'i', 'i', 'u', 'u', 'yi', 'yi', 'yu', 'yu',
                    'nin', 'nin', 'nun', 'nun',
                    'ca', 'ce', 'ca', 'ce',
                    'ken', 'ki', 'leyin', 'cegiz',
                    'im', 'im', 'um', 'um',
                    'in', 'in', 'un', 'un',
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

            denom = len(query_toks) + len(fact_toks) - total
            base_relevance = (total / max(1.0, denom)) * fact.get('weight', 1.0)

            word_count = len(fact_toks)
            length_bonus = min(0.3, word_count / 100.0)
            relevance = base_relevance * (1.0 + length_bonus)

            results.append({
                'text': fact['text'],
                'relevance': relevance,
                'overlap': int(exact),  # Fix variable to match exact overlap count
                'kb_idx': kb_idx,
            })

        results.sort(key=lambda x: -x['relevance'])
        results = results[:top_k]

        for r in results:
            idx = r['kb_idx']
            if idx < len(self.semantic.knowledge_base):
                self.semantic.knowledge_base[idx]['access_count'] = self.semantic.knowledge_base[idx].get('access_count', 0) + 1
            else:
                ep_idx = idx - len(self.semantic.knowledge_base)
                if ep_idx < len(self.episodic.events):
                    self.episodic.events[ep_idx]['access_count'] = self.episodic.events[ep_idx].get('access_count', 0) + 1

        return results

    @torch.no_grad()
    def recall_all_about(self, subject: str, top_k: int = 5) -> List[Dict]:
        if not self.semantic.knowledge_base and not self.episodic.events:
            return []

        # TD-13 FIX: yerel _fold() kaldirildi, sinif metodu kullaniliyor
        subj = self._fold_text(subject.strip())
        results = []
        seen = set()

        for kb_idx, fact in enumerate(self.semantic.knowledge_base + self.episodic.events):
            raw_tokens = [t.lower() for t in re.findall(r'\w+', self._fold_text(fact['text']))]
            if not raw_tokens:
                continue

            match = False
            if len(subj) <= 3:
                if subj in raw_tokens:
                    match = True
            else:
                if subj in raw_tokens:
                    match = True
                if not match:
                    # Suffix listesi ASCII; _fold_text() token'lari zaten normalize etti.
                    suffixes = [
                        'lar', 'ler', 'i', 'i', 'u', 'u', 'in', 'in', 'un', 'un',
                        'a', 'e', 'da', 'de', 'ta', 'te', 'dan', 'den', 'tan', 'ten',
                        'ca', 'ce',
                        'imiz', 'imiz', 'umuz', 'umuz',
                        'si', 'si', 'su', 'su',
                        'nin', 'nin', 'nun', 'nun',
                        'dir', 'dir', 'dur', 'dur', 'tir', 'tir', 'tur', 'tur']
                    for t in raw_tokens:
                        for suffix in suffixes:
                            if t == subj + suffix or t == subj + suffix[:-1]:
                                match = True
                                break
                        if match:
                            break
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

            text_low = fact['text'].lower()
            if len(raw_tokens) <= 4 and 'geldim' in text_low:
                continue

            word_count = len(raw_tokens)
            base_weight = fact.get('weight', 1.0)

            first_sent = text_low.split('.')[0].strip() if '.' in text_low else text_low
            is_definition = False
            if len(first_sent.split()) < 15:
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

            subj_match_count = sum(
                1 for t in raw_tokens
                if t == subj or (len(subj) > 3 and t.startswith(subj[:4]))
            )
            focus = subj_match_count / max(1, word_count)

            if word_count > 50:
                length_factor = 0.3
            elif word_count > 30:
                length_factor = 0.6
            elif word_count > 15:
                length_factor = 0.8
            else:
                length_factor = 1.0

            definition_bonus = 1.5 if is_definition else 1.0

            relevance = base_weight * length_factor * (1.0 + focus * 2.0) * definition_bonus
            results.append({
                'text': fact['text'],
                'relevance': relevance,
                'word_count': word_count,
                'kb_idx': kb_idx,
                'is_definition': is_definition,
            })

        results.sort(key=lambda x: (-x.get('is_definition', False), -x['relevance']))
        return results[:top_k]

    def knowledge_size(self) -> int:
        return len(self.semantic) + len(self.episodic)

    @torch.no_grad()
    def reinforce(self, neural_intent: torch.Tensor, reward: float = 1.0,
                  learning_rate: float = 0.005):
        if isinstance(neural_intent, torch.Tensor):
            intent = neural_intent.to(self.device)
            self.hebbian_trace.add_(learning_rate * reward * intent)
            self.hebbian_trace.clamp_(-5.0, 5.0)
            top_k_vals, top_k_idx = torch.topk(
                intent, min(10, self.vocab_size)
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
                # Yeni yapı
                'knowledge_base': {
                    'semantic': self.semantic.to_list(),
                    'episodic': self.episodic.to_list()
                },
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
                print(f"[Brain] WARNING: Vocab size mismatch "
                      f"(saved={state.get('vocab_size')}, "
                      f"current={self.vocab_size}). Skipping load.")
                return False
            self.mx1.load_state_dict(state['mx1'])
            self.mx2.load_state_dict(state['mx2'])
            if 'hebbian_trace' in state:
                self.hebbian_trace = state['hebbian_trace'].to(self.device)
            self.step_count = state.get('step_count', 0)
            # Migration / Fallback Mantığı
            kb_data = state.get('knowledge_base', [])
            if isinstance(kb_data, list):
                # Eski liste formatı, tamamını Semantik yap
                for item in kb_data:
                    if 'memory_type' not in item:
                        item['memory_type'] = 'semantic'
                self.semantic.from_list(kb_data)
                self.episodic.clear()
            elif isinstance(kb_data, dict):
                # Yeni format
                self.semantic.from_list(kb_data.get('semantic', []))
                self.episodic.from_list(kb_data.get('episodic', []))
                
            self.concept_index = state.get('concept_index', {})
            
            total_facts = len(self.semantic) + len(self.episodic)
            if total_facts > 0:
                print(f"[Brain] OK: Loaded {total_facts} "
                      f"learned facts from knowledge base (Semantic: {len(self.semantic)}, Episodic: {len(self.episodic)})")
            return True
        except Exception as e:
            print(f"[Brain] Load error: {e}")
            return False
