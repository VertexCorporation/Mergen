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
from typing import Optional, Dict, List, Any
from core.turkish_grammar import TurkishGrammar


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
        self.vocab = kwargs.get('vocab', None)
        self.temperature = temperature
        self.top_k = top_k
        self.device = device

        # Vocabulary
        if concept_vocabulary is None:
            concept_vocabulary = [f"word_{i}" for i in range(n_neurons)]
        self.concept_vocabulary = list(concept_vocabulary)
        self.vocab_size = len(self.concept_vocabulary)
        # BUG-04 FIX: O(n) list.index() yerine O(1) ters indeks sozlugu.
        # _neural_detect_intent() icindeki top3 dongusunde kullanilir.
        self._concept_to_idx = {w: i for i, w in enumerate(self.concept_vocabulary)}

        if self.engine:
            self.motor_layer_size = self.engine.motor_layer_size
        else:
            self.motor_layer_size = motor_layer_size

        # SNN Motor Projection Layer (Restored)
        self.motor_projection = nn.Linear(self.vocab_size, self.motor_layer_size).to(self.device)

        self.grammar = TurkishGrammar()

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
            except Exception as e:
                print(f"[Broca] Neural intent conversion failed: {e}")
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

        activation_strength = kwargs.get('activation_strength', 1.0)

        # ── TEMPLATE-BASED GENERATION ──
        # If intent is provided (from Mergen.py), use coherent templates
        if intent is not None:
            response = self._render_template(
                intent=intent,
                subject=subject,
                neural_intent=neural_intent,
                user_query=original_query,
                activation_strength=activation_strength,
            )
            response = self._sanitize(response)
            self._last_response = response
            return response

        # ── NEURAL INTENT DETECTION ──
        # Detect intent directly from the neural activation map
        auto_intent = self._neural_detect_intent(neural_intent)

        # ── NEURAL SOV GENERATION ──
        # Build sentences dynamically based on grammar instead of templates for some intents
        if auto_intent in ['DISCUSSION', 'COMMAND', 'INQUIRY']:
            sov_response = self._generate_sov_sentence(auto_intent, neural_intent, subject, original_query)
            if sov_response:
                sov_response = self._sanitize(sov_response)
                self._last_response = sov_response
                return sov_response

        response = self._render_template(
            intent=auto_intent,
            subject=self._auto_detect_subject(original_query) if original_query else "",
            neural_intent=neural_intent,
            user_query=original_query,
            activation_strength=activation_strength,
        )
        response = self._sanitize(response)

        # BUG-01 FIX: Sablon basarisiz olursa (bos/"..." yanit) builtin decoder devreye girer.
        if not response or response == "...":
            response = self._builtin_decode(neural_intent, max_words)
            response = self._sanitize(response)

        self._last_response = response
        return response

    def _neural_detect_intent(self, neural_intent: torch.Tensor) -> str:
        """Detect intent based on neural spike activations (concept categories)."""
        if self.vocab is None or not hasattr(self.vocab, 'category_ranges'):
            return 'UNKNOWN'

        # Calculate average activation per category
        cat_activations = {}
        for cat, (start, end) in self.vocab.category_ranges.items():
            if end > start:
                cat_activations[cat] = neural_intent[start:end].max().item()

        if not cat_activations:
            return 'UNKNOWN'

        # Sort categories by activation strength
        sorted_cats = sorted(cat_activations.items(), key=lambda x: x[1], reverse=True)
        top_cat_name = sorted_cats[0][0]
        top_cat_val = sorted_cats[0][1]

        # Check top specific concept
        top_concept_idx = torch.argmax(neural_intent).item()
        top_concept = self.concept_vocabulary[top_concept_idx] if neural_intent[top_concept_idx] > 0.0 else ""

        # Skip structural tokens
        if top_concept_idx < 21:  # NUM_MASKED_TOKENS
            return 'UNKNOWN'

        # Social signal keywords: highest priority
        if top_concept in ['merhaba', 'selam', 'günaydın', 'hello', 'hi']:
            return 'GREETING'
        if top_concept in ['teşekkür', 'sağol', 'teşekkürler']:
            return 'GRATITUDE'
        if top_concept in ['evet', 'tamam', 'doğru', 'kesinlikle']:
            return 'AFFIRMATION'
        if top_concept in ['hayır', 'yanlış', 'değil']:
            return 'NEGATION'
        if top_concept in ['kim', 'ne', 'nasıl', 'neden', 'niçin', 'nerede']:
            return 'INQUIRY'

        # Category-level mapping (relative: top category > 2x next best)
        if len(sorted_cats) >= 2:
            second_val = sorted_cats[1][1]
            is_dominant = (top_cat_val > 0.001 and 
                          (second_val < 1e-9 or top_cat_val / (second_val + 1e-9) > 1.2))
        else:
            is_dominant = top_cat_val > 0.001

        if is_dominant:
            if top_cat_name in ['physics', 'software', 'philosophy', 'natural_science', 'social_science']:
                return 'DISCUSSION'
            if top_cat_name == 'turkish_common':
                return 'DISCUSSION'
            if top_cat_name in ['verbs', 'extended_verbs']:
                return 'COMMAND'
            if top_cat_name in ['daily', 'adjectives']:
                return 'GREETING'
            if top_cat_name == 'connectives':
                return 'INQUIRY'

        # If any physics/science concept is in top 3 concepts
        top3_indices = torch.topk(neural_intent, min(3, neural_intent.shape[0])).indices.tolist()
        top3_concepts = [self.concept_vocabulary[i] for i in top3_indices]

        science_cats = {'physics', 'software', 'philosophy', 'natural_science', 'social_science'}
        for c in top3_concepts:
            for sci_cat in science_cats:
                if sci_cat in self.vocab.category_ranges:
                    start, end = self.vocab.category_ranges[sci_cat]
                    # BUG-04 FIX: list.index() -> O(1) ters sozluk erisimi
                    idx = self._concept_to_idx.get(c, -1)
                    if start <= idx < end:
                        return 'DISCUSSION'

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
        'DISCUSSION': [
            "İlginç bir konu. {top_concept} kavramı aklıma geldi. Sen ne düşünüyorsun?",
            "{top_concept} hakkında konuşmak zihnimi açıyor. Devam edelim.",
            "Evet, {subject} bağlamında {top_concept} önemli bir faktör.",
            "Nöronlarım {top_concept} üzerine ateşlendi. Konuyu derinleştirelim."
        ]
    }

    def _generate_sov_sentence(self, intent: str, neural_intent: torch.Tensor, subject: Optional[str], original_query: Optional[str]) -> Optional[str]:
        """Generates dynamic SOV (Subject-Object-Verb) sentence using TurkishGrammar."""
        # Get top concepts to form the sentence
        top_k_indices = torch.topk(neural_intent, 5).indices.tolist()
        concepts = [self.concept_vocabulary[idx] for idx in top_k_indices if neural_intent[idx] > 0.05]
        
        if len(concepts) < 2:
            return None # Not enough concepts to form a rich sentence, fallback to templates
            
        subj = subject if subject else (concepts[0].capitalize() if intent != 'COMMAND' else "Ben")
        obj = concepts[1] if len(concepts) > 1 else ""
        verb = ""
        
        # Try to find a verb in top concepts
        for c in concepts:
            if c.endswith('mek') or c.endswith('mak'):
                verb = c
                break
                
        if not verb:
            if intent == 'DISCUSSION': verb = "düşünmek"
            elif intent == 'COMMAND': verb = "yapmak"
            elif intent == 'INQUIRY': verb = "bilmek"
            else: verb = "olmak"
            
        negative = False
        if any(w in concepts for w in ['hayır', 'değil', 'yok', 'olumsuz']):
            negative = True
            
        person = 1 # Ben (Default for Mergen)
        tense = 'present'
        if intent == 'COMMAND':
            person = 1 # "Ben yapıyorum"
        elif intent == 'INQUIRY':
            person = 1 # "Ben biliyorum/bilmiyorum"
            if not obj: obj = "bunu"
            
        # Decide object case loosely based on verb
        obj_case = 'accusative'
        if verb in ['gitmek', 'bakmak', 'çalışmak']: obj_case = 'dative'
        elif verb in ['korkmak', 'kaçmak', 'gelmek']: obj_case = 'ablative'
        elif verb in ['olmak', 'bulunmak', 'düşünmek']: obj_case = 'locative'
        
        target_sentence = self.grammar.build_sov(subject=subj, obj=obj, verb=verb, obj_case=obj_case, tense=tense, person=person, negative=negative)
        if not target_sentence:
            return None

        # ── SNN BIOLOGICAL TEACHER MODE ──
        # Instead of returning the string directly (GOFAI), we use the grammar string
        # as a "Supervisory Signal" (Innate Universal Grammar) to train the LanguageEngine
        # via Hebbian STDP, and then generate the actual text via SNN motor spikes.
        if self.engine is not None:
            # Tokenize into words
            words = [w.strip() for w in re.findall(r"[\w']+|[.,!?;]", target_sentence) if w.strip()]
            T = len(words)
            if T > 0:
                with torch.no_grad():
                    try:
                        # Generate base motor projection from intent
                        base_motor = self.motor_projection(neural_intent)
                        device = getattr(self.engine, 'device', 'cpu')
                        base_motor_engine = base_motor.to(device)
                        motor_sequence = torch.zeros(T, self.motor_layer_size, device=device)
                        
                        for t in range(T):
                            # Create a unique temporal motor spike pattern for this step
                            temporal_noise = torch.randn_like(base_motor_engine) * 0.1
                            motor_sequence[t] = base_motor_engine * (1.0 - t * 0.05) + temporal_noise
                            
                            # Online STDP Training: Train LanguageEngine on the fly
                            self.engine.strengthen_association(words[t], motor_sequence[t])
                    
                        # ACTUAL GENERATION: SNN LanguageEngine decodes the motor sequence
                        return self.engine.speak(motor_sequence)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        return f"[SNN Error: {e}]"

        return target_sentence

    def _render_template(
        self,
        intent: str,
        subject: Optional[str],
        neural_intent: torch.Tensor,
        user_query: Optional[str] = None,
        activation_strength: float = 1.0,
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
                except Exception as e:
                    print(f"[Broca] Sampling fallback used: {e}")
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
    from cognitive.mergen_brain import MergenConfig, MergenBrain
    try:
        from core.mergen_vocab import MergenVocab
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
