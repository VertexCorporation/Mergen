"""
╔══════════════════════════════════════════════════════════════════════╗
║          MERGEN — DIGITAL BRAIN v7.0 (Full Integration)              ║
║                                                                      ║
║  "A thinking digital brain — not a chatbot."                        ║
║                                                                      ║
║  ARCHITECTURE:                                                       ║
║    Wernicke (perception) → Intent Analysis → Brain Processing       ║
║    → Knowledge Recall → Response Synthesis → Broca (expression)     ║
║                                                                      ║
║  Author:  Vertex Corporation — Mergen Project                       ║
║  License: Apache-2.0                                                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys
import json
import time
import signal
import random
import threading
import re
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Set
from collections import Counter

# ── Core modules ──
try:
    from core.mergen_vocab import MergenVocab
except ImportError as e:
    print(f"[Mergen] ✗ Cannot import MergenVocab: {e}")
    sys.exit(1)

try:
    from cognitive.intent_analyzer import IntentAnalyzer
except ImportError as e:
    print(f"[Mergen] ✗ Cannot import IntentAnalyzer: {e}")
    sys.exit(1)

try:
    from cognitive.broca_area import BrocaArea
except ImportError as e:
    print(f"[Mergen] ✗ Cannot import broca_area: {e}")
    sys.exit(1)

from memory.conversation_memory import ConversationMemory
from learning.hebbian_engine import HybridHebbianLearner
from learning.cortical_column import CorticalColumn
from cognitive.limbic_executive_layer import LimbicExecutiveLayer
from cognitive.response_generator import ResponseGenerator

try:
    from cognitive.mergen_brain import MergenBrain
    from cognitive.mergen_brain_wrapper import EnhancedMergenBrain
except ImportError as e:
    print(f"[Mergen] ✗ Cannot import mergen_brain / wrapper: {e}")
    sys.exit(1)


# ── RAG + Biyolojik bileşenler (opsiyonel) ──
try:
    from rag_engine import RAGEngine
    from data_loader import TurkishDataLoader
    from hebbian_rag_bridge import HebbianRAGBridge
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False

try:
    from core.turkish_morph import TurkishMorph
    _MORPH_AVAILABLE = True
except ImportError:
    _MORPH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
#  MERGEN — Full Digital Brain Orchestrator
# ═══════════════════════════════════════════════════════════════════

class MergenConfig:
    INPUT_DIM: int = 768
    HIDDEN_DIM: int = 256
    OUTPUT_SIZE: int = 1136  # Updated: matches current MergenVocab size
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MX_WEIGHTS_PATH: str = "./mergen_weights.mx"
    VOCAB_SAVE_PATH: str = "./mergen_vocab.json"
    MX_MEMORY_PATH: str = "./mergen_matrix_memory.json"
    MX_KNOWLEDGE_PATH: str = "./mergen_knowledge.mx"

class MergenBrain_v7:
    """
    Integrated Digital Brain v7.0

    The conductor that binds:
    • MergenVocab        → Concept vocabulary
    • IntentAnalyzer     → Semantic intent classification
    • EnhancedMergenBrain → Neural core + Wernicke perception
    • ConversationMemory → Multi-turn context
    • ResponseSynthesizer → Spike-based response generation
    • BrocaArea          → Language expression
    """

    VERSION = "7.0"

    def __init__(
        self,
        config: Any = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self._start_time = time.time()

        # 1. Config
        self.config = config or MergenConfig()

        # 2. Vocabulary
        vocab_path = getattr(self.config, 'VOCAB_SAVE_PATH', './mergen_vocab.json')
        if Path(vocab_path).exists():
            if verbose:
                print(f"[Mergen] Loading vocabulary from {vocab_path}")
            self.vocab = MergenVocab.load(vocab_path)
        else:
            if verbose:
                print(f"[Mergen] Building fresh vocabulary")
            self.vocab = MergenVocab()

        # Set output size to match vocab
        self.config.OUTPUT_SIZE = self.vocab.size()
        if verbose:
            print(f"[Mergen] Vocabulary: {self.vocab.size()} concepts")

        # 3. Wernicke Area (Perception)
        self.wernicke = None
        try:
            from cognitive.wernicke_area import WernickeArea
            self.wernicke = WernickeArea(
                embedding_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                n_neurons=768,
                time_window=50,
                encoding='rate',
                device=getattr(self.config, 'DEVICE', 'cpu'),
                embed_cache_size=256,   # LRU önbellek — tekrarlı girdilerde encoder.encode() atlanır
            )
            if verbose:
                print("[Mergen] Wernicke Area loaded")
        except ImportError:
            if verbose:
                print("[Mergen] Wernicke Area not available")
        except Exception as e:
            if verbose:
                print(f"[Mergen] Wernicke Area initialization failed: {e}")
            self.wernicke = None

        # 5. Intent Analyzer
        mx_mem_path = getattr(self.config, 'MX_MEMORY_PATH', './mergen_matrix_memory.json')
        self.analyzer = IntentAnalyzer(memory_path=mx_mem_path)

        # 6. Conversation Memory
        self.conv_memory = ConversationMemory(
            window_size=20,
            persistence_path='./mergen_conversation_memory.json',
        )
        if verbose:
            print(f"[Mergen] Conversation memory: {self.conv_memory.total_turns} previous turns")

        # 8. Broca Area (fallback expression)
        self.broca = self._init_broca()

        # ── 4. Neural Core (MergenBrain) & Enhanced Wrapper ──
        self.device = getattr(self.config, 'DEVICE', 'cpu')
        self.brain = MergenBrain(vocab_size=self.vocab.size(), config=self.config)

        # Load previous knowledge base state
        kb_path = getattr(self.config, 'MX_KNOWLEDGE_PATH', './mergen_knowledge.mx')
        if Path(kb_path).exists():
            if verbose:
                print(f"[Mergen] Loading knowledge base from {kb_path}")
            self.brain.load(kb_path)
        else:
            if verbose:
                print(f"[Mergen] No previous knowledge base found at {kb_path}, starting fresh")

        self.enhanced_brain = EnhancedMergenBrain(
            brain=self.brain,
            config=self.config,
            use_wernicke=False,
            device=self.device,
        )
        if getattr(self, 'wernicke', None):
            self.enhanced_brain.wernicke = self.wernicke
            self.enhanced_brain.use_wernicke = True

        self.response_generator = ResponseGenerator(vocab=self.vocab, brain=self.brain)


        # ── 11. Biological Core (Limbic + Hebbian) ──
        # BUG-02 FIX: hebbian_trace tensor'una eş zamanlı erişimi korumak için merkezi lock.
        # Bu lock hem HebbianRAGBridge (arka plan thread) hem de gelecekteki
        # doğrudan trace güncellemeleri tarafından kullanılmalıdır.
        self._hebbian_lock = threading.RLock()

        try:
            self.hebbian_engine = CorticalColumn(
                n_pre=768,       # Wernicke spike boyutu
                n_post=self.vocab.size(),
                n_hidden=1024,   # İç katman genişliği (L4, L23)
                lateral_k=50,    # k-WTA: top-50 nöron aktif (yeterli çözünürlük sağlar)
                device=getattr(self.config, 'DEVICE', 'cpu')
            )
            
            # Load Innate Priors (Semantic Graph)
            # Öncelik: mergen_cortical_priors.pt (v2.0, çok katmanlı)
            # Fallback: mergen_innate_priors.pt (v1.0, sadece L5)
            cortical_priors_path = Path('./mergen_cortical_priors.pt')
            legacy_priors_path = Path('./mergen_innate_priors.pt')

            priors_loaded = False
            if cortical_priors_path.exists():
                try:
                    state = torch.load(
                        cortical_priors_path,
                        map_location=self.hebbian_engine.device,
                        weights_only=True,
                    )
                    if isinstance(state, dict) and state.get('version') == '2.0':
                        loaded = 0
                        if hasattr(self.hebbian_engine, 'L4') and 'L4_weights' in state:
                            w4 = state['L4_weights']
                            if w4.shape == self.hebbian_engine.L4.weights.shape:
                                self.hebbian_engine.L4.weights.data = w4.to(self.hebbian_engine.device)
                                loaded += 1
                        if hasattr(self.hebbian_engine, 'L23') and 'L23_weights' in state:
                            w23 = state['L23_weights']
                            if w23.shape == self.hebbian_engine.L23.weights.shape:
                                self.hebbian_engine.L23.weights.data = w23.to(self.hebbian_engine.device)
                                loaded += 1
                        if hasattr(self.hebbian_engine, 'L5') and 'L5_weights' in state:
                            w5 = state['L5_weights']
                            if w5.shape == self.hebbian_engine.L5.weights.shape:
                                self.hebbian_engine.L5.weights.data = w5.to(self.hebbian_engine.device)
                                loaded += 1
                        if hasattr(self.hebbian_engine, 'L6') and 'L6_weights' in state:
                            w6 = state['L6_weights']
                            if w6.shape == self.hebbian_engine.L6.weights.shape:
                                self.hebbian_engine.L6.weights.data = w6.to(self.hebbian_engine.device)
                                loaded += 1
                        if loaded > 0:
                            priors_loaded = True
                            if self.verbose:
                                print(f"[Mergen] \u2713 Cortical Priors loaded ({loaded}/3 layers)")
                    else:
                        if self.verbose:
                            print(f"[Mergen] \u26a0 Cortical priors format tanınmadı, atlandı.")
                except Exception as e:
                    if self.verbose:
                        print(f"[Mergen] \u26a0 Cortical priors yüklenemedi: {e}")

            if not priors_loaded and legacy_priors_path.exists():
                try:
                    innate_weights = torch.load(
                        legacy_priors_path,
                        map_location=self.hebbian_engine.device,
                        weights_only=True,
                    )
                    if innate_weights.shape == self.hebbian_engine.weights.shape:
                        self.hebbian_engine.weights.data = innate_weights
                        priors_loaded = True
                        if self.verbose:
                            print(f"[Mergen] \u2713 Legacy Innate Priors loaded (L5 only)")
                    else:
                        if self.verbose:
                            print(
                                f"[Mergen] \u26a0 Legacy priors shape {innate_weights.shape} "
                                f"!= L5 weights {self.hebbian_engine.weights.shape}. Atlandı."
                            )
                except Exception as e:
                    if self.verbose:
                        print(f"[Mergen] \u26a0 Legacy priors yüklenemedi: {e}")

            if not priors_loaded and self.verbose:
                print(f"[Mergen] \u26a0 Priors bulunamadı — fresh-init. 'python scripts/generate_innate_priors.py' çalıştırın.")


            
            wernicke_inst = getattr(self, 'wernicke', None)
            
            self.limbic = LimbicExecutiveLayer(
                mergen_engine=self.hebbian_engine,
                broca=self.broca,
                wernicke=wernicke_inst,
                mx_path=getattr(self.config, 'MX_WEIGHTS_PATH', './mergen_weights.mx'),
                user_id="default",
                rag=getattr(self, 'rag', None)
            )
            
            if self.verbose:
                print(f"[Mergen] Biological Core (Limbic + Hebbian) initialized.")
        except Exception as e:
            if self.verbose:
                print(f"[Mergen] ⚠ Biological Core initialization failed: {e}")
            self.limbic = None

        # 9. Türkçe Morfoloji (UTF-8 + Zeyrek)
        self.morph: Optional[Any] = None
        if _MORPH_AVAILABLE:
            try:
                self.morph = TurkishMorph(verbose=verbose)
            except Exception as e:
                if self.verbose:
                    print(f"[Mergen] TurkishMorph baslatilamadi: {e}")
                self.morph = None

        # 10. RAG Motoru (Transformer-free BioVectorizer + HTM)
        self.rag: Optional[Any] = None
        self._rag_loader: Optional[Any] = None
        self._hebb_bridge: Optional[Any] = None
        if _RAG_AVAILABLE:
            try:
                self.rag = RAGEngine(db_path="./mergen_rag_db")
                ok = self.rag.initialize(verbose=verbose)
                if ok:
                    # Hebbian-RAG köprüsü — BUG-02 FIX: brain_lock iletiliyor
                    self._hebb_bridge = HebbianRAGBridge(
                        brain=self.brain,
                        vocab=self.vocab,
                        verbose=verbose,
                        brain_lock=self._hebbian_lock,  # BUG-02 FIX
                    )
                    self._rag_loader = TurkishDataLoader(
                        rag_engine=self.rag,
                        hebbian_bridge=self._hebb_bridge,
                        verbose=verbose,
                    )
                    if verbose:
                        print(
                            f"[Mergen] RAG aktif — {self.rag.count()} kayıt "
                            f"(BioVectorizer + HTM). Veri için: rag:yukle"
                        )
                else:
                    self.rag = None
            except Exception as _e:
                if verbose:
                    print(f"[Mergen] RAG başlatılamadı: {_e}")
                self.rag = None

        # Interaction log
        self.interaction_log = []
        self.last_recall_metadata: Dict[str, Any] = {}
        self.is_running = True

        # Background reflection
        self._reflection_thread: Optional[threading.Thread] = None
        self._reflection_done = threading.Event()

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_shutdown)

        if verbose:
            print(f"[Mergen] ✓ Digital Brain v{self.VERSION} initialized.\n")

    # ─────────────────────────────────────────────────────────
    #  BROCA INITIALIZATION
    # ─────────────────────────────────────────────────────────

    def _init_broca(self) -> BrocaArea:
        """Initialize BrocaArea for fallback expression."""
        try:
            from cognitive.language_engine import LanguageEngine
            lang_engine = LanguageEngine(
                motor_layer_size=min(self.vocab.size() * 4, 10_000),
                motor_rows=40,
                motor_cols=50,
                vocabulary=self.vocab.all_words,
                temperature=getattr(self.config, 'TEMPERATURE', 0.9),
                top_k=getattr(self.config, 'TOP_K', 40),
                device=getattr(self.config, 'DEVICE', 'cpu'),
            )
            broca = BrocaArea(
                language_engine=lang_engine,
                n_neurons=self.vocab.size(),
                concept_vocabulary=self.vocab.all_words,
                device=getattr(self.config, 'DEVICE', 'cpu'),
            )
            broca.vocab = self.vocab
            return broca
        except (ImportError, TypeError) as e:
            if self.verbose:
                print(f"[Mergen] ⚠ LanguageEngine unavailable ({e})")

            class StubEngine:
                def __init__(self, vocab_words, motor_sz: int = 2000):
                    self.vocabulary = vocab_words
                    self.motor_layer_size = motor_sz
                    self.device = 'cpu'
                def speak(self, *args, **kwargs):
                    import random
                    return random.choice(self.vocabulary[:50])
                def strengthen_association(self, *args, **kwargs):
                    return None

            return BrocaArea(
                language_engine=StubEngine(self.vocab.all_words),
                n_neurons=self.vocab.size(),
                concept_vocabulary=self.vocab.all_words,
                vocab=self.vocab,
            )

    # ─────────────────────────────────────────────────────────
    #  MAIN PIPELINE: perceive → think → respond
    # ─────────────────────────────────────────────────────────

    def respond(self, user_input: str) -> str:
        """
        Main conversational pipeline:
        PHASE 4: BIOLOGICAL CORE INTEGRATION
        Routes directly to Limbic System.
        """
        if not user_input or not user_input.strip():
            return ""

        # Step 0: Ensure Limbic is alive; lazy-init if respond() called without run()
        if hasattr(self, 'limbic') and self.limbic is not None:
            if not getattr(self.limbic, 'is_running', False):
                try:
                    self.limbic.wake_up()
                except Exception as e:
                    if self.verbose:
                        print(f"[Mergen] Limbic wake_up failed: {e}")
            self.limbic.trigger_wakeup()

        # Step 1: Resolve pronouns using conversation context
        resolved_input = self.conv_memory.resolve_references(user_input)

        report = self.analyzer.analyze_intent(resolved_input)
        intent = report.get('primary_intent', 'UNKNOWN')
        subject = report.get('subject')
        morphology = report.get('morphology', {})
        derived_subject = self._extract_question_subject(resolved_input)
        if derived_subject and self._should_override_subject(subject, derived_subject):
            subject = derived_subject
            report['subject'] = subject
        biological_response = ""

        # Step 2: Biological activation for inner-state continuity
        if getattr(self, 'limbic', None) and getattr(self.limbic, 'is_running', False):
            biological_response = self.limbic.respond(resolved_input)
            
            # Print internal thought for visibility
            if self.verbose and self.limbic.last_thought:
                print(f"  [Mergen İç-Sesi] Ateşlenen kavramlar: {self.limbic.last_thought}")
                
        else:
            # Fallback if Limbic is not running
            biological_response = "Limbic system is offline."

        # Check if it looks like an arithmetic query
        is_math = self._looks_like_math(resolved_input)

        knowledge_intents = {'INQUIRY', 'IDENTITY', 'GREETING', 'WELLBEING', 'GRATITUDE'}
        is_question = bool(morphology.get('is_question')) or self._looks_like_question(resolved_input) or is_math

        if intent in knowledge_intents or is_question:
            self._last_query_concepts = list(self._content_tokens(resolved_input))
            facts = self._recall_knowledge(resolved_input, intent, subject)
            response = self.response_generator.generate(
                query=resolved_input,
                intent=intent,
                subject=subject,
                knowledge_facts=facts,
                conversation_context=self.conv_memory.get_context_summary(),
            )
        else:
            response = biological_response or self.response_generator.generate(
                query=resolved_input,
                intent=intent,
                subject=subject,
                knowledge_facts=[],
                conversation_context=self.conv_memory.get_context_summary(),
            )

        # Step 3: Active Learning (Keep this to learn from statements)
        cleaned_input_for_learning = resolved_input
        # Remove any leading/trailing math operators if any, but actually _try_learn_from_input handles it
        learned_fact = self._try_learn_from_input(resolved_input, intent)
        if learned_fact:
            response = f"{response}\n(Bunu öğrendim: {learned_fact})"

        # Step 4: Store in conversation memory
        self.conv_memory.add_turn(
            user_input=user_input,
            response=response,
            intent=intent,
            subject=subject,
        )

        # Step 5: Log interaction
        self.interaction_log.append({
            'timestamp': time.time(),
            'input': user_input[:200],
            'intent': intent,
            'confidence': 1.0,
            'response': response[:300],
            'internal_thought': getattr(self.limbic, 'last_thought', ''),
        })

        return response

    def _looks_like_question(self, text: str) -> bool:
        """Language-level question guard used before active learning."""
        text_lower = (text or "").strip().lower()
        if not text_lower:
            return False
        if "?" in text_lower:
            return True
        if "ne yapar" in text_lower:
            return True
        question_phrases = (
            "nedir", "ne demek", "ne ise", "ne işe", "neyi", "neye",
            "neden", "niye", "nasil", "nasıl", "hangi", "kim", "kimdir",
            "nerede", "nereye", "kac", "kaç", "what", "why", "how",
            "who", "where", "when",
        )
        return any(phrase in text_lower for phrase in question_phrases)

    def _looks_like_math(self, text: str) -> bool:
        """Check if the text represents an arithmetic query."""
        text_lower = (text or "").strip().lower()
        if not text_lower:
            return False
        # Direct digits and operators
        if bool(re.search(r'\d+\s*[\+\-\*\/x]\s*\d+', text_lower)):
            return True
        # Verbal math words
        math_words = {'artı', 'arti', 'eksi', 'çarpı', 'carpi', 'bölü', 'bolu', 'esittir', 'eşit', 'eder', '/'}
        input_toks = set(re.findall(r'\w+', text_lower))
        if math_words & input_toks:
            base_num_words = {
                'sıfır', 'sifir', 'bir', 'iki', 'üç', 'uc', 'dört', 'dort', 'beş', 'bes', 'altı', 'alti', 
                'yedi', 'sekiz', 'dokuz', 'on', 'yirmi', 'otuz', 'kırk', 'kirk', 'elli', 'altmış', 'altmis', 
                'yetmiş', 'yetmis', 'seksen', 'doksan', 'yüz', 'yuz'
            }
            has_num = False
            for tok in input_toks:
                if tok.isdigit():
                    has_num = True
                    break
                if tok in base_num_words:
                    has_num = True
                    break
                # Check for compound numbers like 'oniki', 'yirmibir'
                for tens in ['on', 'yirmi', 'otuz', 'kirk', 'kırk', 'elli', 'altmis', 'altmış', 'yetmis', 'yetmiş', 'seksen', 'doksan', 'yuz', 'yüz']:
                    if tok.startswith(tens) and len(tok) > len(tens) and tok[len(tens):] in base_num_words:
                        has_num = True
                        break
                if has_num:
                    break
            if has_num:
                return True
        return False

    def _extract_question_subject(self, text: str) -> Optional[str]:
        text_clean = re.sub(r'\s+', ' ', (text or '').strip())
        if not text_clean:
            return None
        low = text_clean.lower().rstrip(' ?!.')

        patterns = [
            r'(.+?)\s+ne\s+i[şs]e\s+yarar$',
            r'(.+?)\s+ne\s+yapar$',
            r'(.+?)\s+neyi\s+de[ğg]i[şs]tirir$',
            r'(.+?)\s+ne\s+demek$',
            r'(.+?)\s+nedir$',
            r'(.+?)\s+kimdir$',
            r'(.+?)\s+nas[ıi]l\s+temsil\s+edilir$',
        ]
        for pattern in patterns:
            match = re.search(pattern, low, flags=re.UNICODE)
            if not match:
                continue
            candidate = match.group(1).strip()
            candidate = re.sub(r'[^\wçğıöşüÇĞİÖŞÜ\s-]', '', candidate).strip()
            if candidate and candidate not in {'ne', 'kim', 'neden', 'nasil', 'nasıl'}:
                return ' '.join(part.capitalize() if part.islower() else part for part in candidate.split())
        return None

    def _should_override_subject(self, current: Optional[str], derived: str) -> bool:
        if not derived:
            return False
        if not current:
            return True
        current_words = self._content_tokens(str(current))
        derived_words = self._content_tokens(derived)
        if len(derived_words) > len(current_words):
            return True
        current_low = str(current).lower()
        weak_subjects = {'konsolidasyonu', 'konsolidasyon', 'öğrenme', 'ogrenme'}
        return current_low in weak_subjects

    def _try_learn_from_input(self, user_input: str, intent: str) -> Optional[str]:
        """
        Aktif öğrenme: Kullanıcı bir bilgi veriyorsa (soru değil, bildirme),
        bunu KB'ye fact olarak ekle ve kalıcı kaydet.
        """
        import re

        text = user_input.strip()
        if len(text) < 5:
            return None

        text_lower = text.lower()
        if intent in {'INQUIRY', 'IDENTITY', 'GREETING', 'WELLBEING', 'GRATITUDE'}:
            return None
        if self._looks_like_question(text) or self._looks_like_math(text):
            return None

        # Skip questions — they are not learnable facts
        question_words = r'\b(nedir|ne demek|nasıl|neden|niçin|kimdir|nerede|hangi|kaç|mi|mı|mu|mü|kim|ne|nere|niye|nereye|neden|niçin)\b'
        if re.search(question_words, text_lower):
            return None

        # Skip interrogative patterns
        if re.search(r'\b(neresi|kimin|neyin|nasıl|neden)\b', text_lower):
            return None

        # Skip commands
        if text.startswith('/'):
            return None

        # Skip conversational inputs
        conv_patterns = [
            r'^(merhaba|selam|hey|sa|as|selamlar|iyi akşamlar|günaydın|iyi geceler)\b',
            r'^(teşekkür|sağol|eyvallah|tamam|evet|hayır|peki|olur|tamam mı)\b',
            r'^(nasılsın|naber|ne yapıyorsun|iyi misin)\b',
            r'^(görüşürüz|hoşçakal|bye|güle güle)\b',
        ]
        for pat in conv_patterns:
            if re.search(pat, text_lower):
                return None

        # Skip very short sentences
        words = text.split()
        if len(words) < 3:
            return None

        # Clean trailing punctuation
        fact_text = text.rstrip('.!?').strip()

        # Check for duplicates
        fact_lower = fact_text.lower()
        for existing in self.brain.semantic.knowledge_base + self.brain.episodic.events:
            existing_text = existing.get('text', '').lower()
            # Exact match or very similar
            if fact_lower == existing_text:
                return None
            # One contains the other and both are short
            if len(fact_text) < 40 and (fact_lower in existing_text or existing_text in fact_lower):
                return None

        # Add to KB
        tokens = re.findall(r'\w+', fact_text.lower())
        matched_ids = [self.vocab.get_id(t) for t in tokens if self.vocab.contains(t)]
        
        # Kullanıcı konuşmaları Episodik belleğe
        kb_idx = self.brain.episodic.add_event(fact_text, list(set(matched_ids)), weight=1.0)
        
        # Episodik belleği concept_index'e eklemiyoruz çünkü geçicidir ve indexleri kayar.
        # recall_raw ve recall_all_about zaten her iki belleği de tarayarak bulacaktır.

        # Save to disk
        try:
            brain_path = getattr(self.config, 'MX_KNOWLEDGE_PATH', './mergen_knowledge.mx')
            self.brain.save(brain_path)
        except Exception as e:
            if self.verbose:
                print(f"[Mergen] Knowledge save failed after learning: {e}")

        if self.verbose:
            print(f"[Mergen 📚] Öğrendim: {fact_text[:80]}")

        return fact_text

    def _recall_knowledge(
        self,
        query: str,
        intent: str,
        subject: Optional[str],
        semantic_fallback: bool = False,
    ) -> list:
        """
        Multi-strategy knowledge recall with intelligent merging.
        """
        candidates: List[Dict[str, Any]] = []
        seen_texts: Set[str] = set()
        semantic_used = False

        query_lower = query.lower()
        is_def_query = bool(re.search(
            r'\bnedir\b|\bne demek\b|\bne anlama\b|\bwhat is\b|\bwhat are\b|\bkimdir\b',
            query_lower
        ))
        subject_terms = self._subject_aliases(subject)
        raw_queries = self._query_aliases(query)

        # Strategy 1: Subject-based recall (for definition queries)
        if is_def_query:
            for subject_term in subject_terms:
                try:
                    facts = self.enhanced_brain.recall_all_about(subject_term, top_k=8)
                    for f in facts:
                        self._add_recall_candidate(
                            candidates, seen_texts, f,
                            source='kb_subject',
                            query=query,
                            subject=subject,
                        )
                except Exception as e:
                    if self.verbose:
                        print(f"[Mergen] Subject recall failed ({subject_term}): {e}")

        # Strategy 2: Raw text recall from full query
        for raw_query in raw_queries:
            try:
                facts = self.enhanced_brain.recall_raw(raw_query, top_k=8)
                for f in facts:
                    self._add_recall_candidate(
                        candidates, seen_texts, f,
                        source='kb_raw',
                        query=query,
                        subject=subject,
                    )
            except Exception as e:
                if self.verbose:
                    print(f"[Mergen] Raw recall failed ({raw_query}): {e}")

        # Strategy 3: Also search with subject-only (broader match)
        if subject_terms and not is_def_query:
            for subject_term in subject_terms:
                try:
                    facts = self.enhanced_brain.recall_raw(subject_term, top_k=5)
                    for f in facts:
                        self._add_recall_candidate(
                            candidates, seen_texts, f,
                            source='kb_subject_raw',
                            query=query,
                            subject=subject,
                        )
                except Exception as e:
                    if self.verbose:
                        print(f"[Mergen] Subject raw recall failed ({subject_term}): {e}")

        # Strategy 5: RAG — BioVectorizer + HTM biyolojik arama
        if self.rag and self.rag.ready:
            try:
                # Türkçe morfoloji ile sorgu normalizasyonu
                if self.morph:
                    _, concepts = self.morph.normalize_query(query, vocab=self.vocab)
                    # Önce kök kavramlarla ara, sonra ham soru ile de dene
                    rag_queries = []
                    if concepts:
                        rag_queries.append(" ".join(concepts))
                    for subject_term in subject_terms:
                        rag_queries.append(self.morph.lemmatize(subject_term, vocab=self.vocab))
                    rag_queries.append(query)
                    rag_queries.extend(raw_queries)
                else:
                    rag_queries = list(subject_terms) + raw_queries

                for rq in rag_queries:
                    if not rq or not rq.strip():
                        continue
                    rag_hits = self.rag.search(rq, top_k=5)
                    for hit in rag_hits:
                        self._add_recall_candidate(
                            candidates, seen_texts, hit,
                            source='rag',
                            query=query,
                            subject=subject,
                            original_source=hit.get('source', 'rag'),
                        )
            except Exception as e:
                if self.verbose:
                    print(f"[Mergen] RAG recall failed: {e}")

        # Expensive Wernicke semantic fallback is opt-in only.
        if semantic_fallback and not candidates and getattr(self.enhanced_brain, 'use_wernicke', False):
            try:
                facts = self.enhanced_brain.recall_semantic(query, top_k=5)
                semantic_used = True
                for f in facts:
                    self._add_recall_candidate(
                        candidates, seen_texts, f,
                        source='semantic_fallback',
                        query=query,
                        subject=subject,
                    )
            except Exception as e:
                if self.verbose:
                    print(f"[Mergen] Semantic recall failed: {e}")

        self._score_recall_candidates(candidates, query=query, subject=subject)
        candidates.sort(key=lambda x: (-x.get('final_score', 0.0), -x.get('base_score', 0.0)))
        selected = candidates[:8]

        self.last_recall_metadata = {
            'semantic_fallback_used': semantic_used,
            'candidate_count': len(candidates),
            'returned_count': len(selected),
            'top_candidates': [
                {
                    'text': c.get('text', '')[:180],
                    'source': c.get('source'),
                    'original_source': c.get('original_source'),
                    'base_score': c.get('base_score', 0.0),
                    'rag_score': c.get('rag_score', 0.0),
                    'hebbian_score': c.get('hebbian_score', 0.0),
                    'limbic_score': c.get('limbic_score', 0.0),
                    'final_score': c.get('final_score', 0.0),
                    'matched_concepts': c.get('matched_concepts', [])[:8],
                }
                for c in selected[:5]
            ],
        }
        return selected

    def _add_recall_candidate(
        self,
        candidates: List[Dict[str, Any]],
        seen_texts: Set[str],
        fact: Dict[str, Any],
        source: str,
        query: str,
        subject: Optional[str],
        original_source: Optional[str] = None,
    ) -> None:
        text = (fact.get('text') or '').strip()
        if not text:
            return
        if self._is_recall_question_residue(text, subject):
            return

        key = re.sub(r'\s+', ' ', text[:160].lower())
        if key in seen_texts:
            return
        seen_texts.add(key)

        base_score = float(fact.get('relevance', fact.get('weight', 0.0)) or 0.0)
        candidate = {
            'text': text,
            'source': source,
            'original_source': original_source or fact.get('source') or source,
            'base_score': round(base_score, 4),
            'relevance': round(base_score, 4),
            'rag_score': 0.0,
            'hebbian_score': 0.0,
            'limbic_score': 0.0,
            'final_score': 0.0,
            'matched_concepts': self._matched_vocab_concepts(text, query, subject),
        }
        if 'kb_idx' in fact:
            candidate['kb_idx'] = fact['kb_idx']
        if 'is_definition' in fact:
            candidate['is_definition'] = bool(fact.get('is_definition'))
        candidates.append(candidate)

    def _score_recall_candidates(
        self,
        candidates: List[Dict[str, Any]],
        query: str,
        subject: Optional[str],
    ) -> None:
        raw_query_tokens = self._content_tokens(query)
        query_tokens = {self._fold_text(t) for t in raw_query_tokens}
        raw_subject_tokens = self._content_tokens(subject or '')
        subject_tokens = {self._fold_text(t) for t in raw_subject_tokens}
        limbic_concepts = self._limbic_signal_concepts()

        for candidate in candidates:
            text = candidate.get('text', '')
            raw_text_tokens = self._content_tokens(text)
            text_tokens = {self._fold_text(t) for t in raw_text_tokens}
            overlap = len(query_tokens & text_tokens) / max(1, len(query_tokens))
            subject_overlap = (
                len(subject_tokens & text_tokens) / max(1, len(subject_tokens))
                if subject_tokens else 0.0
            )
            alias_subject_score = self._subject_signal_score(text, subject)

            definition_bonus = 0.12 if candidate.get('is_definition') else 0.0
            if subject and self._starts_with_subject(text, subject):
                definition_bonus += 0.10
            definition_bonus += self._definition_quality_bonus(text, subject)

            rag_score = 0.0
            if candidate.get('source') == 'rag':
                rag_score = min(1.0, 0.20 + float(candidate.get('base_score', 0.0)))

            hebbian_score = self._hebbian_trace_score(candidate.get('matched_concepts', []))
            limbic_score = self._limbic_candidate_score(
                limbic_concepts=limbic_concepts,
                candidate_concepts=candidate.get('matched_concepts', []),
                text_tokens=text_tokens,
            )

            base = min(1.0, float(candidate.get('base_score', 0.0)))
            final = (
                base * 0.45
                + overlap * 0.20
                + max(subject_overlap, alias_subject_score) * 0.22
                + rag_score * 0.15
                + hebbian_score * 0.16
                + limbic_score * 0.15
                + definition_bonus
            )

            candidate['query_overlap_score'] = round(overlap, 4)
            candidate['subject_score'] = round(max(subject_overlap, alias_subject_score) + definition_bonus, 4)
            candidate['rag_score'] = round(rag_score, 4)
            candidate['hebbian_score'] = round(hebbian_score, 4)
            candidate['limbic_score'] = round(limbic_score, 4)
            candidate['final_score'] = round(final, 4)

    def _limbic_signal_concepts(self) -> List[str]:
        concepts = []
        limbic = getattr(self, 'limbic', None)
        last_thought = getattr(limbic, 'last_thought', '') if limbic else ''

        for token in re.split(r'\s*->\s*|\s+', last_thought or ''):
            token = token.strip().lower()
            if token:
                concepts.extend(self._matched_vocab_concepts(token))

        internal = getattr(limbic, 'internal_thoughts', None) if limbic else None
        if internal:
            try:
                for item in list(internal)[-5:]:
                    thought = item.get('thought', '') if isinstance(item, dict) else str(item)
                    concepts.extend(self._matched_vocab_concepts(thought))
            except Exception as e:
                if self.verbose:
                    print(f"[Mergen] Limbic thought history scoring skipped: {e}")

        # Mevcut sorgunun kavramlarını dahil et — böylece soru ile KB fact'ları
        # arasında Limbic overlap sağlanır (last_thought boş olsa bile çalışır).
        # Limbic katmanı devre dışıysa (ablasyon) bu yolu da kapat.
        if limbic is not None:
            query_concepts = getattr(self, '_last_query_concepts', [])
            if query_concepts:
                concepts.extend(q for q in query_concepts if self.vocab.contains(q))

        return sorted(set(concepts))

    def _subject_signal_score(self, text: str, subject: Optional[str]) -> float:
        aliases = self._subject_aliases(subject)
        if not aliases:
            return 0.0
        folded_text = self._fold_text(text)
        hits = 0
        for alias in aliases:
            folded_alias = self._fold_text(alias)
            if folded_alias and folded_alias in folded_text:
                hits += 1
        return min(1.0, hits / max(1, min(3, len(aliases))))

    def _definition_quality_bonus(self, text: str, subject: Optional[str]) -> float:
        if not subject:
            return 0.0
        folded = self._fold_text(text)
        aliases = [self._fold_text(a) for a in self._subject_aliases(subject)]
        starts_with_alias = any(alias and folded.startswith(alias[:4]) for alias in aliases)
        definition_markers = (
            ' arasinda cekim ', ' cekim etkisi ', ' uzay ve zamani buk',
            ' sinaptik izleri ', ' hafiza izlerini ', ' geri cagir',
            ' yaniti destekler', ' biyolojiden ilham',
        )
        bonus = 0.0
        if starts_with_alias:
            bonus += 0.18
        if any(marker in folded for marker in definition_markers):
            bonus += 0.16
        broad_context_markers = (
            'gunes sistemi', 'mars', 'filmde', 'ev sahipligi', 'yasami cercevesinde',
            'evlendirildi', 'destansi', 'kara delik',
        )
        if any(marker in folded for marker in broad_context_markers):
            bonus -= 0.22
        return max(-0.25, min(0.35, bonus))

    def _limbic_candidate_score(
        self,
        limbic_concepts: List[str],
        candidate_concepts: List[str],
        text_tokens: Set[str],
    ) -> float:
        if not limbic_concepts:
            return 0.0

        limbic_set = set(limbic_concepts)
        candidate_set = set(candidate_concepts)
        direct = len(limbic_set & candidate_set) / max(1, len(limbic_set))

        soft_hits = 0
        for concept in limbic_set:
            if any(token.startswith(concept[:4]) or concept.startswith(token[:4]) for token in text_tokens if len(token) >= 4):
                soft_hits += 1
        soft = soft_hits / max(1, len(limbic_set))

        trace_affinity = self._hebbian_trace_score(list(limbic_set & candidate_set))
        return min(1.0, direct * 0.55 + soft * 0.35 + trace_affinity * 0.10)

    def _content_tokens(self, text: str) -> Set[str]:
        stop = {
            'bir', 've', 'ile', 'bu', 'su', 'o', 'de', 'da', 'mi', 'mı', 'mu',
            'mü', 'ki', 'icin', 'için', 'gibi', 'olan', 'cok', 'çok', 'daha',
            'en', 'ne', 'nedir', 'demek', 'neden', 'nasil', 'nasıl', 'hangi',
            'kim', 'neyi', 'neye', 'is', 'are', 'the', 'what', 'why', 'how',
        }
        return {
            token for token in re.findall(r'\w+', (text or '').lower(), flags=re.UNICODE)
            if len(token) > 2 and token not in stop
        }

    def _is_recall_question_residue(self, text: str, subject: Optional[str]) -> bool:
        folded = (text or '').lower()
        tokens = re.findall(r'\w+', folded, flags=re.UNICODE)
        if not tokens:
            return True
        if '?' in text:
            return True

        question_terms = {
            'ne', 'neyi', 'neye', 'nedir', 'neler', 'neden', 'niye',
            'nasil', 'nasıl', 'kim', 'kimdir', 'nerede', 'nereye',
            'kac', 'kaç', 'what', 'why', 'how', 'who', 'where', 'when',
        }
        question_hits = sum(1 for token in tokens if token in question_terms)
        if len(tokens) <= 4 and question_hits > 0:
            return True

        subject_tokens = self._content_tokens(subject or '')
        if len(tokens) <= 6 and question_hits > 0 and subject_tokens:
            if self._content_tokens(text) & subject_tokens:
                return True
        return False

    def _subject_aliases(self, subject: Optional[str]) -> List[str]:
        terms = []
        if subject:
            terms.append(str(subject).strip())

        alias_map = {
            'yerçekimi': ['kütleçekim', 'gravitasyonel', 'gravity'],
            'yercekimi': ['kütleçekim', 'gravitasyonel', 'gravity'],
            'kütleçekim': ['yerçekimi', 'gravitasyonel', 'gravity'],
            'kutlecekim': ['yerçekimi', 'gravitasyonel', 'gravity'],
            'rüya': ['dream', 'konsolidasyon'],
            'ruya': ['dream', 'konsolidasyon'],
            'dream': ['rüya', 'konsolidasyon'],
            'rüya konsolidasyonu': ['dream konsolidasyonu', 'konsolidasyon', 'dream'],
            'ruya konsolidasyonu': ['dream konsolidasyonu', 'konsolidasyon', 'dream'],
            'dream konsolidasyonu': ['rüya konsolidasyonu', 'konsolidasyon', 'rüya'],
        }

        folded = self._fold_text(subject or '')
        for key, aliases in alias_map.items():
            if key in folded:
                terms.extend(aliases)

        return self._dedupe_terms(terms)

    def _query_aliases(self, query: str) -> List[str]:
        terms = [query]
        replacements = [
            ('yerçekimi', 'kütleçekim'),
            ('yercekimi', 'kütleçekim'),
            ('rüya konsolidasyonu', 'dream konsolidasyonu'),
            ('ruya konsolidasyonu', 'dream konsolidasyonu'),
            ('dream konsolidasyonu', 'rüya konsolidasyonu'),
        ]
        folded_query = self._fold_text(query)
        for source, target in replacements:
            if source in folded_query:
                terms.append(re.sub(source, target, folded_query, flags=re.IGNORECASE))
                terms.append(target)
        return self._dedupe_terms(terms)

    def _dedupe_terms(self, terms: List[str]) -> List[str]:
        result = []
        seen = set()
        for term in terms:
            clean = re.sub(r'\s+', ' ', str(term or '').strip())
            if not clean:
                continue
            key = self._fold_text(clean)
            if key in seen:
                continue
            seen.add(key)
            result.append(clean)
        return result

    def _fold_text(self, text: str) -> str:
        table = str.maketrans({
            'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
            'Ç': 'c', 'Ğ': 'g', 'İ': 'i', 'I': 'i', 'Ö': 'o', 'Ş': 's', 'Ü': 'u',
        })
        return (text or '').translate(table).lower()

    def _matched_vocab_concepts(
        self,
        text: str,
        query: str = '',
        subject: Optional[str] = None,
    ) -> List[str]:
        tokens = self._content_tokens(text or '')
        concepts = []
        for token in tokens:
            if self.vocab.contains(token):
                concepts.append(token)
                continue
            for suffix_len in (4, 3, 2, 1):
                if len(token) > suffix_len + 2:
                    stem = token[:-suffix_len]
                    if self.vocab.contains(stem):
                        concepts.append(stem)
                        break
        return sorted(set(concepts))

    def _hebbian_trace_score(self, concepts: List[str]) -> float:
        trace = getattr(self.brain, 'hebbian_trace', None)
        if trace is None or not concepts:
            return 0.0

        ids = [self.vocab.get_id(c, None) for c in concepts if self.vocab.contains(c)]
        ids = [idx for idx in ids if idx is not None and 0 <= idx < len(trace)]
        if not ids:
            return 0.0

        try:
            values = trace[ids].detach().abs().float()
            sorted_trace = trace.detach().abs().float().sort().values
            n = len(sorted_trace)
            if n == 0:
                return 0.0
            # Rank percentile: büyük bir max değeri diğer kavramları ezmez.
            # Kavramın trace'deki yüzdelik dilimi doğrudan skor olur.
            ranks = torch.searchsorted(sorted_trace, values)
            return float(torch.clamp(ranks.float().mean() / n, 0.0, 1.0).item())
        except Exception as e:
            if self.verbose:
                print(f"[Mergen] Hebbian trace scoring failed: {e}")
            return 0.0

    def _starts_with_subject(self, text: str, subject: str) -> bool:
        first_words = re.findall(r'\w+', (text or '').lower(), flags=re.UNICODE)
        subject_tokens = self._content_tokens(subject)
        if not first_words or not subject_tokens:
            return False
        return any(first_words[0].startswith(s[:4]) for s in subject_tokens if len(s) >= 4)

    def _broca_generate(
        self, neural_intent, original_query: str, report: Dict
    ) -> str:
        """Fallback to BrocaArea for response generation."""
        intent = report.get('primary_intent', 'UNKNOWN')
        subject = report.get('subject')

        for method_name in ('generate', 'express', 'speak'):
            if hasattr(self.broca, method_name):
                method = getattr(self.broca, method_name)
                try:
                    return method(
                        neural_intent=neural_intent,
                        original_query=original_query,
                        intent=intent,
                        subject=subject,
                    )
                except TypeError:
                    try:
                        return method(neural_intent=neural_intent, original_query=original_query)
                    except TypeError:
                        try:
                            return method(neural_intent)
                        except Exception as e:
                            if self.verbose:
                                print(f"[Mergen] Broca fallback failed ({method_name}): {e}")
                            continue
        return "Anlayamadım, tekrar eder misin?"

    # ─────────────────────────────────────────────────────────
    #  FILE INGESTION (oku:file.txt)
    # ─────────────────────────────────────────────────────────

    def ingest_file(self, filepath: str) -> str:
        """
        Read external file and ACTIVELY LEARN from its contents.
        """
        path = Path(filepath.strip())
        if not path.exists():
            return f"[Mergen] Dosya bulunamadı: {filepath}"

        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                content = path.read_text(encoding='latin-1')
            except Exception as e:
                return f"[Mergen] Dosya okuma hatası: {e}"

        filename_stem = path.stem

        if not content.strip():
            return "[Mergen] Dosya boş."

        if self.verbose:
            print(f"[Mergen] '{path.name}' okunuyor ({len(content)} karakter)...")

        # Split into learning units
        import re
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip() and len(p.strip()) > 20]
        sentences = [s.strip() for s in re.split(r'[.!?]+\s*', content) if s.strip() and len(s.strip()) > 10]

        base_report = {
            'primary_intent': 'INQUIRY',
            'confidence_score': 0.9,
            'sentiment': {'sentiment_score': 0.0, 'excitement': 0.0},
            'subject': filename_stem,
        }

        # RAG indexleme: oku: komutu artık RAG veritabanına da yazar
        if self.rag and self.rag.ready:
            try:
                rag_sentences = [s for s in sentences[:300] if len(s) > 20]
                if rag_sentences:
                    indexed = self.rag.index_texts(
                        texts=rag_sentences,
                        source=filename_stem,
                    )
                    if self.verbose:
                        print(f"[Mergen] RAG: {indexed} cümle '{filename_stem}' kaynağıyla indekslendi.")
                    if hasattr(self, 'limbic') and self.limbic is not None:
                        # Sleep debt increases dynamically based on number of paragraphs ingested
                        self.limbic.increase_sleep_debt(len(paragraphs) * 2.0)
            except Exception as _rag_err:
                if self.verbose:
                    print(f"[Mergen] RAG indeksleme hatası: {_rag_err}")

        # 1. Cümle öğrenme — Tek geçişte hem Hebbian ağırlık güncellemesi hem de KB kaydı
        total_words = 0
        for sent in sentences[:200]:
            try:
                result = self.brain.learn_from_text(
                    text=sent,
                    vocab=self.vocab,
                    intent_report=base_report,
                    learning_rate=0.02,
                    reward=1.0,
                )
                total_words += result.get('words_learned', 0)
            except Exception as e:
                if self.verbose:
                    print(f"[Mergen] Sentence learning skipped: {e}")
                continue

        # Hebbian-RAG köprüsü: dosyadan gelen cümleler sinaptik izleri günceller
        if self._hebb_bridge is not None:
            self._hebb_bridge.update_from_batch(
                sentences[:150], source=filename_stem, reward=1.0
            )



        # Save
        try:
            brain_path = getattr(self.config, 'MX_KNOWLEDGE_PATH', './mergen_knowledge.mx')
            self.brain.save(brain_path)
        except Exception as e:
            if self.verbose:
                print(f"[Mergen] Knowledge save failed after ingestion: {e}")

        hebb_count = self._hebb_bridge.update_count if self._hebb_bridge else 0
        rag_count  = self.rag.count() if self.rag else 0

        summary = (
            f"'{path.name}' dosyasını öğrendim.\n"
            f"  • İşlenen paragraf:    {len(paragraphs)}\n"
            f"  • İşlenen cümle:       {len(sentences)}\n"
            f"  • Öğrenilen kelime:    {total_words}\n"
            f"  • Hafızadaki fact:     {self.brain.knowledge_size()}\n"
            f"  • Hebbian güncellemesi:{hebb_count}\n"
            f"  • RAG veritabanı:      {rag_count} kayıt"
        )

        # Background reflection
        self._start_reflection(path.name)

        return summary

    def _train_math(self, tier: int = 0, difficulty: int = 0) -> str:
        """
        Train the brain on arithmetic facts generated by MathTeacher.
        """
        try:
            from datasets.generators.math_teacher import MathTeacher, OP_SUBJECT, SAYI_ADI, OP_ADI
        except ImportError as e:
            return f"[Mergen] Matematik eğitim modülü bulunamadı: {e}"

        if self.verbose:
            print(f"[Mergen] Matematik eğitimi başlatılıyor: tier={tier}, difficulty={difficulty}...")

        teacher = MathTeacher(tier=tier, difficulty=difficulty)
        problems = teacher.enumerate_all()
        if not problems:
            return f"[Mergen] Bu tier ({tier}) ve difficulty ({difficulty}) için problem üretilemedi."

        # Setup intent report matching scripts/math_training.py
        intent_report = {
            "primary_intent": "EXPERIENCE",
            "confidence_score": 0.9,
            "sentiment": {"sentiment_score": 0.0, "excitement": 0.0},
            "subject": "aritmetik",
        }

        kb_before = self.brain.knowledge_size()
        facts_added = 0
        total_words = 0

        import re as _re
        for prob in problems:
            fact_text = MathTeacher.format_fact(prob)

            # 1. KB'ye doğrudan ekle (learn_from_text'in token overlap guard'ını bypass etmek için)
            kb_idx = len(self.brain.semantic.knowledge_base)
            tokens = _re.findall(r'\w+', fact_text.lower())
            concept_ids = []
            for tok in tokens:
                if self.vocab.contains(tok):
                    cid = self.vocab.get_id(tok)
                    if cid not in concept_ids:
                        concept_ids.append(cid)

            # Duplication check
            is_dup = False
            for existing in self.brain.semantic.knowledge_base:
                if existing['text'] == fact_text:
                    is_dup = True
                    break

            if not is_dup:
                self.brain.semantic.add_fact(fact_text, concept_ids, weight=1.0)
                for cid in concept_ids:
                    if cid not in self.brain.concept_index:
                        self.brain.concept_index[cid] = []
                    self.brain.concept_index[cid].append(kb_idx)
                facts_added += 1

            # 2. learn_from_text(store_in_kb=False) ile eğit
            result = self.brain.learn_from_text(
                text=fact_text,
                vocab=self.vocab,
                intent_report=intent_report,
                learning_rate=0.02,
                reward=1.0,
                store_in_kb=False,
            )
            total_words += result.get('words_learned', 0)

        # Save brain
        try:
            brain_path = getattr(self.config, 'MX_KNOWLEDGE_PATH', './mergen_knowledge.mx')
            self.brain.save(brain_path)
        except Exception as e:
            if self.verbose:
                print(f"[Mergen] Matematik kaydetme hatası: {e}")

        # Start reflection
        self._start_reflection("matematik")

        return (
            f"Matematik eğitimi tamamlandı (Tier: {tier}, Difficulty: {difficulty}).\n"
            f"  • Eğitilen problem:    {len(problems)}\n"
            f"  • Eklenen yeni fact:   {facts_added}\n"
            f"  • Toplam fact sayısı:  {self.brain.knowledge_size()}\n"
            f"  • Öğrenilen kelime:    {total_words}"
        )

    def _run_dream_consolidation(self, cycles: int = 100) -> str:
        """Run a bounded Dream consolidation cycle and reload Limbic state."""
        mx_path = Path(getattr(self.config, 'MX_WEIGHTS_PATH', './mergen_weights.mx'))
        had_mx_before = mx_path.exists()
        limbic = getattr(self, 'limbic', None)
        has_limbic = limbic is not None

        try:
            from cognitive.dream import MergenDream

            if has_limbic:
                saved = limbic.save_state()
                if not saved:
                    return "[Mergen] Dream konsolidasyon hatasi: Limbic state kaydedilemedi."

            dream = MergenDream(
                config_path="config.py",
                verbose=self.verbose,
                visualize=False,
            )

            # Faz 4: Uyku öncesi Episodik anıları Semantik'e konsolide et
            if hasattr(dream, 'consolidate_episodes'):
                dream.consolidate_episodes(self.brain)

            # NEW-02 FIX: dream.sleep() bloklaması interactive loop'u donduruyordu.
            # Daemon thread'de çalıştır; kısa bekleme sonrası arka plana al.
            _dream_exc = []
            def _sleep_worker():
                try:
                    dream.sleep(cycles=cycles)
                except Exception as _ex:
                    _dream_exc.append(_ex)

            _dream_thread = threading.Thread(target=_sleep_worker, daemon=True)
            _dream_thread.start()
            # Kullanıcı 60 saniyeye kadar bekliyorsa senkron tamamla, sonra arka plan.
            _dream_thread.join(timeout=60.0)
            dream_still_running = _dream_thread.is_alive()

            if _dream_exc:
                return f"[Mergen] Dream konsolidasyon hatasi (thread): {_dream_exc[0]}"

            reloaded = False
            if not dream_still_running and has_limbic:
                reloaded = limbic.load_state()

            stats = dream.dream_stats
            dream_log_path = dream.config.get('DREAM_LOG_PATH', './dream_log.npz')
            lines = [
                "[Mergen] Dream konsolidasyon tamamlandi." if not dream_still_running
                else "[Mergen] Dream konsolidasyon arka planda devam ediyor...",
                f"  Cycle:          {cycles}",
                f"  .mx yolu:       {mx_path}",
                f"  NREM cycle:     {stats.get('nrem_cycles', 0)}",
                f"  REM cycle:      {stats.get('rem_cycles', 0)}",
                f"  Pruned synapse: {stats.get('pruned_synapses', 0)}",
                f"  Dream log:      {dream_log_path}",
            ]

            if dream_still_running:
                lines.append("  Not: 60 sn. asıldi; dream arka planda bitiyor.")
            if not had_mx_before:
                lines.append("  Not: .mx yoktu; Dream taze agirlik matrisi baslatti.")
            if not has_limbic:
                lines.append("  Not: Limbic aktif degil, sadece .mx konsolidasyonu yapildi.")
            elif not reloaded and not dream_still_running:
                lines.append("  Uyari: Dream sonrasi Limbic state yeniden yuklenemedi.")

            return "\n".join(lines)
        except Exception as e:
            return f"[Mergen] Dream konsolidasyon hatasi: {e}"


    # ─────────────────────────────────────────────────────────
    #  COMMANDS
    # ─────────────────────────────────────────────────────────

    def handle_command(self, cmd: str) -> Optional[str]:
        """Handle special commands."""
        cmd = cmd.strip()
        cmd_lower = cmd.lower()

        # Exit
        if cmd_lower in ('/exit', '/quit', '/çık', '/cik', 'exit', 'quit', 'çık'):
            self.is_running = False
            return "[Mergen] Kapatılıyor..."

        # Stats
        if cmd_lower in ('/stats', 'istatistik'):
            return self._format_stats()

        # Introspection
        if cmd_lower in ('/introspect', '/içebakış'):
            return self._format_introspect()

        # Clear memory
        if cmd_lower == '/clear':
            self.conv_memory.clear()
            return "[Mergen] Konuşma hafızası temizlendi."

        # Dream: manual bounded consolidation
        if (
            cmd_lower == 'dream:run'
            or cmd_lower == 'dream:uyku'
            or cmd_lower.startswith('dream:run ')
            or cmd_lower.startswith('dream:uyku ')
        ):
            parts = cmd_lower.split(maxsplit=1)
            if len(parts) == 1:
                cycles = 100
            else:
                try:
                    cycles = int(parts[1].strip())
                except ValueError:
                    return "[Mergen] Kullanim: dream:run [1-1000] veya dream:uyku [1-1000]"
            if not 1 <= cycles <= 1000:
                return "[Mergen] Kullanim: dream:run [1-1000] veya dream:uyku [1-1000]"
            return self._run_dream_consolidation(cycles=cycles)

        # File ingestion
        if cmd_lower.startswith('oku:') or cmd_lower.startswith('read:'):
            filepath = cmd.split(':', 1)[1].strip()
            return self.ingest_file(filepath)

        # Matematik Eğitimi
        if cmd_lower.startswith('matematik:egit') or cmd_lower.startswith('matematik:eğit'):
            parts = cmd_lower.split()
            tier = 0
            difficulty = 0
            if len(parts) > 1:
                try:
                    tier = int(parts[1])
                except ValueError:
                    pass
            if len(parts) > 2:
                try:
                    difficulty = int(parts[2])
                except ValueError:
                    pass
            return self._train_math(tier, difficulty)

        # RAG: veri yükle
        if cmd_lower in ('rag:yukle', 'rag:yükle', 'rag:setup', 'rag:load'):
            if self.rag is None or self._rag_loader is None:
                return (
                    "[Mergen] RAG motoru aktif değil.\n"
                    "Kurulum: pip install chromadb"
                )
            print("[Mergen] Türkçe veri indiriliyor, bu biraz sürebilir...")
            print("[Mergen] NOT: Bu komut GitHub API'den indirir. Alternatif olarak")
            print("[Mergen]      'oku:dosya.txt' komutuyla yerel dosya da yükleyebilirsin.")
            try:
                results = self._rag_loader.load_all()
            except Exception as _e:
                return f"[Mergen] RAG yükleme hatası: {_e}"
            if not results:
                return (
                    f"[Mergen] Veri yüklenemedi.\n"
                    f"  Olası nedenler:\n"
                    f"  • GitHub API rate-limit (60 istek/saat — anonim)\n"
                    f"  • İnternet bağlantısı\n"
                    f"  Şu an veritabanında: {self.rag.count()} kayıt\n"
                    f"  Alternatif: 'oku:dosya.txt' ile yerel dosya yükle."
                )
            lines = ["[Mergen] RAG veri yükleme tamamlandı:"]
            for src, cnt in results.items():
                lines.append(f"  • {src}: {cnt} kayıt")
            lines.append(f"  • Toplam veritabanı: {self.rag.count()} kayıt")
            return "\n".join(lines)

        # RAG: durum
        if cmd_lower in ('rag:durum', 'rag:status', 'rag:bilgi'):
            if self.rag is None:
                return "[Mergen] RAG motoru aktif değil."
            status = self._rag_loader.status() if self._rag_loader else {}
            lines = [f"RAG Durumu — {self.rag.count()} toplam kayıt:"]
            for key, info in status.items():
                if key == "toplam_kayit":
                    continue
                durum = "✓ yüklü" if info.get("indexed") else "✗ yüklenmedi"
                lines.append(f"  {durum}  {info['label']}")
            return "\n".join(lines)

        # RAG: manuel arama
        if cmd_lower.startswith('rag:ara ') or cmd_lower.startswith('rag:search '):
            if self.rag is None or not self.rag.ready:
                return "[Mergen] RAG motoru aktif değil."
            query_part = cmd.split(' ', 1)[1].strip()
            if not query_part:
                return "[Mergen] Kullanım: rag:ara <sorgu>"
            hits = self.rag.search(query_part, top_k=5)
            if not hits:
                return f"[Mergen] '{query_part}' için RAG'da sonuç bulunamadı. (Kayıt sayısı: {self.rag.count()})"
            lines = [f"RAG sonuçları '{query_part}':"]
            for i, h in enumerate(hits, 1):
                lines.append(f"  [{i}] ({h['relevance']:.3f}) [{h['source']}] {h['text'][:120]}")
            return "\n".join(lines)

        # Help
        if cmd_lower in ('/help', 'yardım'):
            return (
                "Komutlar:\n"
                "  /stats              — İstatistikler\n"
                "  /introspect         — Öz model\n"
                "  /clear              — Hafızayı temizle\n"
                "  /exit               — Kapat\n"
                "  dream:run [cycles]  - Manuel Dream konsolidasyonu (1-1000)\n"
                "  dream:uyku [cycles] - Manuel Dream konsolidasyonu (1-1000)\n"
                "  oku:dosya.txt       — Dosya öğren (RAG'a da yazar)\n"
                "  rag:yukle           — GitHub'dan Türkçe veri indir & indeksle\n"
                "  rag:durum           — RAG veritabanı durumu\n"
                "  rag:ara <sorgu>     — RAG veritabanında manuel arama\n"
                "  /help               — Yardım"
            )

        return None

    def _format_stats(self) -> str:
        uptime = time.time() - self._start_time

        try:
            import torch
            weight_mean = self.brain.mx2.weight.abs().mean().item()
            weight_max = self.brain.mx2.weight.abs().max().item()
            trace_energy = self.brain.hebbian_trace.abs().sum().item()
            active_neurons = (self.brain.hebbian_trace.abs() > 0.01).sum().item()
        except Exception as e:
            if self.verbose:
                print(f"[Mergen] Stats telemetry fallback used: {e}")
            weight_mean = weight_max = trace_energy = 0.0
            active_neurons = 0

        rag_count = self.rag.count() if self.rag else 0

        lines = [
            "━━━ MERGEN DIGITAL BRAIN v7.0 ━━━",
            f"  Uptime:             {uptime:.1f}s",
            f"  Vocabulary:         {self.vocab.size()} concepts",
            f"  Brain steps:        {self.brain.step_count}",
            f"  Interactions:       {len(self.interaction_log)}",
            f"  Conversation turns: {self.conv_memory.total_turns}",
            "",
            "━━━ LEARNING STATE ━━━",
            f"  Mean weight:        {weight_mean:.4f}",
            f"  Max weight:         {weight_max:.4f}",
            f"  Trace energy:       {trace_energy:.4f}",
            f"  Active neurons:     {active_neurons}/{self.vocab.size()}",
            f"  Learned facts:      {self.brain.knowledge_size()}",
            f"  RAG kayıtları:      {rag_count}",
        ]

        # Add Neuromodulation Stats
        if hasattr(self, 'limbic') and hasattr(self.limbic, 'neuro'):
            nl = self.limbic.neuro.get_levels()
            lines.extend([
                "",
                "━━━ NEUROMODULATION ━━━",
                f"  Dopamine (DA):      {nl.get('DA', 0):.4f}",
                f"  Serotonin (5-HT):   {nl.get('5-HT', 0):.4f}",
                f"  Noradrenaline (NE): {nl.get('NE', 0):.4f}",
                f"  Acetylcholine (ACh):{nl.get('ACh', 0):.4f}",
            ])

        tele = self.analyzer.get_telemetry()
        lines.extend([
            "",
            "━━━ INTENT MEMORY ━━━",
            f"  Total analyses:     {tele.get('lifetime_analyses', 0)}",
            f"  Avg confidence:     {tele.get('avg_confidence', 0):.3f}",
        ])

        intents = tele.get('intent_distribution', {})
        if intents:
            lines.append("  Intent distribution:")
            for k, v in sorted(intents.items(), key=lambda x: -x[1]):
                lines.append(f"    {k:12s}: {v}")

        conv_tele = self.conv_memory.get_telemetry()
        if conv_tele.get('current_topics'):
            lines.append(f"  Current topics: {', '.join(conv_tele['current_topics'][:3])}")

        return "\n".join(lines)

    def _format_introspect(self) -> str:
        lines = [
            "━━━ MERGEN SELF-MODEL ━━━",
            f"  I am Mergen — a biological cognitive AI.",
            f"  My vocabulary: {self.vocab.size()} concepts.",
            f"  My knowledge base: {self.brain.knowledge_size()} facts.",
            f"  My conversation turns: {self.conv_memory.total_turns}.",
            f"  My last intent: {self.analyzer.last_intent}.",
            f"  My last subject: {self.analyzer.last_subject}.",
        ]

        topics = self.conv_memory.current_topics
        if topics:
            lines.append(f"  Current topics: {', '.join(topics[:3])}")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────
    #  KNOWLEDGE EXTRACTION & SUMMARY FACTS
    # ─────────────────────────────────────────────────────────

    def _extract_key_concepts(self, content: str, source: str) -> List[str]:
        """Dosya içeriğinden ana kavramları çıkar."""
        import re
        words = re.findall(r'\w+', content.lower())
        
        # Stop words
        _STOP = {
            'bir', 've', 'ile', 'bu', 'şu', 'de', 'da', 'ki', 'mi', 'mı', 'mu', 'mü',
            'the', 'a', 'an', 'is', 'are', 'was', 'in', 'of', 'to', 'and', 'or', 'but',
            'için', 'gibi', 'kadar', 'olarak', 'olan', 'var', 'yok', 'daha', 'en',
            'çok', 'az', 'ben', 'sen', 'o', 'biz', 'siz', 'onlar',
            'ne', 'nasıl', 'neden', 'niye', 'nerede', 'kim', 'hangi',
        }
        
        # Add to semantic memory
        fact_text = f"{source} dosyasından veri yüklendi."
        concept_ids = []
        for w in set(words):
            if len(w) > 3 and w not in _STOP and self.vocab.contains(w):
                concept_ids.append(self.vocab.get_id(w))
        
        if concept_ids:
            kb_idx = self.brain.semantic.add_fact(fact_text, concept_ids, weight=1.0)
            for cid in concept_ids:
                if cid not in self.brain.concept_index:
                    self.brain.concept_index[cid] = []
                self.brain.concept_index[cid].append(kb_idx)
        
        # Count word frequency
        word_counts = Counter()
        for w in words:
            if len(w) > 3 and w not in _STOP:
                word_counts[w] += 1
        
        # Get top concepts (most frequent meaningful words)
        concepts = [word for word, count in word_counts.most_common(10)]
        return concepts

    def _add_summary_fact(self, source: str, concept: str):
        """Özet fact ekle - Mergen'in kendi cümlesini kurabilmesi için."""
        import re
        
        # Check if concept already has facts
        existing = []
        for fact in self.brain.semantic.knowledge_base:
            text = fact.get('text', '').lower()
            if concept in text:
                existing.append(text)
        
        # Create summary facts that help Mergen GENERATE its own sentences
        summary_templates = [
            f"{concept} {source} dosyasında öğrenilen bir kavramdır.",
            f"{source} konusunda {concept} hakkında bilgiler öğrendim.",
            f"{concept} konusu hakkında hafızamda bilgiler var.",
        ]
        
        for template in summary_templates:
            tokens = re.findall(r'\w+', template.lower())
            matched_ids = []
            for tok in tokens:
                if self.vocab.contains(tok):
                    matched_ids.append(self.vocab.get_id(tok))
            
            # Avoid duplicates
            is_dup = False
            for fact in self.brain.semantic.knowledge_base + self.brain.episodic.events:
                if template.lower() in fact.get('text', '').lower():
                    is_dup = True
                    break
            
            if not is_dup:
                kb_idx = self.brain.semantic.add_fact(template, list(set(matched_ids)), weight=1.5)
                for cid in set(matched_ids):
                    if cid not in self.brain.concept_index:
                        self.brain.concept_index[cid] = []
                    self.brain.concept_index[cid].append(kb_idx)

    # ─────────────────────────────────────────────────────────
    #  BACKGROUND REFLECTION
    # ─────────────────────────────────────────────────────────

    def _reflect_in_background(self, source_name: str):
        """Strengthen Hebbian connections between co-occurring concepts."""
        try:
            kb = self.brain.semantic.knowledge_base
            if len(kb) < 2:
                return

            _STOP = {'bir', 've', 'ile', 'bu', 'de', 'da', 'mi', 'mı',
                     'the', 'a', 'an', 'is', 'in', 'of', 'to'}

            def _tok(t):
                return {w for w in re.findall(r'\w+', t.lower())
                        if len(w) > 3 and w not in _STOP}

            # Son eklenen fact'leri tara (sliding window) — sabit ilk-50 yerine.
            consolidated = 0
            n_recent = min(len(kb), 200)
            start_idx = max(0, len(kb) - n_recent)

            # NEW-01 FIX: hebbian_trace yazmaları _hebbian_lock ile korunuyor.
            # BUG-02'de eklenen merkezi lock; bu path de aynı trace'e yazıyor.
            with self._hebbian_lock:
                for i in range(start_idx, len(kb)):
                    tok_i = _tok(kb[i]['text'])
                    for j in range(i + 1, len(kb)):
                        tok_j = _tok(kb[j]['text'])
                        shared = tok_i & tok_j
                        if len(shared) >= 2:
                            for word in shared:
                                if self.vocab.contains(word):
                                    wid = self.vocab.get_id(word)
                                    if 0 <= wid < self.vocab.size():
                                        self.brain.hebbian_trace[wid] += 0.05
                            consolidated += 1

                max_trace = self.brain.hebbian_trace.max().item()
                if max_trace > 5.0:
                    self.brain.hebbian_trace.div_(max_trace / 5.0)

            if self.verbose:
                print(f"\n[Mergen 🧠] Refleksiyon: {consolidated} bağ güçlendirildi ({source_name})")

        except Exception as e:
            if self.verbose:
                print(f"[Mergen] Refleksiyon hatası: {e}")
        finally:
            self._reflection_done.set()


    def _start_reflection(self, source_name: str):
        self._reflection_done.clear()
        self._reflection_thread = threading.Thread(
            target=self._reflect_in_background,
            args=(source_name,),
            daemon=True,
        )
        self._reflection_thread.start()
        if self.verbose:
            print(f"[Mergen 🧠] Arka planda sindiriyorum...")

    def save(self) -> bool:
        """Explicit save delegation for external training scripts."""
        try:
            brain_path = getattr(self.config, 'MX_KNOWLEDGE_PATH', './mergen_knowledge.mx')
            return self.brain.save(brain_path)
        except Exception as e:
            if self.verbose:
                print(f"[Mergen] Explicit save failed: {e}")
            return False

    # ─────────────────────────────────────────────────────────
    #  SHUTDOWN
    # ─────────────────────────────────────────────────────────

    def _signal_shutdown(self, signum, frame):
        print("\n[Mergen] Kapatılıyor...")
        self.shutdown()
        sys.exit(0)

    def shutdown(self):
        """Persist everything."""
        if self.verbose:
            print("[Mergen] Kaydediliyor...")

        if hasattr(self, 'limbic') and self.limbic is not None:
            self.limbic.shutdown()

        # Vocab
        try:
            vocab_path = getattr(self.config, 'VOCAB_SAVE_PATH', './mergen_vocab.json')
            self.vocab.save(vocab_path)
        except Exception as e:
            print(f"  ⚠ Vocab save: {e}")

        # Brain weights
        try:
            brain_path = getattr(self.config, 'MX_KNOWLEDGE_PATH', './mergen_knowledge.mx')
            self.brain.save(brain_path)
        except Exception as e:
            print(f"  ⚠ Brain save: {e}")

        # Intent memory
        try:
            self.analyzer._save_memory()
        except Exception as e:
            print(f"  ⚠ MX save: {e}")

        # Conversation memory (auto-saves)

        # Interaction log
        try:
            log_path = Path('./mergen_interactions.json')
            log_path.write_text(
                json.dumps(self.interaction_log, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
        except Exception as e:
            print(f"  ⚠ Log save: {e}")

        if self.verbose:
            print("[Mergen] Görüşürüz. 🌙\n")

    # ─────────────────────────────────────────────────────────
    #  INTERACTIVE LOOP
    # ─────────────────────────────────────────────────────────

    def run(self):
        """Main interactive loop."""
        print("╔" + "═" * 63 + "╗")
        print("║  MERGEN — Digital Brain v7.0                               ║")
        print("║  Developed by Vertex Corporation                           ║")
        print("║                                                            ║")
        print("║  Commands: /stats, /introspect, /clear, /exit, oku:.txt    ║")
        print("║  Help: /help                                               ║")
        print("╚" + "═" * 63 + "╝\n")

        if hasattr(self, 'limbic') and self.limbic is not None:
            self.limbic.wake_up()

        while self.is_running:
            try:
                user_input = input("Sen > ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                break

            if not user_input:
                continue

            # Command handling
            cmd_response = self.handle_command(user_input)
            if cmd_response is not None:
                print(f"\nMergen > {cmd_response}\n")
                continue

            # Normal conversation
            start_inf = time.time()
            response = self.respond(user_input)
            latency_ms = (time.time() - start_inf) * 1000
            print(f"\nMergen > {response}")
            print(f"  [Gecikme (Latency): {latency_ms:.1f} ms]\n")

        self.shutdown()


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    mergen = MergenBrain_v7(verbose=True)
    mergen.run()


if __name__ == "__main__":
    main()
