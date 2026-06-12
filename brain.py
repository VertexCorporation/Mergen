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
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from collections import Counter

# ── Core modules ──
try:
    from mergen_vocab import MergenVocab
except ImportError as e:
    print(f"[Mergen] ✗ Cannot import MergenVocab: {e}")
    sys.exit(1)

try:
    from intent_analyzer import IntentAnalyzer
except ImportError as e:
    print(f"[Mergen] ✗ Cannot import IntentAnalyzer: {e}")
    sys.exit(1)

try:
    from broca_area import BrocaArea, MergenBrain, MergenConfig
except ImportError as e:
    print(f"[Mergen] ✗ Cannot import broca_area: {e}")
    sys.exit(1)

# ── New brain integration modules ──
from conversation_memory import ConversationMemory
from response_generator import ResponseGenerator
from mergen_brain_wrapper import EnhancedMergenBrain

# ── RAG + Biyolojik bileşenler (opsiyonel) ──
try:
    from rag_engine import RAGEngine
    from data_loader import TurkishDataLoader
    from hebbian_rag_bridge import HebbianRAGBridge
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False

try:
    from turkish_morph import TurkishMorph
    _MORPH_AVAILABLE = True
except ImportError:
    _MORPH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
#  MERGEN — Full Digital Brain Orchestrator
# ═══════════════════════════════════════════════════════════════════

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

        # 3. MergenBrain (neural core)
        self.brain = MergenBrain(
            vocab_size=self.vocab.size(),
            config=self.config,
        )

        # Load previous weights
        brain_path = getattr(self.config, 'MX_WEIGHTS_PATH', './mergen_weights.mx')
        if Path(brain_path).exists():
            self.brain.load(brain_path)
            if verbose:
                print(f"[Mergen] Loaded brain weights")

        # 4. Enhanced Brain (Wernicke integration)
        self.enhanced_brain = EnhancedMergenBrain(
            brain=self.brain,
            config=self.config,
            use_wernicke=True,
            device=getattr(self.config, 'DEVICE', 'cpu'),
        )

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

        # 7. Response Generator (Generative, not Extractive)
        self.generator = ResponseGenerator(
            vocab=self.vocab,
            brain=self.brain,
        )

        # 8. Broca Area (fallback expression)
        self.broca = self._init_broca()

        # 9. Türkçe Morfoloji (UTF-8 + Zeyrek)
        self.morph: Optional[Any] = None
        if _MORPH_AVAILABLE:
            try:
                self.morph = TurkishMorph(verbose=verbose)
            except Exception:
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
                    # Hebbian-RAG köprüsü
                    self._hebb_bridge = HebbianRAGBridge(
                        brain=self.brain,
                        vocab=self.vocab,
                        verbose=verbose,
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
            from language_engine import LanguageEngine
            lang_engine = LanguageEngine(
                motor_layer_size=min(self.vocab.size() * 4, 10_000),
                motor_rows=40,
                motor_cols=50,
                vocabulary=self.vocab.all_words,
                temperature=getattr(self.config, 'TEMPERATURE', 0.9),
                top_k=getattr(self.config, 'TOP_K', 40),
                device=getattr(self.config, 'DEVICE', 'cpu'),
            )
            return BrocaArea(
                language_engine=lang_engine,
                n_neurons=self.vocab.size(),
                concept_vocabulary=self.vocab.all_words,
                device=getattr(self.config, 'DEVICE', 'cpu'),
            )
        except (ImportError, TypeError) as e:
            if self.verbose:
                print(f"[Mergen] ⚠ LanguageEngine unavailable ({e})")

            class StubEngine:
                def __init__(self, vocab_words):
                    self.vocabulary = vocab_words
                def speak(self, *args, **kwargs):
                    import random
                    return random.choice(self.vocabulary[:50])
                def strengthen_association(self, *args, **kwargs):
                    pass

            return BrocaArea(
                language_engine=StubEngine(self.vocab.all_words),
                n_neurons=self.vocab.size(),
                concept_vocabulary=self.vocab.all_words,
            )

    # ─────────────────────────────────────────────────────────
    #  MAIN PIPELINE: perceive → think → respond
    # ─────────────────────────────────────────────────────────

    def respond(self, user_input: str) -> str:
        """
        Main conversational pipeline:
          input → resolve references → analyze intent → process brain
          → recall knowledge → synthesize response → learn → store context
        """
        if not user_input or not user_input.strip():
            return ""

        # Step 1: Resolve pronouns using conversation context
        resolved_input = self.conv_memory.resolve_references(user_input)

        # Step 2: Intent analysis
        try:
            report = self.analyzer.analyze_intent(resolved_input)
        except Exception as e:
            print(f"[Mergen] ⚠ Intent error: {e}")
            report = {
                'primary_intent': 'UNKNOWN',
                'confidence_score': 0.0,
                'sentiment': {'sentiment_score': 0.0, 'excitement': 0.0},
                'subject': None,
            }

        intent = report.get('primary_intent', 'UNKNOWN')
        subject = report.get('subject')

        # Step 3: Brain processing (neural activation)
        try:
            brain_output = self.enhanced_brain.process_with_intent(
                text=resolved_input,
                intent_report=report,
            )
            neural_intent = brain_output.get('neural_intent')
        except Exception as e:
            print(f"[Mergen] ⚠ Brain error: {e}")
            import torch
            neural_intent = torch.zeros(self.vocab.size())

        # Step 4: Passive learning from input
        try:
            self.brain.learn_from_text(
                text=resolved_input,
                vocab=self.vocab,
                intent_report=report,
                learning_rate=0.005,
                reward=0.5,
            )
        except Exception:
            pass

        # Step 4b: ACTIVE LEARNING — extract facts from user input
        learned_fact = self._try_learn_from_input(resolved_input, intent)

        # If we just learned something, acknowledge it
        if learned_fact:
            response = random.choice([
                f"Bunu öğrendim: {learned_fact}",
                f"Bunu not ettim: {learned_fact}",
                f"Bunu hafızama ekledim: {learned_fact}",
                f"Öğrendim! {learned_fact}",
            ])
        else:
            # Step 5: Knowledge recall — multi-strategy
            knowledge_facts = self._recall_knowledge(resolved_input, intent, subject)

            # Step 6: Get conversation context
            conv_context = self.conv_memory.get_context_summary(max_turns=5)

            # Step 7: Generate response (Generative)
            response = self.generator.generate(
                query=resolved_input,
                intent=intent,
                subject=subject,
                knowledge_facts=knowledge_facts,
                conversation_context=conv_context,
            )

            # If generator failed, try Broca fallback
            if not response or len(response) < 3:
                try:
                    response = self._broca_generate(neural_intent, resolved_input, report)
                except Exception:
                    response = "Anlayamadım, tekrar eder misin?"

        # Step 8: Store in conversation memory
        self.conv_memory.add_turn(
            user_input=user_input,
            response=response,
            intent=intent,
            subject=subject,
        )

        # Step 9: Log interaction
        self.interaction_log.append({
            'timestamp': time.time(),
            'input': user_input[:200],
            'intent': intent,
            'confidence': report.get('confidence_score'),
            'response': response[:300],
        })

        return response

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
            r'^(merhaba|selam|hey|sa|as|selamlar|iyi akşamlar|günaydın|iyi geceler)',
            r'^(teşekkür|sağol|eyvallah|tamam|evet|hayır|peki|olur|tamam mı)',
            r'^(nasılsın|naber|ne yapıyorsun|iyi misin)',
            r'^(görüşürüz|hoşçakal|bye|güle güle)',
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
        for existing in self.brain.knowledge_base:
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
        kb_idx = len(self.brain.knowledge_base)
        self.brain.knowledge_base.append({
            'text': fact_text,
            'concept_ids': list(set(matched_ids)),
            'weight': 1.0,
            'access_count': 0,
        })
        for cid in set(matched_ids):
            if cid not in self.brain.concept_index:
                self.brain.concept_index[cid] = []
            self.brain.concept_index[cid].append(kb_idx)

        # Save to disk
        try:
            brain_path = getattr(self.config, 'MX_WEIGHTS_PATH', './mergen_weights.mx')
            self.brain.save(brain_path)
        except Exception:
            pass

        if self.verbose:
            print(f"[Mergen 📚] Öğrendim: {fact_text[:80]}")

        return fact_text

    def _recall_knowledge(
        self,
        query: str,
        intent: str,
        subject: Optional[str],
    ) -> list:
        """
        Multi-strategy knowledge recall with intelligent merging.
        """
        all_facts = []
        seen_texts = set()

        query_lower = query.lower()
        is_def_query = bool(re.search(
            r'\bnedir\b|\bne demek\b|\bne anlama\b|\bwhat is\b|\bwhat are\b|\bkimdir\b',
            query_lower
        ))

        # Strategy 1: Subject-based recall (for definition queries)
        if subject:
            try:
                facts = self.enhanced_brain.recall_all_about(subject, top_k=8)
                for f in facts:
                    key = f['text'][:80].lower()
                    if key not in seen_texts:
                        seen_texts.add(key)
                        all_facts.append(f)
            except Exception:
                pass

        # Strategy 2: Raw text recall from full query
        try:
            facts = self.enhanced_brain.recall_raw(query, top_k=8)
            for f in facts:
                key = f['text'][:80].lower()
                if key not in seen_texts:
                    seen_texts.add(key)
                    all_facts.append(f)
        except Exception:
            pass

        # Strategy 3: Also search with subject-only (broader match)
        if subject and not is_def_query:
            try:
                facts = self.enhanced_brain.recall_raw(subject, top_k=5)
                for f in facts:
                    key = f['text'][:80].lower()
                    if key not in seen_texts:
                        seen_texts.add(key)
                        all_facts.append(f)
            except Exception:
                pass

        # Strategy 4: Semantic similarity (Wernicke)
        try:
            facts = self.enhanced_brain.recall_semantic(query, top_k=5)
            for f in facts:
                key = f['text'][:80].lower()
                if key not in seen_texts:
                    seen_texts.add(key)
                    all_facts.append(f)
        except Exception:
            pass

        # Strategy 5: RAG — BioVectorizer + HTM biyolojik arama
        if self.rag and self.rag.ready:
            try:
                # Türkçe morfoloji ile sorgu normalizasyonu
                if self.morph:
                    _, concepts = self.morph.normalize_query(query)
                    # Önce kök kavramlarla ara, sonra ham soru ile de dene
                    rag_queries = []
                    if concepts:
                        rag_queries.append(" ".join(concepts))
                    if subject:
                        rag_queries.append(self.morph.lemmatize(subject))
                    rag_queries.append(query)
                else:
                    rag_queries = [subject, query] if subject else [query]

                for rq in rag_queries:
                    if not rq or not rq.strip():
                        continue
                    rag_hits = self.rag.search(rq, top_k=5)
                    for hit in rag_hits:
                        key = hit['text'][:80].lower()
                        if key not in seen_texts:
                            seen_texts.add(key)
                            all_facts.append({
                                'text':      hit['text'],
                                'relevance': hit['relevance'] * 0.9,
                                'source':    hit.get('source', 'rag'),
                            })
            except Exception:
                pass

        # Sort by relevance and return
        all_facts.sort(key=lambda x: -x.get('relevance', 0))
        return all_facts[:8]

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
                        except Exception:
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

        # Learn paragraphs (moderate reward) — primary knowledge units
        for para in paragraphs[:60]:
            self.brain.learn_from_text(
                text=para,
                vocab=self.vocab,
                intent_report=base_report,
                learning_rate=0.02,
                reward=1.5,
            )

        # Learn individual sentences (lower reward) — granular facts
        for sent in sentences[:100]:
            self.brain.learn_from_text(
                text=sent,
                vocab=self.vocab,
                intent_report=base_report,
                learning_rate=0.01,
                reward=0.8,
            )

        # Hebbian learning from full content (for weight updates, NOT KB storage)
        self.brain.learn_from_text(
            text=content.strip()[:3000],
            vocab=self.vocab,
            intent_report=base_report,
            learning_rate=0.01,
            reward=0.5,
            store_in_kb=False,
        )
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
            except Exception:
                continue

        # IMPORTANT: Create "summary facts" that Mergen can use to generate its OWN sentences
        # Extract key concepts from the file content
        key_concepts = self._extract_key_concepts(content, filename_stem)
        for concept in key_concepts:
            self._add_summary_fact(filename_stem, concept)

        # Hebbian-RAG köprüsü: dosyadan gelen cümleler sinaptik izleri günceller
        if self._hebb_bridge is not None:
            self._hebb_bridge.update_from_batch(
                sentences[:150], source=filename_stem, reward=1.0
            )

        # Save
        try:
            brain_path = getattr(self.config, 'MX_WEIGHTS_PATH', './mergen_weights.mx')
            self.brain.save(brain_path)
        except Exception:
            pass

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

        # File ingestion
        if cmd_lower.startswith('oku:') or cmd_lower.startswith('read:'):
            filepath = cmd.split(':', 1)[1].strip()
            return self.ingest_file(filepath)

        # RAG: veri yükle
        if cmd_lower in ('rag:yukle', 'rag:yükle', 'rag:setup', 'rag:load'):
            if self.rag is None or self._rag_loader is None:
                return (
                    "[Mergen] RAG motoru aktif değil.\n"
                    "Kurulum: pip install chromadb sentence-transformers"
                )
            print("[Mergen] Türkçe veri indiriliyor, bu biraz sürebilir...")
            results = self._rag_loader.load_all()
            lines = [f"[Mergen] RAG veri yükleme tamamlandı:"]
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

        # Help
        if cmd_lower in ('/help', 'yardım'):
            return (
                "Komutlar:\n"
                "  /stats         — İstatistikler\n"
                "  /introspect    — Öz model\n"
                "  /clear         — Hafızayı temizle\n"
                "  /exit          — Kapat\n"
                "  oku:dosya.txt  — Dosya öğren\n"
                "  rag:yukle      — Türkçe veri indir & indeksle\n"
                "  rag:durum      — RAG veritabanı durumu\n"
                "  /help          — Yardım"
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
        except Exception:
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
        for fact in self.brain.knowledge_base:
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
            for fact in self.brain.knowledge_base:
                if template.lower() in fact.get('text', '').lower():
                    is_dup = True
                    break
            
            if not is_dup:
                kb_idx = len(self.brain.knowledge_base)
                self.brain.knowledge_base.append({
                    'text': template,
                    'concept_ids': list(set(matched_ids)),
                    'weight': 1.5,  # Higher weight for summary facts
                    'access_count': 0,
                })
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
            kb = self.brain.knowledge_base
            if len(kb) < 2:
                return

            _STOP = {'bir', 've', 'ile', 'bu', 'de', 'da', 'mi', 'mı',
                     'the', 'a', 'an', 'is', 'in', 'of', 'to'}

            def _tok(t):
                return {w for w in re.findall(r'\w+', t.lower())
                        if len(w) > 3 and w not in _STOP}

            consolidated = 0
            for i in range(min(len(kb), 50)):
                tok_i = _tok(kb[i]['text'])
                for j in range(i + 1, min(len(kb), 50)):
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

        # Vocab
        try:
            vocab_path = getattr(self.config, 'VOCAB_SAVE_PATH', './mergen_vocab.json')
            self.vocab.save(vocab_path)
        except Exception as e:
            print(f"  ⚠ Vocab save: {e}")

        # Brain weights
        try:
            brain_path = getattr(self.config, 'MX_WEIGHTS_PATH', './mergen_weights.mx')
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
            response = self.respond(user_input)
            print(f"\nMergen > {response}\n")

        self.shutdown()


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    mergen = MergenBrain_v7(verbose=True)
    mergen.run()


if __name__ == "__main__":
    main()
