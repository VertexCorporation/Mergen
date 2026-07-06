"""MERGEN - deterministic response synthesis.

This module does not behave like an LLM. It receives ranked facts/candidates
from brain.py and turns the top evidence into short, inspectable answers.
"""

import re
import random
from typing import Optional, Dict, List


class ResponseGenerator:
    """Controlled answer generator for Mergen's KB/RAG evidence."""

    CONVERSATIONAL = {
        'greeting': [
            'Merhaba, seni dinliyorum.',
            'Selam. Hangi konuya bakalim?',
        ],
        'wellbeing': [
            'Calisiyorum. Bellek, kavram ve geri cagirma katmanlarimi kullaniyorum.',
            'Iyiyim; yeni girdileri sinaptik izler ve bilgi tabaniyla iliskilendiriyorum.',
        ],
        'gratitude': [
            'Rica ederim.',
            'Devam edebiliriz.',
        ],
        'identity': [
            'Ben Mergen. Hebbian ogrenme, STDP, RAG, Limbic ve Dream katmanlariyla calisan biyolojiden ilhamli deneysel bir dijital beyinim.',
        ],
        'unknown': [
            'Bu konuda guvenilir bir bilgi bulamadim.',
            'Bu konu icin bilgi tabanimda yeterli kanit yok.',
        ],
    }

    STOP_SUBJECTS = {
        'bir', 've', 'ile', 'bu', 'su', 'o', 'de', 'da', 'mi', 'mi', 'mu', 'mu',
        'ne', 'nasil', 'neden', 'niye', 'nedir', 'kim', 'nerede', 'ne zaman',
        'the', 'a', 'an', 'is', 'in', 'of', 'to', 'and', 'or', 'but',
        'kur', 'ver', 'soyle', 'goster', 'yaz', 'cumle', 'ornek',
    }

    QUESTION_TERMS = {
        'ne', 'neyi', 'neye', 'nedir', 'neler', 'nasil', 'nasıl', 'neden',
        'niye', 'kim', 'kimdir', 'nerede', 'nereye', 'ne zaman', 'kac', 'kaç',
        'what', 'why', 'how', 'who', 'where', 'when',
    }

    META_TRAINING_TERMS = {
        'egitiminde', 'eğitiminde', 'egitim', 'eğitim', 'partisinden',
        'degerlendirilmelidir', 'değerlendirilmelidir', 'kademeli',
        'veri seti', 'veri-seti', 'curriculum',
    }

    GENERIC_PATTERNS = [
        r'^\w+.*?\b(?:isim|fiil|sifat|zarf|baglac|edat|zamir)dir',
        r'^masada\s+bir\s+',
        r'^bu\s+\w+\s+guzel',
        r'^gunluk\s+hayatta\s+\w+\s+kullanilir',
        r'^\w+\s+onemli',
        r'^her\s+gun\s+',
        r'^insanlar\s+',
        r'^hava\s+cok\s+',
    ]

    def __init__(self, vocab: object, brain: object):
        self.vocab = vocab
        self.brain = brain
        self.response_count = 0

    def generate(
        self,
        query: str,
        intent: str = 'UNKNOWN',
        subject: Optional[str] = None,
        knowledge_facts: Optional[List[Dict]] = None,
        conversation_context: Optional[Dict] = None,
    ) -> str:
        self.response_count += 1
        query_lower = (query or '').lower().strip()

        # ── ARITHMETIC QUERY RESOLVER ──
        # 1. Direct digit expression parsing (e.g., "3*4", "3 * 4", "3 x 4")
        math_match = re.search(r'(\d+)\s*([\+\-\*\/x])\s*(\d+)', query_lower)
        if math_match:
            a_val = int(math_match.group(1))
            op_sym = math_match.group(2)
            b_val = int(math_match.group(3))
            if op_sym == 'x':
                op_sym = '*'
            
            kb = getattr(self.brain, 'knowledge_base', []) or []
            for fact in kb:
                text = fact.get('text', '')
                parts = text.split()
                if "esittir" in parts:
                    try:
                        eq_idx = parts.index("esittir")
                        f_subject = parts[eq_idx+2]
                        f_a = int(parts[eq_idx+3])
                        f_b = int(parts[eq_idx+4])
                        f_res = int(parts[eq_idx+5])
                        
                        op_match = False
                        if op_sym == '+' and f_subject == 'toplama':
                            op_match = True
                        elif op_sym == '-' and f_subject == 'cikarma':
                            op_match = True
                        elif op_sym == '*' and f_subject == 'carpma':
                            op_match = True
                        elif op_sym == '/' and f_subject == 'bolme':
                            op_match = True
                            
                        if op_match and f_a == a_val and f_b == b_val:
                            return str(f_res)
                    except (ValueError, IndexError):
                        continue
            # If digit match but not found in KB, return controlled refusal
            return self.CONVERSATIONAL['unknown'][0]

        # 2. Verbal/Fact-based arithmetic parsing (e.g., "üç çarpı dört")
        if knowledge_facts:
            for fact in knowledge_facts:
                text = fact.get('text', '')
                parts = text.split()
                if "esittir" in parts:
                    try:
                        eq_idx = parts.index("esittir")
                        if len(parts) > eq_idx + 5:
                            f_subject = parts[eq_idx+2]
                            if f_subject in ('toplama', 'cikarma', 'carpma', 'bolme'):
                                op_keywords = {
                                    'toplama': {'+', 'arti', 'toplam', 'topla'},
                                    'cikarma': {'-', 'eksi', 'cikar', 'fark'},
                                    'carpma': {'*', 'x', 'carpi', 'carp', 'kere'},
                                    'bolme': {'/', 'bolu', 'bol', 'bolme'}
                                }[f_subject]
                                q_tokens_with_symbols = {self._ascii_fold(t) for t in re.findall(r'\w+|\+|\-|\*|\/', query_lower)}
                                if not (op_keywords & q_tokens_with_symbols):
                                    continue
                                q_tokens = {self._ascii_fold(t) for t in re.findall(r'\w+', query_lower)}
                                f_tokens_before_eq = {self._ascii_fold(t) for t in parts[:eq_idx]}
                                ignore_terms = {'arti', 'eksi', 'carpi', 'bolu', 'kac', 'eder', 'esit', 'esittir', 'toplama', 'cikarma', 'carpma', 'bolme'}
                                q_operands = {self._number_to_word(tok) for tok in q_tokens if tok not in ignore_terms}
                                f_operands = {self._number_to_word(tok) for tok in f_tokens_before_eq if tok not in ignore_terms}
                                if q_operands and q_operands == f_operands:
                                    return parts[eq_idx+5]
                    except (ValueError, IndexError):
                        pass

        # ── DREAM QUERY RESOLVER ──
        if re.search(r'ne ruyasi|ne rüyası|ne gordun|ne gördün|ruya|rüya', query_lower):
            from pathlib import Path
            import json
            diary_path = Path("./dream_diary.json")
            if diary_path.exists():
                try:
                    with open(diary_path, 'r', encoding='utf-8') as f:
                        diary_data = json.load(f)
                    concepts = diary_data.get('concepts', [])
                    if concepts:
                        clean_c = [c for c in concepts if c and len(c) > 2 and c not in ['.', ',', '?', '!']]
                        if clean_c:
                            return f"Rüyamda şu kavramları birleştirdim ve konsolide ettim: {', '.join(clean_c)}."
                except Exception as e:
                    pass
            return "Dün gece derin bir uyku çektim ama rüyamı net hatırlayamıyorum."

        if intent == 'GREETING':
            return self.CONVERSATIONAL['greeting'][0]
        if intent == 'WELLBEING' or re.search(r'nasilsin|nasılsın|naber|ne yapiyorsun|ne yapıyorsun', query_lower):
            return self.CONVERSATIONAL['wellbeing'][0]
        if intent == 'GRATITUDE':
            return self.CONVERSATIONAL['gratitude'][0]
        if intent == 'IDENTITY' or re.search(r'adin ne|adın ne|kimsin|kendini tanit|kendini tanıt|nesin|kim bu', query_lower):
            return self.CONVERSATIONAL['identity'][0]

        if re.search(r'bir cumle kur|bir cümle kur|ornek ver|örnek ver|cumle|cümle', query_lower):
            return self._generate_example_sentence()

        if knowledge_facts:
            result = self._explain_from_facts(
                facts=knowledge_facts,
                subject=subject,
                query=query_lower,
            )
            if result and len(result) > 10:
                return result

        if subject:
            subj_clean = self._clean_subject(subject)
            if subj_clean and subj_clean not in self.STOP_SUBJECTS:
                result = self._explain_from_subject(subj_clean)
                if result and len(result) > 10:
                    return result

        return self.CONVERSATIONAL['unknown'][0]

    def _explain_from_facts(
        self,
        facts: List[Dict],
        subject: Optional[str],
        query: str,
    ) -> str:
        subject_str = subject if subject else self._extract_subject(query)
        subject_clean = self._clean_subject(subject_str or '')

        primary: List[str] = []
        secondary: List[str] = []
        scored_primary = []
        scored_secondary = []
        seen = set()
        is_identity_query = self._is_identity_query(query)

        for fact in facts:
            text = (fact.get('text') or '').strip()
            if not self._is_usable_fact(text):
                continue
            if self._looks_like_question_residue(text, subject_clean):
                continue

            text_lower = text.lower()
            if text_lower in seen:
                continue
            seen.add(text_lower)

            has_subject = self._has_subject_signal(text_lower, subject_clean)
            is_generic = any(re.search(p, self._ascii_fold(text_lower)) for p in self.GENERIC_PATTERNS)
            is_meta_training = self._is_meta_training_fact(text_lower, subject_clean)
            quality = self._fact_quality(text, subject_clean, is_identity_query)

            if has_subject and not is_generic and not is_meta_training and quality >= 0.25:
                scored_primary.append((quality, text))
            elif quality >= 0.15:
                scored_secondary.append((quality, text))

        if is_identity_query and not primary:
            if not scored_primary:
                return ''
        scored_primary.sort(key=lambda item: item[0], reverse=True)
        scored_secondary.sort(key=lambda item: item[0], reverse=True)
        primary = [text for _, text in scored_primary]
        secondary = [text for _, text in scored_secondary]
        selected = (primary or secondary)[:2]
        return self._compose_explanation(selected, subject_str)

    def _explain_from_subject(self, subject: str) -> str:
        kb = getattr(self.brain, 'knowledge_base', []) or []
        good_sentences = []
        seen = set()
        for fact in kb:
            text = (fact.get('text') or '').strip()
            if not self._is_usable_fact(text):
                continue
            if self._looks_like_question_residue(text, subject):
                continue

            text_lower = text.lower()
            if subject not in self._ascii_fold(text_lower) or text_lower in seen:
                continue
            seen.add(text_lower)

            first_word = self._ascii_fold(text_lower.split()[0]) if text_lower.split() else ''
            if first_word == subject or first_word.startswith(subject[:4]):
                good_sentences.insert(0, text)
            else:
                good_sentences.append(text)

        return self._compose_explanation(good_sentences[:2], subject)

    def _generate_example_sentence(self) -> str:
        kb = getattr(self.brain, 'knowledge_base', []) or []

        for fact in kb:
            text = (fact.get('text') or '').strip()
            if self._is_usable_fact(text) and 3 < len(text.split()) < 15:
                return self._clean(f'Ornek cumle: {text}')
        return 'Cumle kurmak icin bilgi tabanimda yeterli kanit yok.'

    def _compose_explanation(self, sentences: List[str], subject: Optional[str]) -> str:
        if not sentences:
            return ''

        compact = self._dedupe_compact([self._compact_fact(s, subject) for s in sentences if s])
        if not compact:
            return ''
        if len(compact) == 1:
            return self._clean(compact[0])

        return self._clean(f'{compact[0]}. Ayrica {compact[1][0].lower() + compact[1][1:]}')

    def _dedupe_compact(self, sentences: List[str]) -> List[str]:
        result = []
        seen_tokens = []
        for sentence in sentences:
            clean = sentence.strip()
            if not clean:
                continue
            tokens = set(re.findall(r'\w+', self._ascii_fold(clean)))
            if any(tokens and len(tokens & prev) / max(1, min(len(tokens), len(prev))) > 0.6 for prev in seen_tokens):
                continue
            seen_tokens.append(tokens)
            result.append(clean)
        return result

    def _compact_fact(self, sentence: str, subject: Optional[str]) -> str:
        text = re.sub(r'\s+', ' ', sentence or '').strip().strip('.!?')
        if not text:
            return ''

        subject_clean = (subject or '').strip()
        words = text.split()

        if subject_clean:
            # Drop any leading word that echoes a subject token, so a
            # multi-word subject ("Hebbian ogrenme") is not half-repeated
            # in the fragment.
            subject_prefixes = [
                self._ascii_fold(tok)[:4]
                for tok in subject_clean.split()
                if len(self._ascii_fold(tok)) >= 3
            ]
            kept = []
            for word in words:
                folded = self._ascii_fold(word)
                if any(prefix and folded.startswith(prefix) for prefix in subject_prefixes):
                    continue
                kept.append(word)
            fragment = ' '.join(kept[:12]).strip()
            if fragment:
                return f'{subject_clean} hakkinda: {fragment}'

        if len(words) > 12:
            return ' '.join(words[:12])
        if len(words) > 4:
            return ' '.join(words[:-1])
        return text

    def _looks_like_question_residue(self, text: str, subject_clean: str) -> bool:
        folded = self._ascii_fold(text)
        tokens = re.findall(r'\w+', folded)
        if not tokens:
            return True
        if '?' in text:
            return True

        question_hits = sum(1 for token in tokens if token in self.QUESTION_TERMS)
        subject_hits = 0
        if subject_clean:
            folded_subject = self._ascii_fold(subject_clean)
            subject_hits = sum(
                1 for token in tokens
                if len(folded_subject) >= 3 and token.startswith(folded_subject[:4])
            )

        if len(tokens) <= 4 and question_hits > 0:
            return True
        if len(tokens) <= 6 and question_hits > 0 and subject_hits > 0:
            return True
        return False

    def _extract_subject(self, query: str) -> Optional[str]:
        patterns = [
            r'(\S+)\s+nedir',
            r'(\S+)\s+ne demek',
            r'(\S+)\s+nasil',
            r'(\S+)\s+nasıl',
            r'(\S+)\s+hakkinda',
            r'(\S+)\s+hakkında',
            r'(\S+)\s+kimdir',
            r'(\S+)\s+neler',
            r'(\S+)\s+ne ise',
            r'(\S+)\s+ne işe',
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                candidate = self._clean_subject(match.group(1))
                if candidate and len(candidate) > 2 and candidate not in self.STOP_SUBJECTS:
                    return candidate
        return None

    def _is_usable_fact(self, text: str) -> bool:
        if not text or len(text) < 5:
            return False
        words = text.split()
        return 2 <= len(words) <= 50

    def _is_meta_training_fact(self, text_lower: str, subject_clean: str) -> bool:
        if subject_clean != 'mergen':
            return False
        folded = self._ascii_fold(text_lower)
        return any(self._ascii_fold(term) in folded for term in self.META_TRAINING_TERMS)

    def _is_identity_query(self, query: str) -> bool:
        folded = self._ascii_fold(query)
        return any(term in folded for term in ('kimdir', 'kimsin', 'nedir', 'ne demek'))

    def _has_subject_signal(self, text_lower: str, subject_clean: str) -> bool:
        if not subject_clean:
            return False
        folded_text = self._ascii_fold(text_lower)
        for alias in self._subject_aliases(subject_clean):
            if alias and alias in folded_text:
                return True
        return False

    def _subject_aliases(self, subject_clean: str) -> List[str]:
        folded = self._ascii_fold(subject_clean)
        aliases = [folded]
        alias_map = {
            'yercekimi': ['kutlecekim', 'gravitasyonel', 'gravity'],
            'kutlecekim': ['yercekimi', 'gravitasyonel', 'gravity'],
            'ruya': ['dream', 'konsolidasyon'],
            'dream': ['ruya', 'konsolidasyon'],
            'ruyakonsolidasyonu': ['dreamkonsolidasyonu', 'konsolidasyon', 'dream', 'ruya'],
            'dreamkonsolidasyonu': ['ruyakonsolidasyonu', 'konsolidasyon', 'dream', 'ruya'],
        }
        for key, values in alias_map.items():
            if key in folded:
                aliases.extend(values)
        return list(dict.fromkeys(aliases))

    def _fact_quality(self, text: str, subject_clean: str, is_identity_query: bool) -> float:
        folded = self._ascii_fold(text)
        tokens = re.findall(r'\w+', folded)
        if not tokens:
            return 0.0

        score = 0.2
        if subject_clean:
            subject_folded = self._ascii_fold(subject_clean)
            if folded.startswith(subject_folded[:4]):
                score += 0.25
            if self._has_subject_signal(text, subject_clean):
                score += 0.20

        if is_identity_query:
            definition_terms = (
                'bir ', 'dir', 'tir', 'olarak', 'kurucusu', 'hukumdari',
                'imparatoru', 'lideri', 'bilim insani', 'mimaridir', 'beyindir',
            )
            if any(term in folded for term in definition_terms):
                score += 0.30
            low_quality_terms = (
                'evlendirildi', 'kizi', 'kizi', 'destansi', 'oykudur',
                'cercevesinde olusmus', 'yasami cercevesinde',
            )
            if any(term in folded for term in low_quality_terms):
                score -= 0.35

        direct_answer_terms = (
            'arasinda cekim', 'cekim etkisi', 'uzay ve zamani buk',
            'sinaptik izleri', 'hafiza izlerini', 'geri cagir', 'yaniti destekler',
            'biyolojiden ilham',
        )
        if any(term in folded for term in direct_answer_terms):
            score += 0.20

        broad_context_terms = (
            'gunes sistemi', 'mars', 'filmde', 'ev sahipligi', 'yasami cercevesinde',
            'evlendirildi', 'destansi', 'kara delik',
        )
        if any(term in folded for term in broad_context_terms):
            score -= 0.25

        if len(tokens) > 32:
            score -= 0.10
        return max(0.0, min(1.0, score))

    def _clean_subject(self, subject: str) -> str:
        # Keep internal whitespace so multi-word subjects ("Hebbian ogrenme")
        # stay token-separated and can match spaced fact text. Only punctuation
        # is stripped; runs of whitespace collapse to a single space.
        cleaned = re.sub(r'[^\wçğıöşüÇĞİÖŞÜ\s]', '', subject or '')
        return re.sub(r'\s+', ' ', cleaned).strip().lower()

    def _ascii_fold(self, text: str) -> str:
        table = str.maketrans({
            'ç': 'c', 'ğ': 'g', 'ı': 'i', 'i': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
            'Ç': 'c', 'Ğ': 'g', 'İ': 'i', 'I': 'i', 'Ö': 'o', 'Ş': 's', 'Ü': 'u',
        })
        return (text or '').translate(table).lower()

    def _clean(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\s+\.', '.', text)
        if text:
            text = text[0].upper() + text[1:]
        if text and text[-1] not in '.!?':
            text += '.'
        return text

    def _number_to_word(self, t: str) -> str:
        t_clean = t.strip()
        if not t_clean.isdigit():
            return t_clean
        n = int(t_clean)
        if n == 0:
            return 'sifir'
        birler = ['', 'bir', 'iki', 'uc', 'dort', 'bes', 'alti', 'yedi', 'sekiz', 'dokuz']
        onlar = ['', 'on', 'yirmi', 'otuz', 'kirk', 'elli', 'altmis', 'yetmis', 'seksen', 'doksan']
        if n < 10:
            return birler[n]
        elif n < 20:
            return 'on' + birler[n % 10]
        elif n < 100:
            o, b = divmod(n, 10)
            return onlar[o] + birler[b]
        return t_clean
