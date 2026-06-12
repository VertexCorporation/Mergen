"""
╔══════════════════════════════════════════════════════════════════════╗
║         MERGEN V3 — INTENT ANALYZER (Fixed)                          ║
║                                                                      ║
║  Author:  Vertex Corporation — Mergen Project                       ║
║  License: Apache-2.0                                                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import re
import json
import math
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from collections import deque, Counter


class IntentAnalyzer:
    """
    Mergen's semantic traffic analyzer.
    """

    INTENT_KEYWORDS = {
        'GREETING': {
            'weight': 2.5,
            'keywords': [
                'merhaba', 'selam', 'selamlar', 'hello', 'hi', 'hey',
                'günaydın', 'iyi günler', 'iyi akşamlar', 'iyi geceler',
                'good morning', 'good evening', 'nasılsın', 'naber',
                'hoş geldin', 'hoşbulduk', 'merhabalar',
                'iyi', 'günaydın', 'iyi akşamlar',
            ],
        },
        'IDENTITY': {
            'weight': 2.5,
            'keywords': [
                'kimsin', 'adın ne', 'kendini tanıt', 'tanıtır mısın',
                'who are you', "what's your name", 'introduce yourself',
                'sen kimsin', 'sen nesin', 'hakkında', 'about you',
                'nesin', 'adın', 'sen kim', 'kendin',
                'ben kimim', 'kim bu', 'bu kim',
            ],
        },
        'INQUIRY': {
            'weight': 2.0,
            'keywords': [
                'nedir', 'ne demek', 'açıkla', 'anlat', 'what is', 'explain',
                'tell me', 'describe', 'nasıl', 'neden', 'niye',
                'why', 'how', 'what', 'hangi', 'which', 'kim', 'nerede',
                'ne zaman', 'kaç', 'hangisi', 'neler', 'kimdir',
                'nedi', 'neymiş', 'nası', 'nedeni', 'sebebi',
                'bilgi ver', 'söyle', 'öğrenmek', 'bilmek istiyorum',
                'merak ediyorum', 'ne anlama', 'neye', 'neyden',
                'hangi', 'hangisi', 'ne tür', 'ne çeşit',
            ],
        },
        'COMMAND': {
            'weight': 2.0,
            'keywords': [
                'yap', 'et', 'oluştur', 'üret', 'hesapla', 'çöz',
                'do', 'make', 'create', 'generate', 'calculate', 'solve',
                'çalıştır', 'başlat', 'durdur', 'run', 'start', 'stop',
                'kur', 'yaz', 'göster', 'bul', 'getir', 'ver',
                'oku', 'öğren', 'kaydet', 'sil', 'değiştir',
                'bir cümle', 'örnek', 'listele', 'sırala',
            ],
        },
        'EMOTION': {
            'weight': 2.0,
            'keywords': [
                'hissediyorum', 'üzgünüm', 'mutluyum', 'sinirliyim',
                'heyecanlıyım', 'yoruldum', 'sıkıldım',
                'i feel', 'i am sad', 'i am happy', 'i am angry',
                'excited', 'tired', 'bored', 'frustrated',
                'seviyorum', 'nefret', 'korkuyorum', 'endişeli',
                'kızgın', 'memnun', 'hayal kırıklığı',
            ],
        },
        'AFFIRMATION': {
            'weight': 2.0,
            'keywords': [
                'evet', 'tabii', 'tamam', 'doğru', 'haklısın',
                'yes', 'yeah', 'sure', 'correct', 'right', 'agreed',
                'olur', 'peki', 'anladım', 'güzel', 'süper',
                'harika', 'mükemmel', 'tamamdır', 'eyvallah',
            ],
        },
        'NEGATION': {
            'weight': 2.0,
            'keywords': [
                'hayır', 'yok', 'yanlış', 'değil',
                'no', 'nope', 'wrong', 'incorrect', 'not',
                'asla', 'istemiyorum', 'istemem', 'olmaz',
                'katılmıyorum', 'yanlış', 'saçma',
            ],
        },
        'GRATITUDE': {
            'weight': 2.5,
            'keywords': [
                'teşekkür', 'sağ ol', 'thanks', 'thank you',
                'teşekkürler', 'eyvallah', 'minnettarım',
                'sağ olun', 'çok teşekkürler', 'çok sağ ol',
            ],
        },
        'WELLBEING': {
            'weight': 2.5,
            'keywords': [
                'nasılsın', 'nasılsınız', 'nasıl gidiyor', 'keyifler nasıl',
                'how are you', 'how do you do', 'ne yapıyorsun', 'ne yapıyorsunuz',
                'meşgul', 'ne yapıyosun', 'naber', 'napıyorsun',
            ],
        },
    }

    QUESTION_SUFFIXES = ['mı', 'mi', 'mu', 'mü', 'mıdır', 'midir',
                         'mudur', 'müdür']
    TENSE_MARKERS = {
        'past':   ['dı', 'di', 'du', 'dü', 'tı', 'ti', 'tu', 'tü',
                   'mıştı', 'mişti', 'muştu', 'müştü'],
        'present': ['yor', 'ıyor', 'iyor', 'uyor', 'üyor'],
        'future': ['cak', 'cek', 'acak', 'ecek'],
    }
    MODALITY_MARKERS = {
        'necessity':  ['gerek', 'lazım', 'şart', 'zorunlu', 'must', 'need'],
        'possibility': ['bilir', 'bilir mi', 'olabilir', 'maybe', 'can', 'could'],
        'ability':    ['abilir', 'ebilir', 'abilir misin', 'can you', 'could you'],
    }

    SENTIMENT_POS = [
        'harika', 'mükemmel', 'süper', 'çok iyi', 'güzel', 'sevdim',
        'teşekkür', 'mutlu', 'başarılı',
        'great', 'awesome', 'excellent', 'wonderful', 'happy', 'love',
    ]
    SENTIMENT_NEG = [
        'berbat', 'kötü', 'saçma', 'sıkıcı', 'anlamsız', 'yanlış',
        'sinir', 'üzgün', 'yorgun',
        'terrible', 'bad', 'awful', 'boring', 'wrong', 'angry', 'sad',
    ]

    EXCITEMENT_MARKERS = ['!', '!!!', '?!', 'ohh', 'waow', 'vay']

    # ── Subject extraction: multi-word patterns ──
    QUESTION_PATTERNS = [
        (r'((?:\S+\s+){0,3}\S+)\s+nedir', 1),
        (r'((?:\S+\s+){0,3}\S+)\s+ne demek', 1),
        (r'((?:\S+\s+){0,3}\S+)\s+nasıl', 1),
        (r'((?:\S+\s+){0,3}\S+)\s+neden', 1),
        (r'((?:\S+\s+){0,3}\S+)\s+kimdir', 1),
        (r'((?:\S+\s+){0,3}\S+)\s+kim', 1),
        (r'((?:\S+\s+){0,3}\S+)\s+ne zaman', 1),
        (r'((?:\S+\s+){0,3}\S+)\s+nerede', 1),
        (r'((?:\S+\s+){0,3}\S+)\s+kaç', 1),
        (r'((?:\S+\s+){0,3}\S+)\s+hakkında', 1),
        (r'((?:\S+\s+){0,3}\S+)\s+ile\s+ilgili', 1),
        (r'((?:\S+\s+){0,3}\S+)\s+neler', 1),
        # New patterns for better understanding
        (r'((?:\S+\s+){0,3}\S+)\s+ne işe yarar', 1),
        (r'((?:\S+\s+){0,3}\S+)\s+ne anlama gelir', 1),
        (r'((?:\S+\s+){0,3}\S+)\s+ne için kullanılır', 1),
        (r'((?:\S+\s+){0,3}\S+)\s+nasıl yapılır', 1),
        (r'((?:\S+\s+){0,3}\S+)\s+nasıl kullanılır', 1),
        (r'((?:\S+\s+){0,3}\S+)\s+neden olur', 1),
        (r'((?:\S+\s+){0,3}\S+)\s+niye var', 1),
    ]

    def __init__(
        self,
        memory_path: str = './mergen_matrix_memory.json',
        context_window: int = 10,
        low_confidence_threshold: float = 0.4,
    ):
        self.memory_path = Path(memory_path)
        self.context_window = context_window
        self.low_confidence_threshold = low_confidence_threshold
        self.context_buffer: deque = deque(maxlen=context_window)
        self.last_subject: Optional[str] = None
        self.last_intent: Optional[str] = None
        self.memory = self._load_memory()
        self.total_analyses = 0
        self.low_confidence_count = 0

    def _load_memory(self) -> Dict:
        default_memory = {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'stats': {
                'total_analyses': 0,
                'intent_counts': {},
                'low_confidence_events': 0,
                'avg_confidence': 0.0,
                'subject_frequency': {},
            },
            'analyze_logs': [],
            'low_confidence_queue': [],
        }
        if self.memory_path.exists():
            try:
                loaded = json.loads(
                    self.memory_path.read_text(encoding='utf-8')
                )
                if 'stats' not in loaded:
                    loaded['stats'] = default_memory['stats']
                else:
                    for k, v in default_memory['stats'].items():
                        if k not in loaded['stats']:
                            loaded['stats'][k] = v
                for k in ('analyze_logs', 'low_confidence_queue'):
                    if k not in loaded:
                        loaded[k] = []
                return loaded
            except Exception:
                pass
        return default_memory

    def _save_memory(self):
        try:
            self.memory_path.write_text(
                json.dumps(self.memory, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
        except Exception as e:
            print(f"[IntentAnalyzer] Memory save error: {e}")

    def _score_intents(self, text_low: str) -> Dict[str, float]:
        """Score intents with exact word matching (no prefix matching)."""
        words = set(re.findall(r'\w+', text_low))
        total_words = max(1, len(re.findall(r'\w+', text_low)))

        scores = {}
        for intent, spec in self.INTENT_KEYWORDS.items():
            weight = spec['weight']
            keywords = spec['keywords']
            matches = 0
            position_bonus = 0.0

            for kw in keywords:
                if ' ' in kw:
                    if kw in text_low:
                        matches += 1
                        pos = text_low.find(kw)
                        position_bonus += 1.0 - (pos / max(1, len(text_low)))
                else:
                    if kw in words:
                        matches += 1
                        pos = text_low.find(kw)
                        position_bonus += 1.0 - (pos / max(1, len(text_low)))

            if matches == 0:
                scores[intent] = 0.0
                continue

            density = matches / total_words
            base = matches * weight
            pos_contrib = min(1.0, position_bonus / max(1, matches))
            scores[intent] = base * (1.0 + density) * (1.0 + pos_contrib * 0.3)

        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        return scores

    def _analyze_morphology(self, text: str, text_low: str) -> Dict:
        morphology = {
            'is_question': False,
            'question_type': None,
            'tense': None,
            'modality': None,
            'has_exclamation': '!' in text,
            'capital_ratio': 0.0,
        }

        if '?' in text:
            morphology['is_question'] = True
            morphology['question_type'] = 'direct'
        else:
            words = text_low.split()
            for w in words:
                clean = re.sub(r'[^\w]', '', w)
                if clean in self.QUESTION_SUFFIXES:
                    morphology['is_question'] = True
                    morphology['question_type'] = 'yes_no'
                    break
                for suf in self.QUESTION_SUFFIXES:
                    if clean.endswith(suf) and len(clean) > len(suf) + 2:
                        morphology['is_question'] = True
                        morphology['question_type'] = 'embedded'
                        break

        for tense, markers in self.TENSE_MARKERS.items():
            for m in markers:
                if re.search(rf'\w+{m}\b', text_low):
                    morphology['tense'] = tense
                    break
            if morphology['tense']:
                break

        for mod, markers in self.MODALITY_MARKERS.items():
            for m in markers:
                if m in text_low:
                    morphology['modality'] = mod
                    break
            if morphology['modality']:
                break

        alpha = [c for c in text if c.isalpha()]
        if alpha:
            morphology['capital_ratio'] = sum(
                1 for c in alpha if c.isupper()
            ) / len(alpha)

        return morphology

    def _extract_subject_action(self, text: str, text_low: str) -> Dict:
        """
        Extract subject using pattern matching first, then heuristics.
        """
        subject = None
        action = None
        context_link = None

        # Strategy 1: Regex patterns for common question structures
        for pattern, group in self.QUESTION_PATTERNS:
            match = re.search(pattern, text_low)
            if match:
                candidate = match.group(group)
                # Clean punctuation but KEEP spaces
                candidate = re.sub(r'[^\wçğıöşüÇĞİÖŞÜ\s]', '', candidate).strip()
                if candidate and len(candidate) > 1:
                    # Title case: "hibrit otomobil" → "Hibrit Otomobil"
                    subject = ' '.join(w.capitalize() for w in candidate.split())
                    break

        # Strategy 2: Look for noun before question word
        if subject is None:
            question_words = ['nedir', 'ne', 'nasıl', 'neden', 'niye',
                            'kim', 'nerede', 'neler', 'kimdir', 'kaç']
            words = text_low.split()
            for i, w in enumerate(words):
                w_clean = re.sub(r'[^\w]', '', w)
                if w_clean in question_words and i > 0:
                    candidate = words[i - 1]
                    candidate = re.sub(r'[^\wçğıöşüÇĞİÖŞÜ]', '', candidate)
                    if candidate and len(candidate) > 2:
                        subject = candidate.capitalize()
                        break

        # Strategy 3: For "ne yapıyorsun" type questions, extract action
        if subject is None:
            action_patterns = [
                (r'ne\s+yapıyorsun', 'yapıyorsun'),
                (r'ne\s+yapıyorsunuz', 'yapıyorsunuz'),
                (r'ne\s+yapıyosun', 'yapıyorsun'),
                (r'napıyorsun', 'yapıyorsun'),
                (r'ne\s+oluyor', 'oluyor'),
                (r'ne\s+var', 'var'),
                (r'ne\s+oldu', 'oldu'),
                (r'ne\s+haber', 'haber'),
                (r'bir\s+cümle\s+kur', 'kur'),
                (r'cümle\s+kur', 'kur'),
                (r'örnek\s+ver', 'ver'),
                (r'örnek\s+cümle', 'cümle'),
            ]
            for pattern, act in action_patterns:
                if re.search(pattern, text_low):
                    action = act
                    subject = None  # Let context handle it
                    break

        # Strategy 4: First content word (stop-word filtered)
        if subject is None and action is None:
            stop_words = {
                'bir', 've', 'ile', 'bu', 'şu', 'o', 'bunu', 'şunu', 'onu',
                'ben', 'sen', 'biz', 'siz', 'onlar', 'için', 'gibi', 'kadar',
                'de', 'da', 'ki', 'mi', 'mı', 'mu', 'mü', 'ne', 'nasıl',
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                'and', 'or', 'but', 'for', 'with', 'of', 'in', 'to', 'from',
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that',
                'nedir', 'ne', 'kim', 'nere', 'kaç', 'neler', 'nasıl', 'neden',
                'niye', 'hangisi', 'hangi', 'acaba', 'sence', 'bence',
                'merhaba', 'selam', 'hello', 'hi',
                'yapıyorsun', 'yapıyorsunuz', 'yapıyosun', 'napıyorsun',
                'kur', 'ver', 'söyle', 'göster', 'yaz',
            }
            words = re.findall(r'\w+', text_low)
            content_words = [w for w in words if w not in stop_words and len(w) > 2]
            if content_words:
                subject = content_words[0].capitalize()

        # Pronoun resolution
        reference_pronouns = {
            'o', 'onu', 'ona', 'onun', 'bu', 'bunu', 'buna', 'şu', 'şunu',
            'it', 'this', 'that', 'him', 'her', 'them',
        }
        if subject and subject.lower() in reference_pronouns:
            for past in reversed(self.context_buffer):
                if past.get('subject') and past['subject'].lower() not in reference_pronouns:
                    context_link = past['subject']
                    subject = context_link
                    break
            if context_link is None and self.last_subject:
                subject = self.last_subject

        # Action extraction
        verb_patterns = [
            r'\w+(?:yor|ıyor|iyor|uyor|üyor)\b',
            r'\w+(?:dı|di|du|dü|tı|ti|tu|tü)\b',
            r'\w+(?:cak|cek|acak|ecek)\b',
            r'\b(?:is|was|were|are|be|do|does|did|have|had)\b',
        ]
        for pattern in verb_patterns:
            match = re.search(pattern, text_low)
            if match:
                action = match.group()
                break

        if action is None:
            words = re.findall(r'\w+', text_low)
            stop_words = {'bir', 've', 'ile', 'bu', 'şu', 'o', 'de', 'da', 'mi', 'mı', 'ne', 'nasıl'}
            content_words = [w for w in words if w not in stop_words and len(w) > 2]
            if len(content_words) > 1:
                action = content_words[-1]

        return {
            'subject': subject,
            'action': action,
            'context_link': context_link,
        }

    def _analyze_sentiment(self, text: str, text_low: str, morphology: Dict) -> Dict:
        pos_count = sum(1 for w in self.SENTIMENT_POS if w in text_low)
        neg_count = sum(1 for w in self.SENTIMENT_NEG if w in text_low)
        total = pos_count + neg_count
        sentiment_score = (pos_count - neg_count) / total if total > 0 else 0.0

        excitement = 0.0
        excitement += text.count('!') * 0.2
        excitement += morphology.get('capital_ratio', 0) * 0.5
        for marker in self.EXCITEMENT_MARKERS:
            if marker in text_low:
                excitement += 0.3

        if excitement > 0.6:
            tone = 'excited'
        elif sentiment_score < -0.3:
            tone = 'frustrated' if excitement > 0.3 else 'negative'
        elif sentiment_score > 0.3:
            tone = 'positive'
        else:
            tone = 'neutral'

        return {
            'sentiment_score': round(sentiment_score, 3),
            'excitement': round(min(1.0, excitement), 3),
            'tone': tone,
            'pos_words': pos_count,
            'neg_words': neg_count,
        }

    def _compute_confidence(self, intent_scores: Dict[str, float],
                            subject: Optional[str], morphology: Dict) -> float:
        if not intent_scores or max(intent_scores.values()) == 0:
            return 0.1
        sorted_scores = sorted(intent_scores.values(), reverse=True)
        top = sorted_scores[0]
        second = sorted_scores[1] if len(sorted_scores) > 1 else 0
        margin = top - second
        intent_clarity = min(1.0, top + margin)
        subject_bonus = 0.2 if subject else 0.0
        morph_bonus = 0.0
        if morphology.get('is_question') is not None:
            morph_bonus += 0.1
        if morphology.get('tense'):
            morph_bonus += 0.1
        if morphology.get('modality'):
            morph_bonus += 0.05
        confidence = (intent_clarity * 0.6 + subject_bonus + morph_bonus)
        return round(min(1.0, max(0.0, confidence)), 3)

    def analyze_intent(self, user_input: str) -> Dict:
        self.total_analyses += 1
        text = (user_input or "").strip()
        text_low = text.lower()

        if not text:
            return self._empty_report()

        intent_scores = self._score_intents(text_low)
        morphology = self._analyze_morphology(text, text_low)
        subj_act = self._extract_subject_action(text, text_low)
        sentiment = self._analyze_sentiment(text, text_low, morphology)

        if intent_scores and max(intent_scores.values()) > 0:
            primary_intent = max(intent_scores, key=intent_scores.get)
        else:
            primary_intent = 'INQUIRY' if morphology['is_question'] else 'UNKNOWN'

        if morphology['is_question'] and primary_intent not in ('INQUIRY', 'IDENTITY'):
            question_words = {'nedir', 'nasıl', 'neden', 'niye', 'kim', 'nerede', 'ne zaman', 'kaç', 'neler', 'ne demek'}
            words = set(re.findall(r'\w+', text_low))
            if words & question_words:
                primary_intent = 'INQUIRY'
                intent_scores['INQUIRY'] = max(intent_scores.get('INQUIRY', 0), 0.5)

        confidence = self._compute_confidence(intent_scores, subj_act['subject'], morphology)
        is_low_conf = confidence < self.low_confidence_threshold
        if is_low_conf:
            self.low_confidence_count += 1

        report = {
            'timestamp': time.time(),
            'input': text[:200],
            'primary_intent': primary_intent,
            'intent_scores': {k: round(v, 3) for k, v in intent_scores.items() if v > 0.01},
            'subject': subj_act['subject'],
            'action': subj_act['action'],
            'context_link': subj_act['context_link'],
            'morphology': morphology,
            'sentiment': sentiment,
            'confidence_score': confidence,
            'is_low_confidence': is_low_conf,
        }

        self._update_mx_memory(report)
        self.context_buffer.append({
            'subject': subj_act['subject'],
            'intent': primary_intent,
            'timestamp': time.time(),
        })
        reference_pronouns = {'o', 'onu', 'ona', 'onun', 'bu', 'bunu', 'buna', 'şu', 'şunu'}
        if subj_act['subject'] and subj_act['subject'].lower() not in reference_pronouns:
            self.last_subject = subj_act['subject']
        self.last_intent = primary_intent

        return report

    def _empty_report(self) -> Dict:
        return {
            'primary_intent': 'UNKNOWN',
            'intent_scores': {},
            'subject': None, 'action': None, 'context_link': None,
            'morphology': {}, 'sentiment': {},
            'confidence_score': 0.0, 'is_low_confidence': True,
        }

    def _update_mx_memory(self, report: Dict):
        stats = self.memory['stats']
        stats['total_analyses'] += 1
        intent = report['primary_intent']
        stats['intent_counts'][intent] = stats['intent_counts'].get(intent, 0) + 1
        subject = report['subject']
        if subject:
            base = subject.split('→')[0] if '→' in subject else subject
            stats['subject_frequency'][base] = stats['subject_frequency'].get(base, 0) + 1
        n = stats['total_analyses']
        old_avg = stats['avg_confidence']
        stats['avg_confidence'] = (old_avg * (n - 1) + report['confidence_score']) / n
        self.memory['analyze_logs'].append({
            'ts': report['timestamp'],
            'input': report['input'][:100],
            'intent': report['primary_intent'],
            'subject': report['subject'],
            'confidence': report['confidence_score'],
            'is_low_confidence': report['is_low_confidence'],
        })
        if len(self.memory['analyze_logs']) > 1000:
            self.memory['analyze_logs'] = self.memory['analyze_logs'][-1000:]
        if report['is_low_confidence']:
            stats['low_confidence_events'] += 1
            self.memory['low_confidence_queue'].append({
                'ts': report['timestamp'],
                'input': report['input'],
                'intent': report['primary_intent'],
                'confidence': report['confidence_score'],
                'status': 'pending_review',
            })
            if len(self.memory['low_confidence_queue']) > 200:
                self.memory['low_confidence_queue'] = self.memory['low_confidence_queue'][-200:]
        if self.total_analyses % 10 == 0:
            self._save_memory()

    def record_feedback(self, analysis_timestamp: float,
                        was_correct: bool, true_intent: Optional[str] = None):
        for log in reversed(self.memory['analyze_logs']):
            if abs(log['ts'] - analysis_timestamp) < 1.0:
                log['was_correct'] = was_correct
                if true_intent:
                    log['true_intent'] = true_intent
                for item in self.memory['low_confidence_queue']:
                    if abs(item['ts'] - analysis_timestamp) < 1.0:
                        item['status'] = 'resolved'
                        item['was_correct'] = was_correct
                        break
                break
        self._save_memory()

    def get_telemetry(self) -> Dict:
        stats = self.memory['stats']
        return {
            'total_analyses': self.total_analyses,
            'low_confidence_count': self.low_confidence_count,
            'lifetime_analyses': stats.get('total_analyses', 0),
            'avg_confidence': round(stats.get('avg_confidence', 0), 3),
            'intent_distribution': stats.get('intent_counts', {}),
            'top_subjects': dict(Counter(
                stats.get('subject_frequency', {})
            ).most_common(5)),
            'pending_reviews': len([
                q for q in self.memory.get('low_confidence_queue', [])
                if q.get('status') == 'pending_review'
            ]),
        }
