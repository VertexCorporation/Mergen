"""MERGEN — RESPONSE GENERATOR v6 (Explain, don't meta-talk)

Mergen artık "biliyorum", "öğrendim" gibi meta-cümleler kurmuyor.
KB'deki GERÇEK bilgileri kullanarak konuyu AÇIKLIYOR.
"""

import re
import random
from typing import Optional, Dict, List


class ResponseGenerator:
    """
    Mergen'in cevap üretme motoru.
    KB'deki gerçek bilgileri kullanarak AÇIKLAYICI cevaplar üretir.
    """

    # Conversational responses for non-knowledge intents
    CONVERSATIONAL = {
        'greeting': [
            'Merhaba! Sana nasıl yardımcı olabilirim?',
            'Selam! Bugün ne konuşmak istersin?',
            'Merhabalar, seni dinliyorum.',
        ],
        'wellbeing': [
            'Teşekkür ederim, iyiyim! Seninle sohbet etmek güzel.',
            'İyiyim, yeni şeyler öğrenmeye devam ediyorum.',
            'Harikayım! Bana sorduğun için teşekkürler.',
        ],
        'gratitude': [
            'Rica ederim!',
            'Ne demek, yardımcı olabildiysem ne mutlu bana!',
            'Teşekkürler, sohbet güzeldi!',
        ],
        'identity': [
            'Ben Mergen. Biyolojiden ilham alan bir dijital beyinim. Hebbian öğrenme ve nöral ağlarla çalışıyorum.',
            'Adım Mergen. Kendi sözlüğüm ve bilgi tabanım olan bir yapay zekayım. Her konuşmada yeni şeyler öğreniyorum.',
            'Ben Mergen. Türkçe A1 seviyesinde kelimeler bilen, dosyaları okuyup öğrenebilen bir dijital beyinim.',
        ],
        'unknown': [
            'Bunu henüz öğrenemedim. Bana bir dosya okutursan öğrenirim.',
            'Bu konu hakkında bilgim yok. "oku:dosya.txt" ile bana öğretebilirsin.',
        ],
    }

    STOP_SUBJECTS = {
        'bir', 've', 'ile', 'bu', 'şu', 'o', 'de', 'da', 'mi', 'mı', 'mu', 'mü',
        'ne', 'nasıl', 'neden', 'niye', 'nedir', 'kim', 'nerede', 'ne zaman',
        'the', 'a', 'an', 'is', 'in', 'of', 'to', 'and', 'or', 'but',
        'yapıyorsun', 'yapıyorsunuz', 'yapıyosun', 'napıyorsun',
        'kur', 'ver', 'söyle', 'göster', 'yaz',
        'cümle', 'örnek', 'şey', 'kadar', 'için', 'gibi',
    }

    # Sentence starters for natural flow
    OPENERS = [
        '',  # Direct answer, no opener
        '',
        '',
        'Şöyle açıklayabilirim: ',
    ]

    CONNECTORS = [
        'Ayrıca, ',
        'Bunu şöyle de söyleyebilirim: ',
        'Bir başka bilgi de: ',
        '',
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
        """Ana cevap üretme fonksiyonu."""
        self.response_count += 1
        query_lower = query.lower().strip()

        # 1. Conversational intents
        if intent == 'GREETING':
            return random.choice(self.CONVERSATIONAL['greeting'])

        if intent == 'WELLBEING' or re.search(r'nasılsın|naber|ne yapıyorsun', query_lower):
            return random.choice(self.CONVERSATIONAL['wellbeing'])

        if intent == 'GRATITUDE':
            return random.choice(self.CONVERSATIONAL['gratitude'])

        if intent == 'IDENTITY' or re.search(r'adın ne|kimsin|kendini tanıt|nesin|kim bu', query_lower):
            return random.choice(self.CONVERSATIONAL['identity'])

        # 2. "bir cümle kur" - example sentence request
        if re.search(r'bir cümle kur|örnek ver|cümle', query_lower):
            return self._generate_example_sentence(query_lower)

        # 3. Knowledge-based: GERÇEK BİLGİLERİ ANLAT
        if knowledge_facts and len(knowledge_facts) > 0:
            result = self._explain_from_facts(
                facts=knowledge_facts,
                subject=subject,
                query=query_lower,
            )
            if result and len(result) > 10:
                return result

        # 4. Subject-based search
        if subject:
            subj_clean = re.sub(r'[^\wçğıöşüÇĞİÖŞÜ]', '', subject).lower()
            if subj_clean and subj_clean not in self.STOP_SUBJECTS:
                result = self._explain_from_subject(subj_clean, query_lower)
                if result and len(result) > 10:
                    return result

        # 5. Fallback
        return random.choice(self.CONVERSATIONAL['unknown'])

    def _explain_from_facts(
        self,
        facts: List[Dict],
        subject: Optional[str],
        query: str,
    ) -> str:
        """KB'deki gerçek bilgileri kullanarak AÇIKLAYICI cevap üret."""
        subject_str = subject if subject else self._extract_subject(query)
        subject_clean = re.sub(r'[^\wçğıöşüÇĞİÖŞÜ]', '', subject_str).lower() if subject_str else ''

        # KB'den GERÇEK, anlamlı cümleleri çıkar
        seen = set()
        primary = []
        secondary = []

        # Generic A1 template patterns to skip
        _GENERIC_PATTERNS = [
            r'^\w+.*?\b(?:isim|fiil|sıfat|zarf|bağlaç|edat|zamir)dir',
            r'^masada\s+bir\s+',
            r'^bu\s+\w+\s+güzel',
            r'^günlük\s+hayatta\s+\w+\s+kullanılır',
            r'^\w+\s+önemli',
            r'^her\s+gün\s+',
            r'^insanlar\s+',
            r'^hava\s+çok\s+',
        ]

        for fact in facts:
            text = fact.get('text', '').strip()
            if not text or len(text) < 5:
                continue
            text_lower = text.lower()
            has_subject = subject_clean and subject_clean in text_lower
            if text_lower in seen:
                continue
            seen.add(text_lower)
            words = text.split()
            if len(words) < 2 or len(words) > 50:
                continue

            # Skip generic A1 templates if we have better facts
            is_generic = any(re.search(p, text_lower) for p in _GENERIC_PATTERNS)

            if has_subject:
                if is_generic:
                    secondary.append(text)  # Generic templates go to secondary
                else:
                    primary.append(text)  # Real facts first
            else:
                secondary.append(text)

        good_sentences = primary + secondary

        if not good_sentences:
            return ''

        # If we have real facts, prefer them over generic templates
        if len(primary) >= 1:
            selected = primary[:3]
        else:
            selected = good_sentences[:3]

        return self._compose_explanation(selected, subject_str)

    def _explain_from_subject(self, subject: str, query: str) -> str:
        """Subject bazlı arama ile AÇIKLAYICI cevap üret."""
        try:
            kb = self.brain.knowledge_base if hasattr(self.brain, 'knowledge_base') else []
        except Exception:
            return ''

        good_sentences = []
        seen = set()

        for fact in kb:
            text = fact.get('text', '').strip()
            if not text or len(text) < 5:
                continue

            text_lower = text.lower()
            if subject not in text_lower:
                continue

            if text_lower in seen:
                continue
            seen.add(text_lower)

            words = text.split()
            if len(words) < 2 or len(words) > 50:
                continue

            # Subject ile başlayan cümleler öncelikli (tanım cümleleri)
            first_word = text_lower.split()[0] if text_lower.split() else ''
            if first_word == subject or first_word.startswith(subject):
                good_sentences.insert(0, text)
            else:
                good_sentences.append(text)

        if not good_sentences:
            return ''

        selected = good_sentences[:3]
        return self._compose_explanation(selected, subject)

    def _generate_example_sentence(self, query: str) -> str:
        """Örnek cümle isteği."""
        try:
            kb = self.brain.knowledge_base if hasattr(self.brain, 'knowledge_base') else []
        except Exception:
            kb = []

        if kb:
            # Kısa, anlamlı cümleleri tercih et
            good = [f.get('text', '') for f in kb if 3 < len(f.get('text', '').split()) < 15]
            if good:
                sentence = random.choice(good)
                return self._clean(f'İşte bir cümle: {sentence}')

        return 'Henüz cümle kuracak yeterli bilgim yok. Bana bir şeyler öğret!'

    def _compose_explanation(self, sentences: List[str], subject: Optional[str]) -> str:
        """Cümleleri doğal bir şekilde birleştir - AÇIKLAYICI cevap."""
        if not sentences:
            return ''

        if len(sentences) == 1:
            # Tek cümle: doğrudan ver
            return self._clean(sentences[0])

        # Birden fazla cümle: opener + connector'larla birleştir
        parts = []
        opener = random.choice(self.OPENERS)

        for i, sent in enumerate(sentences):
            if i == 0:
                parts.append(opener + sent)
            else:
                connector = random.choice(self.CONNECTORS)
                parts.append(connector + sent)

        return self._clean(' '.join(parts))

    def _extract_subject(self, query: str) -> Optional[str]:
        """Sorgudan konu çıkar."""
        patterns = [
            r'(\S+)\s+nedir',
            r'(\S+)\s+ne demek',
            r'(\S+)\s+nasıl',
            r'(\S+)\s+hakkında',
            r'(\S+)\s+kimdir',
            r'(\S+)\s+neler',
            r'(\S+)\s+ne işe',
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                candidate = match.group(1)
                candidate = re.sub(r'[^\wçğıöşüÇĞİÖŞÜ]', '', candidate)
                if candidate and len(candidate) > 2 and candidate not in self.STOP_SUBJECTS:
                    return candidate.capitalize()
        return None

    def _clean(self, text: str) -> str:
        """Metni temizle."""
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            text = text[0].upper() + text[1:]
        if text and text[-1] not in '.!?':
            text += '.'
        return text
