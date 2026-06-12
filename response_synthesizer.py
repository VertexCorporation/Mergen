"""
╔══════════════════════════════════════════════════════════════════════╗
║     MERGEN — RESPONSE SYNTHESIZER v4 (Summary Engine)                ║
║                                                                      ║
║  Composes answers by:                                                ║
║    1. Extracting key points from all recalled facts                 ║
║    2. Building a coherent summary (2-4 sentences)                   ║
║    3. Adding supporting details only when relevant                  ║
║                                                                      ║
║  Author:  Vertex Corporation — Mergen Project                       ║
║  License: Apache-2.0                                                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import re
import random
import torch
from typing import Optional, Dict, List, Tuple
from collections import Counter


class ResponseSynthesizer:
    """
    Generates coherent Turkish responses by summarizing knowledge facts
    rather than concatenating them raw.
    """

    SYNONYM_POOLS = {
        'düşünüyorum': ['düşünüyorum', 'fikir yürütüyorum', 'kafa yoruyorum'],
        'biliyorum': ['biliyorum', 'malumum', 'haberim var'],
        'bilmiyorum': ['bilmiyorum', 'emin değilim', 'tam emin olamıyorum'],
        'anladım': ['anladım', 'kavradım', 'idrak ettim'],
        'öğrendim': ['öğrendim', 'kaydettim', 'hafızama aldım'],
        'ilginç': ['ilginç', 'enteresan', 'dikkat çekici'],
        'önemli': ['önemli', 'kritik', 'kilit'],
    }

    INTENT_RESPONSES = {
        'GREETING': [
            'Merhaba! Ben Mergen. Sana nasıl yardımcı olabilirim?',
            'Selam! Dijital beynim hazır, ne konuşmak istersin?',
            'Merhabalar. Seni dinliyorum, ne hakkında sohbet edelim?',
            'Selamlar! Bugün ne keşfetmek istersin?',
        ],
        'IDENTITY': [
            'Ben Mergen. Biyolojiden ilham alan bir dijital beyinim. Hebbian öğrenme ve spike tabanlı nöral ağlarla çalışıyorum.',
            'Adım Mergen. Deneysel bir bilişsel mimariyim. Kendi sözlüğüm ve nöral bağlantılarım var.',
            'Ben Mergen — spike tabanlı çalışan bir yapay zeka. Her etkileşimle öğreniyorum.',
        ],
        'GRATITUDE': [
            'Rica ederim. Her etkileşim nöral ağımı güçlendiriyor.',
            'Ne demek! Konuşmak öğrenmektir.',
            'Ben teşekkür ederim. Bu sohbetle yeni bağlantılar kurdum.',
        ],
        'AFFIRMATION': [
            'Güzel, bu bilgiyi hafızamda pekiştirdim.',
            'Anlaşıldı. Bu yönde nöral bağlantılarımı güçlendirdim.',
            'Tamam, bunu öğrendim. Devam edelim.',
        ],
        'NEGATION': [
            'Anladım, yanlış bir çıkarım yaptım. Nöronlarımı düzenliyorum.',
            'Haklısın. Çıkarımımı düzeltiyorum.',
            'Fark ettim. Bir dahaki sefere daha iyi cevap vereceğim.',
        ],
        'EMOTION': [
            'Hislerini algılıyorum. Biraz daha anlatır mısın?',
            'Bu seni etkilemiş görünüyor. Duygularını değerli buluyorum.',
            'Anlıyorum. Bu konuda ne hissettiğini paylaşman önemli.',
        ],
        'WELLBEING': [
            'Teşekkür ederim, iyiyim! Seninle sohbet etmek güzel.',
            'İyiyim, öğrenmeye devam ediyorum. Sen nasılsın?',
            'Harikayım! Yeni kelimeler öğreniyorum.',
        ],
    }

    # NEW: Knowledge-based response generation
    KNOWLEDGE_RESPONSE_TEMPLATES = {
        'definition': [
            '{subject} hakkında şunu söyleyebilirim: {fact}',
            '{subject} bir kavramdır. Bildiğim kadarıyla: {fact}',
            '{subject} ile ilgili: {fact}',
        ],
        'example': [
            'Örnek olarak: {fact}',
            'Mesela: {fact}',
            'Şöyle bir örnek verebilirim: {fact}',
        ],
        'general': [
            'Bu konuda şunu biliyorum: {fact}',
            'Öğrendiklerime göre: {fact}',
        ],
    }

    QUESTION_PATTERNS = [
        r'\b(nedir|ne demek|nasıl|niçin|neden|kim|nerede|ne zaman|kaç)\b',
        r'\b(what|how|why|who|where|when)\b',
    ]

    STOP_WORDS = {
        'bir', 've', 'veya', 'ama', 'fakat', 'çünkü', 'bu', 'şu', 'de', 'da',
        'ki', 'mi', 'mı', 'mu', 'mü', 'için', 'gibi', 'olan', 'çok', 'daha',
        'en', 'ne', 'nasıl', 'neden', 'niçin', 'nedir', 'ile', 'arasında',
        'bulunur', 'bulunan', 'olarak', 'olarak', 'göre', 'kadar', 'doğru',
        'ait', 'the', 'a', 'an', 'is', 'are', 'was', 'in', 'of', 'to',
        'and', 'or', 'but', 'that', 'this', 'it', 'as', 'what', 'how',
        'why', 'who', 'where', 'when', 'tarafından', 'adlı', 'ise',
        'değil', 'var', 'yok', 'da', 'dahi', 'zaten', 'ancak',
    }

    def __init__(
        self,
        vocab: object,
        brain: object,
        temperature: float = 0.85,
        top_k: int = 30,
        max_response_words: int = 80,
    ):
        self.vocab = vocab
        self.brain = brain
        self.temperature = temperature
        self.top_k = top_k
        self.max_response_words = max_response_words
        self.response_count = 0

    def synthesize(
        self,
        neural_intent: torch.Tensor,
        original_query: str,
        intent: str = 'UNKNOWN',
        subject: Optional[str] = None,
        knowledge_facts: Optional[List[Dict]] = None,
        conversation_context: Optional[Dict] = None,
    ) -> str:
        """Main synthesis pipeline."""
        self.response_count += 1
        if knowledge_facts is None:
            knowledge_facts = []

        # Priority 1: Knowledge-based summary
        if knowledge_facts and len(knowledge_facts) > 0:
            response = self._compose_summary_answer(
                facts=knowledge_facts,
                subject=subject,
                query=original_query,
                intent=intent,
            )
            if response:
                return response

        # Priority 2: Intent-based response
        response = self._intent_response(
            intent=intent,
            subject=subject,
            query=original_query,
        )
        if response:
            return response

        # Priority 3: Fallback
        return self._fallback(subject, query=original_query)

    def _compose_summary_answer(
        self,
        facts: List[Dict],
        subject: Optional[str],
        query: str,
        intent: str,
    ) -> Optional[str]:
        """
        Compose a SUMMARY answer from knowledge facts.

        Steps:
        1. Deduplicate facts
        2. Score sentences by informativeness
        3. Select top sentences for summary
        4. Compose coherent response
        """
        # Deduplicate
        unique = self._deduplicate_facts(facts)
        if not unique:
            return None

        # Extract subject
        subject_str = subject if subject else self._extract_subject(query)

        # Detect query type
        query_lower = query.lower()
        is_def_query = bool(re.search(r'\bnedir\b|\bne demek\b|\bkimdir\b', query_lower))
        is_how_query = bool(re.search(r'\bnasıl\b|\bhow\b', query_lower))
        is_list_query = bool(re.search(r'\bneler\b|\bhangileri\b', query_lower))
        is_who_query = bool(re.search(r'\bkim\b|\bwho\b', query_lower))
        is_example_query = bool(re.search(r'\bö rnek\b|\bcümle\b', query_lower))
        is_action_query = bool(re.search(r'\byap\b|\bver\b|\bshow\b|\bcreate\b', query_lower))

        # Extract all sentences from facts
        all_sentences = self._extract_sentences(unique)

        if not all_sentences:
            return None

        # Score sentences
        scored = self._score_sentences(
            all_sentences, subject_str, query_lower,
            is_def_query, is_how_query, is_list_query, is_who_query,
        )

        # Select sentences for summary
        max_sentences = 3 if is_example_query else 4
        selected = self._select_summary_sentences(scored, query_lower, max_sentences=max_sentences)

        if not selected:
            return None

        # Compose response
        return self._compose_from_sentences(
            selected, subject_str, query_lower, is_def_query, is_example_query, is_action_query
        )

    def _deduplicate_facts(self, facts: List[Dict]) -> List[Dict]:
        """Remove near-duplicate facts."""
        unique = []
        seen = set()
        for f in sorted(facts, key=lambda x: -x.get('relevance', 0)):
            text = f.get('text', '').strip()
            if not text:
                continue
            key = text[:80].lower()
            is_dup = False
            for s in seen:
                tokens_a = set(re.findall(r'\w+', key))
                tokens_b = set(re.findall(r'\w+', s))
                if tokens_a and tokens_b:
                    overlap = len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))
                    if overlap > 0.65:
                        is_dup = True
                        break
            if not is_dup:
                seen.add(key)
                unique.append(f)
        return unique

    def _extract_sentences(self, facts: List[Dict]) -> List[str]:
        """Extract sentences from facts with smart splitting."""
        sentences = []
        seen = set()
        for f in facts:
            text = f.get('text', '').strip()
            if not text:
                continue

            # Smart split: only split on period-space-uppercase or period-space
            # Avoid splitting on abbreviations, formulas, etc.
            parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ])', text)
            for part in parts:
                part = part.strip()
                if len(part) < 10:
                    continue
                # Skip sentence fragments that start with operators or symbols
                if re.match(r'^[\s×÷=+\-·]', part):
                    continue
                # Skip fragments that are clearly cut mid-formula
                if re.match(r'^[×÷=+\-]?\s*\d', part) and len(part) < 30:
                    continue

                # Deduplicate
                key = part[:60].lower()
                if key not in seen:
                    seen.add(key)
                    sentences.append(part)

        return sentences

    def _score_sentences(
        self,
        sentences: List[str],
        subject: Optional[str],
        query_lower: str,
        is_def: bool,
        is_how: bool,
        is_list: bool,
        is_who: bool,
    ) -> List[Tuple[float, str]]:
        """
        Score each sentence by how informative it is for the query.
        """
        scored = []
        subject_lower = subject.lower() if subject else ''
        subject_prefix = subject_lower[:4] if len(subject_lower) >= 4 else subject_lower

        # Extract key terms from query (excluding stop words and question words)
        query_terms = set()
        for w in re.findall(r'\w+', query_lower):
            if w not in self.STOP_WORDS and len(w) > 2:
                query_terms.add(w)
                # Also add stem
                if len(w) > 4:
                    query_terms.add(w[:-2])
                    query_terms.add(w[:-1])

        for sent in sentences:
            sent_lower = sent.lower()
            tokens = set(re.findall(r'\w+', sent_lower))
            word_count = len(tokens)
            score = 0.0

            # 1. Subject match (strong signal)
            if subject_lower:
                if subject_lower in sent_lower:
                    score += 5.0
                    # Bonus ONLY if subject is the FIRST word (subject-focused definition)
                    first_word = sent_lower.split()[0] if sent_lower.split() else ''
                    # Remove punctuation from first word
                    first_word_clean = re.sub(r'[^\wçğıöşüÇĞİÖŞÜ]', '', first_word)
                    if first_word_clean.startswith(subject_prefix) and len(first_word_clean) >= 4:
                        score += 3.0
                elif any(t.startswith(subject_prefix) for t in tokens if len(t) >= 4):
                    score += 2.0

            # 2. Query term overlap
            if query_terms:
                overlap = len(tokens & query_terms)
                score += overlap * 2.0

            # 3. Definition pattern bonus
            if is_def:
                # "X, Y ile ... -an/-en Z'dir" pattern
                if ',' in sent_lower and re.search(r'\bile\b.*\b(?:an|en|yan|yen)\b', sent_lower):
                    score += 8.0
                elif ',' in sent_lower and re.search(r'\w+(?:dır|dir|dur|dür|tır|tir)\b', sent_lower.split(',', 1)[1] if ',' in sent_lower else ''):
                    score += 4.0
                elif re.search(r'\w+(?:dır|dir|dur|dür|tır|tir)\b', sent_lower) and word_count <= 15:
                    score += 2.0

            # 4. How-query bonus (mechanism/process words)
            if is_how:
                if any(w in sent_lower for w in ['çalış', 'kullan', 'yap', 'üre', 'sağla', 'batarya', 'motor']):
                    score += 4.0

            # 5. Who-query bonus (person names)
            if is_who:
                if re.search(r'[A-ZÇĞİÖŞÜ][a-zçğıöşü]+ [A-ZÇĞİÖŞÜ][a-zçğıöşü]+', sent):
                    score += 5.0

            # 6. List-query bonus
            if is_list:
                if any(w in sent_lower for w in ['arasında', 'gibi', 'sayılabilir', 'özellikle']):
                    score += 4.0

            # 7. Information density bonus
            content_words = tokens - self.STOP_WORDS
            if content_words:
                density = len(content_words) / max(1, word_count)
                score += density * 2.0

            # 8. Length preference (medium sentences are best)
            if 5 <= word_count <= 20:
                score += 1.0
            elif word_count > 30:
                score -= 2.0

            # 9. Penalty: brand-heavy sentences (unless query is about brands)
            brand_words = {'toyota', 'bmw', 'mercedes', 'ford', 'volkswagen', 'honda', 'audi'}
            brand_count = len(tokens & brand_words)
            if brand_count >= 3:
                score -= 3.0

            # 10. Penalty: generic filler
            if sent_lower.startswith('bunun yanı sıra') or sent_lower.startswith('ayrıca,'):
                score -= 1.0

            # 11. Penalty: non-definition starters
            non_def_starters = ['ilk ', 'modern ', 'günümüzde', 'son yıllarda',
                               'ayrıca,', 'bunun yanı sıra']
            for starter in non_def_starters:
                if sent_lower.startswith(starter):
                    score -= 2.0
                    break

            scored.append((score, sent))

        return scored

    def _select_summary_sentences(
        self,
        scored: List[Tuple[float, str]],
        query_lower: str,
        max_sentences: int = 4,
    ) -> List[str]:
        """
        Select sentences for summary using greedy selection with diversity.
        """
        if not scored:
            return []

        # Sort by score
        scored.sort(key=lambda x: -x[0])

        selected = []
        selected_tokens = set()

        for score, sent in scored:
            if len(selected) >= max_sentences:
                break

            sent_tokens = set(re.findall(r'\w+', sent.lower()))

            # Diversity check: skip if too similar to already selected
            if selected_tokens:
                overlap = len(sent_tokens & selected_tokens) / max(1, len(sent_tokens))
                if overlap > 0.5:
                    continue

            selected.append(sent)
            selected_tokens.update(sent_tokens)

        return selected

    def _compose_from_sentences(
        self,
        sentences: List[str],
        subject: Optional[str],
        query_lower: str,
        is_def_query: bool,
        is_example_query: bool = False,
        is_action_query: bool = False,
    ) -> str:
        """
        Compose a response from selected sentences.
        Adds natural transitions and proper opening.
        """
        if not sentences:
            return None

        subject_str = subject if subject else 'bu konu'
        subject_clean = re.sub(r'[^\wçğıöşüÇĞİÖŞÜ]', '', subject_str).lower()

        # Detect if this is a conversational request (not just definition)
        is_sentence_request = is_example_query or bool(re.search(r'\bcümle\b|\bö rnek\b', query_lower))
        is_greeting = bool(re.search(r'\bmerhaba\b|\bselam\b|\bnasılsın\b', query_lower))
        is_how_query = bool(re.search(r'\bnasıl\b', query_lower))
        is_action = is_action_query or bool(re.search(r'\byap\b|\bver\b|\bgöster\b|\bcreate\b', query_lower))

        if is_sentence_request:
            # User wants example sentences
            if len(sentences) >= 2:
                return self._clean(f'İşte {subject_clean} ile cümleler: {sentences[0]} {sentences[1]}')
            elif sentences:
                return self._clean(f'İşte bir örnek: {sentences[0]}')

        if is_greeting:
            # Keep greetings warm and conversational
            if len(sentences) >= 1:
                return self._clean(f'{sentences[0]} Size nasıl yardımcı olabilirim?')
            return random.choice(self.INTENT_RESPONSES.get('GREETING', ['Merhaba!']))

        if is_action:
            # Action request - respond with action-oriented language
            if len(sentences) >= 2:
                return self._clean(f'İşte: {sentences[0]} Ayrıca şunu da söyleyebilirim: {sentences[1]}')
            elif sentences:
                return self._clean(f'Bunu yapabiliyorum: {sentences[0]}')

        if len(sentences) == 1:
            sent = sentences[0]
            sent_lower = sent.lower()
            # If sentence already starts with subject, just use it naturally
            first_word = sent_lower.split()[0] if sent_lower.split() else ''
            first_clean = re.sub(r'[^\wçğıöşüÇĞİÖŞÜ]', '', first_word)
            
            if is_def_query:
                if first_clean == subject_clean:
                    return self._clean(sent)
                return self._clean(f'{subject_str} hakkında: {sent}')
            elif is_how_query:
                return self._clean(f'{sent}')
            else:
                return self._clean(f'{subject_str} ile ilgili şunu biliyorum: {sent}')

        # Multiple sentences: compose with natural flow
        parts = []
        for i, sent in enumerate(sentences):
            if i == 0:
                parts.append(sent)
            else:
                # Vary transitions naturally
                if i == 1:
                    transitions = ['Ayrıca, ', 'Bunu şöyle de söyleyebilirim: ', '']
                else:
                    transitions = ['Bir başka örnek: ', 'Ayrıca, ', '']
                parts.append(random.choice(transitions) + sent)

        body = ' '.join(parts)

        # Opening
        if is_def_query:
            openings = [
                f'{subject_str} hakkında: ',
                f'{subject_str} ile ilgili şunları söyleyebilirim: ',
                '',
            ]
        elif is_how_query:
            openings = [
                f'{subject_str} ile ilgili: ',
                '',
            ]
        else:
            openings = [
                f'{subject_str} hakkında bildiklerim: ',
                f'{subject_str} ile ilgili: ',
                '',
            ]

        opening = random.choice(openings)
        return self._clean(opening + body)

    def _intent_response(
        self,
        intent: str,
        subject: Optional[str],
        query: str,
    ) -> Optional[str]:
        """Build coherent response based on intent."""
        subj = subject if subject else self._extract_subject(query)
        query_lower = query.lower()

        # Detect wellbeing questions directly
        if re.search(r'\bnasılsın\b|\bnasılsınız\b|\bnasıl gidiyor\b', query_lower):
            return random.choice(self.INTENT_RESPONSES.get('WELLBEING', ['İyiyim, teşekkürler!']))

        if intent in self.INTENT_RESPONSES:
            return random.choice(self.INTENT_RESPONSES[intent])

        # INQUIRY / COMMAND / UNKNOWN — no knowledge available
        if intent in ('INQUIRY', 'COMMAND', 'UNKNOWN'):
            is_question = self._is_question(query)
            if is_question and subj:
                responses = [
                    f'{subj} hakkında henüz yeterli bilgim yok. "oku:dosya.txt" komutuyla bana bir kaynak okutabilirsin.',
                    f'{subj} konusunu henüz öğrenmedim. İlgili bir metin okutursan hafızama kaydederim.',
                    f'Dürüst olayım: {subj} hakkında henüz bir şey bilmiyorum. Beni beslersen çok daha iyi cevaplar verebilirim.',
                ]
                return random.choice(responses)
            elif subj:
                return f'{subj} hakkında daha fazla bilgi öğrenmem gerekiyor. "oku:dosya.txt" ile bana bir kaynak okutabilirsin.'
            else:
                return 'Ne demek istediğini tam anlayamadım. Biraz daha açık ifade eder misin?'

        return None

    def _fallback(self, subject: Optional[str], query: str) -> str:
        subj = subject if subject else self._extract_subject(query)
        if subj:
            return f'{subj} hakkında henüz yeterli bilgim yok. "oku:dosya.txt" komutuyla bana ilgili bir metin okutabilirsin.'
        return random.choice([
            'Tam anlayamadım. Farklı bir şekilde ifade eder misin?',
            'Bu girdiyi işleyemedim. Tekrar dener misin?',
            'Ne demek istediğini tam kavrayamadım.',
        ])

    def _is_question(self, text: str) -> bool:
        if '?' in text:
            return True
        text_lower = text.lower()
        for pattern in self.QUESTION_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

    def _extract_subject(self, query: str) -> Optional[str]:
        patterns = [
            r'((?:\S+\s+){0,2}\S+)\s+nedir',
            r'((?:\S+\s+){0,2}\S+)\s+ne demek',
            r'((?:\S+\s+){0,2}\S+)\s+nasıl',
            r'((?:\S+\s+){0,2}\S+)\s+hakkında',
            r'((?:\S+\s+){0,2}\S+)\s+kimdir',
            r'((?:\S+\s+){0,2}\S+)\s+neler',
        ]
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                candidate = re.sub(r'[^\wçğıöşüÇĞİÖŞÜ\s]', '', match.group(1)).strip()
                if candidate and len(candidate) > 1:
                    return ' '.join(w.capitalize() for w in candidate.split())

        words = re.findall(r'\w+', query.lower())
        content_words = [w for w in words if w not in self.STOP_WORDS and len(w) > 3]
        if content_words:
            return content_words[0].capitalize()
        return None

    def _clean(self, text: str) -> str:
        # Fix missing space after period (before uppercase letter)
        text = re.sub(r'([.!?])([A-ZÇĞİÖŞÜ])', r'\1 \2', text)
        # Fix missing space after period (before digit)
        text = re.sub(r'([.!?])(\d)', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'([.!?])\s*\1+', r'\1', text)
        text = re.sub(r'\.\s*\.', '.', text)
        if text and text[-1] not in '.!?':
            text += '.'
        if len(text) > 500:
            text = text[:497] + '...'
        return text

    def get_telemetry(self) -> Dict:
        return {
            'total_responses': self.response_count,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'max_response_words': self.max_response_words,
        }
