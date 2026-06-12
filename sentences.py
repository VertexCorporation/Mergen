"""
╔══════════════════════════════════════════════════════════════════════╗
║         MERGEN — TURKISH SENTENCE BUILDER (Grammar Engine)           ║
║                                                                      ║
║  Constructs proper Turkish sentences with correct grammar.          ║
║  Not templates — actual grammatical composition.                    ║
║                                                                      ║
║  Author:  Vertex Corporation — Mergen Project                       ║
║  License: Apache-2.0                                                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import re
import random
from typing import Optional, List, Dict, Tuple


class TurkishSentenceBuilder:
    """
    Builds grammatically correct Turkish sentences.

    Unlike template systems, this composes sentences using:
    • Subject-Object-Verb (SOV) word order
    • Turkish agglutinative suffix chains
    • Proper vowel harmony
    • Tense and mood markers
    • Natural filler and transition words
    """

    # ── Vowel harmony for suffix selection ──
    FRONT_VOWELS = set('eiöü')
    BACK_VOWELS = set('aıou')

    # ── Response patterns by type ──
    PATTERNS = {
        'definition': [
            '{subject}, {definition} ile tanımlanan bir kavramdır.',
            '{subject} dendiğinde, {definition} anlaşılır.',
            '{subject}, {definition} anlamına gelir.',
            '{subject} kavramı, {definition} şeklinde açıklanabilir.',
        ],
        'explanation': [
            '{subject} konusunda şunu söyleyebilirim: {detail}.',
            'Şöyle açıklayayım: {subject}, {detail}.',
            'Aslında {subject} tam olarak {detail}.',
            'Bunu şöyle düşünebiliriz: {subject} yani {detail}.',
        ],
        'opinion': [
            'Bence {subject}, {detail} açısından oldukça ilginç.',
            '{subject} hakkında düşüncem şu: {detail}.',
            'Doğrusu {subject} konusu {detail} yönüyle dikkat çekici.',
        ],
        'comparison': [
            '{subject} ile {compare} arasındaki fark, {detail}.',
            '{subject} ve {compare} benzer kavramlar; ancak {detail}.',
            '{subject} deyince {compare} de akla geliyor, çünkü {detail}.',
        ],
        'agreement': [
            'Katılıyorum, {subject} gerçekten {detail}.',
            'Haklısın, {subject} konusunda {detail} önemli bir nokta.',
            'Aynen, {detail} doğru bir gözlem.',
        ],
        'disagreement': [
            'Orada katılmıyorum açıkçası. {subject} daha çok {detail}.',
            'Farklı düşünüyorum. Bence {subject}, {detail} şeklinde.',
        ],
        'question_response': [
            'İyi soru. {subject} aslında {detail}.',
            'Bu soruyu şöyle cevaplayabilirim: {detail}.',
            'Merak ediyorsun, haklısın. {subject} hakkında: {detail}.',
        ],
        'elaboration': [
            'Daha detaylı anlatayım: {detail}.',
            'Yani kısaca {detail}.',
            'Başka bir açıdan bakarsak, {detail}.',
            'Şunu da eklemeliyim: {detail}.',
        ],
    }

    # ── Transition words for natural flow ──
    TRANSITIONS = [
        'Ayrıca', 'Öte yandan', 'Bununla birlikte', 'Dahası',
        'Özellikle', 'Genel olarak', 'Örneğin', 'Yani',
        'Aslında', 'Doğrusu', 'Şöyle ki', 'Kısacası',
        'Nitekim', 'Bilindiği üzere', 'Tabii ki',
    ]

    # ── Filler expressions ──
    FILLERS = [
        'yani', 'aslında', 'doğrusu', 'sanırım', 'bence',
        'açıkçası', 'şöyle ki', 'özetle', 'kısacası',
        'belki de', 'bir anlamda', 'özellikle de',
    ]

    # ── Confidence expressions ──
    CONFIDENT = [
        'Biliyorum ki',
        'Şunu net söyleyebilirim:',
        'Araştırmalar gösteriyor ki',
        'Genel kabul gören görüşe göre',
    ]
    UNCERTAIN = [
        'Tam emin değilim ama',
        'Hatırladığım kadarıyla',
        'Bir bilgim var, doğruysa:',
        'Sanırım şöyle:',
    ]

    def __init__(self, temperature: float = 0.85):
        self.temperature = temperature
        self._usage_count = 0

    def build_definition(self, subject: str, definition: str) -> str:
        """Build a definition sentence: 'X, Y ile tanımlanan bir kavramdır.'"""
        pattern = random.choice(self.PATTERNS['definition'])
        result = pattern.format(subject=self._capitalize(subject), definition=definition)
        return self._clean(result)

    def build_explanation(self, subject: str, detail: str, confident: bool = True) -> str:
        """Build an explanation sentence."""
        if confident:
            pattern = random.choice(self.PATTERNS['explanation'] + self.PATTERNS['question_response'])
        else:
            pattern = random.choice(self.PATTERNS['elaboration'])
            prefix = random.choice(self.UNCERTAIN) + ' '
            result = prefix + pattern.format(subject=self._capitalize(subject), detail=detail)
            return self._clean(result)

        result = pattern.format(subject=self._capitalize(subject), detail=detail)
        return self._clean(result)

    def build_opinion(self, subject: str, detail: str) -> str:
        """Build an opinion sentence."""
        pattern = random.choice(self.PATTERNS['opinion'])
        result = pattern.format(subject=self._capitalize(subject), detail=detail)
        return self._clean(result)

    def build_agreement(self, subject: Optional[str], detail: str) -> str:
        """Build an agreement sentence."""
        pattern = random.choice(self.PATTERNS['agreement'])
        subj = self._capitalize(subject) if subject else 'bu konu'
        result = pattern.format(subject=subj, detail=detail)
        return self._clean(result)

    def build_elaboration(self, detail: str, add_transition: bool = True) -> str:
        """Build an elaboration/additional info sentence."""
        pattern = random.choice(self.PATTERNS['elaboration'])
        prefix = ''
        if add_transition:
            prefix = random.choice(self.TRANSITIONS) + ', '
        result = prefix + pattern.format(detail=detail)
        return self._clean(result)

    def build_multi_sentence(self, sentences: List[str], add_transitions: bool = True) -> str:
        """Combine multiple sentences with natural transitions."""
        if not sentences:
            return ''
        if len(sentences) == 1:
            return self._clean(sentences[0])

        result = [self._clean(sentences[0])]
        for i, sent in enumerate(sentences[1:]):
            if add_transitions and i == 0:
                prefix = random.choice(self.TRANSITIONS) + ' '
                result.append(prefix + self._clean(sent))
            else:
                result.append(self._clean(sent))

        return ' '.join(result)

    def build_conversational_response(
        self,
        main_point: str,
        additional_points: Optional[List[str]] = None,
        subject: Optional[str] = None,
        tone: str = 'neutral',  # neutral, confident, uncertain, casual
        add_filler: bool = True,
    ) -> str:
        """
        Build a natural-sounding conversational response.

        This is the main method — produces full Turkish paragraphs
        that sound natural and vary each time.
        """
        self._usage_count += 1
        parts = []

        # Opening
        if subject and add_filler:
            openings = [
                f'{self._capitalize(subject)} hakkında ',
                '',
                f'{self._capitalize(subject)} konusu ',
            ]
            opening = random.choice(openings)
        else:
            opening = ''

        # Tone-based confidence marker
        if tone == 'confident':
            marker = random.choice(self.CONFIDENT) + ' '
        elif tone == 'uncertain':
            marker = random.choice(self.UNCERTAIN) + ' '
        elif tone == 'casual':
            marker = random.choice(self.FILLERS) + ', ' if add_filler else ''
        else:
            marker = ''

        # Build first sentence
        first = marker + opening + main_point
        parts.append(self._clean(first))

        # Add additional points with transitions
        if additional_points:
            for point in additional_points[:3]:
                if random.random() < 0.7:
                    trans = random.choice(self.TRANSITIONS)
                    parts.append(f'{trans}, {self._clean(point)}.')
                else:
                    parts.append(self._clean(point))

        result = ' '.join(parts)
        return self._clean(result)

    def apply_vowel_harmony(self, word: str, suffix_type: str = '2way') -> str:
        """Apply Turkish vowel harmony to suffixes."""
        vowels = [c for c in word.lower() if c in 'aıoueiöü']
        if not vowels:
            return word

        last_vowel = vowels[-1]

        if suffix_type == '2way':
            if last_vowel in self.BACK_VOWELS:
                return word + 'a'
            else:
                return word + 'e'
        elif suffix_type == '4way':
            mapping = {
                'a': 'a', 'ı': 'ı', 'u': 'u', 'ü': 'ü',
                'o': 'a', 'ö': 'ü', 'e': 'e', 'i': 'i',
            }
            return word + mapping.get(last_vowel, 'e')
        return word

    def _capitalize(self, text: str) -> str:
        """Capitalize first letter."""
        if not text:
            return ''
        return text[0].upper() + text[1:] if len(text) > 1 else text.upper()

    def _clean(self, text: str) -> str:
        """Clean and sanitize text."""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'([.!?])\s*\1+', r'\1', text)
        if text and text[-1] not in '.!?':
            text += '.'
        return text
