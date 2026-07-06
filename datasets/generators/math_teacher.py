"""
MERGEN V3 - MATH TEACHER
Generates symbolic arithmetic problems.

Unlike V1/V2, this module DOES NOT generate signals.
It only generates the abstract problem (Text + Target).
The Brain's own 'SpikeEncoder' handles the signal conversion.

V7 entegrasyonu: generate_fact_text(), enumerate_all(), format_fact()
metodlari ile MergenBrain_v7 KB egitimi desteklenir.
"""

import random
from typing import Dict, List, Optional, Tuple


SAYI_ADI = {
    0: 'sifir', 1: 'bir', 2: 'iki', 3: 'uc', 4: 'dort', 5: 'bes',
    6: 'alti', 7: 'yedi', 8: 'sekiz', 9: 'dokuz', 10: 'on',
    11: 'onbir', 12: 'oniki', 13: 'onuc', 14: 'ondort', 15: 'onbes',
    16: 'onalti', 17: 'onyedi', 18: 'onsekiz', 19: 'ondokuz', 20: 'yirmi',
}

def _build_sayi_adi():
    """0-99 arasi Turkce ASCII sayi adlari."""
    birler = ['', 'bir', 'iki', 'uc', 'dort', 'bes',
              'alti', 'yedi', 'sekiz', 'dokuz']
    onlar = ['', 'on', 'yirmi', 'otuz', 'kirk', 'elli',
             'altmis', 'yetmis', 'seksen', 'doksan']
    tbl = {}
    for n in range(100):
        if n < 10:
            tbl[n] = birler[n] if n > 0 else 'sifir'
        elif n < 20:
            tbl[n] = 'on' + birler[n % 10] if n % 10 else 'on'
        else:
            o, b = divmod(n, 10)
            tbl[n] = onlar[o] + birler[b] if b else onlar[o]
    return tbl

SAYI_ADI = _build_sayi_adi()

OP_ADI = {'+': 'arti', '-': 'eksi', '*': 'carpi', '/': 'bolu'}
OP_SUBJECT = {'+': 'toplama', '-': 'cikarma', '*': 'carpma', '/': 'bolme'}

# Vocab'da mevcut Unicode fiiller — learn_from_text icin eslesme saglayan anchor'lar.
# Bu kelimeler fact metnine eklenerek Hebbian trace ve KB kaydini mumkun kilar.
OP_VOCAB_ANCHOR = {
    '+': 'toplamak hesaplamak sonuç',
    '-': 'çıkarmak hesaplamak sonuç',
    '*': 'çarpmak hesaplamak sonuç',
    '/': 'bölmek hesaplamak sonuç',
}

TIER_RANGES = {
    'toplama':  [(0, 9), (0, 20), (0, 99)],
    'cikarma':  [(0, 9), (0, 20), (0, 99)],
    'carpma':   [(0, 9), (0, 12), (0, 20)],
    'bolme':    [(1, 9), (1, 12), (1, 20)],
}

TIER_OPS = {
    0: ['toplama'],
    1: ['toplama', 'cikarma'],
    2: ['toplama', 'cikarma', 'carpma'],
    3: ['toplama', 'cikarma', 'carpma', 'bolme'],
}


class MathTeacher:
    def __init__(self, operations=None, tier=0, difficulty=0):
        """
        Args:
            operations: liste ['toplama','cikarma',...] — None ise tier'dan turetilir
            tier: 0-3 arasi, hangi operasyonlarin dahil oldugunu belirler
            difficulty: 0-2 arasi, sayi araligini belirler (0=0-9, 1=0-20, 2=0-99)
        """
        if operations is None:
            operations = TIER_OPS.get(tier, ['toplama'])
        self.operations = operations
        self.tier = tier
        self.difficulty = min(difficulty, 2)
        self._op_symbols = []
        for op_name in self.operations:
            if op_name == 'toplama':
                self._op_symbols.append('+')
            elif op_name == 'cikarma':
                self._op_symbols.append('-')
            elif op_name == 'carpma':
                self._op_symbols.append('*')
            elif op_name == 'bolme':
                self._op_symbols.append('/')

    def generate_sample(self):
        """
        Generates a random arithmetic problem.

        Returns:
            input_signal: None (Handled by Brain's Encoder)
            target_class: int (The correct answer)
            problem_text: str (The question, e.g., "3 + 5")
        """
        op = random.choice(self._op_symbols)
        op_name = OP_SUBJECT[op]
        lo, hi = TIER_RANGES[op_name][self.difficulty]

        if op == '+':
            a = random.randint(lo, hi)
            b = random.randint(lo, hi)
            result = a + b
        elif op == '-':
            a = random.randint(lo, hi)
            b = random.randint(lo, a)
            result = a - b
        elif op == '*':
            a = random.randint(lo, hi)
            b = random.randint(lo, hi)
            result = a * b
        elif op == '/':
            b = random.randint(max(1, lo), hi)
            result = random.randint(lo, hi)
            a = b * result
            # a could exceed tier range but that's fine for division
        else:
            raise ValueError(f"Unknown op: {op}")

        problem_text = f"{a} {op} {b}"
        return None, int(result), problem_text

    def get_batch(self, batch_size=1):
        """Helper for batch testing if needed."""
        targets = []
        texts = []
        for _ in range(batch_size):
            _, tar, txt = self.generate_sample()
            targets.append(tar)
            texts.append(txt)
        return None, targets, texts

    def generate_fact_text(self) -> Tuple[str, int, Dict]:
        """V7 KB uyumlu fact metni uret.

        Returns:
            (fact_text, result, meta) where meta = {'a': int, 'b': int, 'op': str}
        """
        _, result, problem = self.generate_sample()
        parts = problem.split()
        a, op, b = int(parts[0]), parts[1], int(parts[2])
        return self.format_fact({'a': a, 'b': b, 'op': op, 'result': result}), result, {'a': a, 'b': b, 'op': op}

    @staticmethod
    def format_fact(problem: Dict) -> str:
        """V7 KB fact metni.

        Yapi: sayi adlari + operasyon + vocab anchor fiilleri.
        Anchor kelimeler (toplamak, hesaplamak, sonuc) learn_from_text'in
        vocab eslesmesini saglar; sayi adlari recall_raw eslesmesi icindir.
        """
        a, b, op = problem['a'], problem['b'], problem['op']
        result = problem['result']
        a_adi = SAYI_ADI.get(a, str(a))
        b_adi = SAYI_ADI.get(b, str(b))
        r_adi = SAYI_ADI.get(result, str(result))
        op_adi = OP_ADI[op]
        subject = OP_SUBJECT[op]
        anchor = OP_VOCAB_ANCHOR[op]
        return f"{a_adi} {op_adi} {b_adi} esittir {r_adi} {subject} {a} {b} {result} {anchor}"

    def enumerate_all(self) -> List[Dict]:
        """Secili operasyonlar ve difficulty icin tum unique problemleri listele."""
        problems = []
        seen = set()
        for op_name in self.operations:
            if op_name == 'toplama':
                lo, hi = TIER_RANGES['toplama'][self.difficulty]
                for a in range(lo, hi + 1):
                    for b in range(lo, hi + 1):
                        key = ('+', a, b)
                        if key not in seen:
                            seen.add(key)
                            problems.append({'a': a, 'b': b, 'op': '+', 'result': a + b})
            elif op_name == 'cikarma':
                lo, hi = TIER_RANGES['cikarma'][self.difficulty]
                for a in range(lo, hi + 1):
                    for b in range(lo, a + 1):
                        key = ('-', a, b)
                        if key not in seen:
                            seen.add(key)
                            problems.append({'a': a, 'b': b, 'op': '-', 'result': a - b})
            elif op_name == 'carpma':
                lo, hi = TIER_RANGES['carpma'][self.difficulty]
                for a in range(lo, hi + 1):
                    for b in range(lo, hi + 1):
                        key = ('*', a, b)
                        if key not in seen:
                            seen.add(key)
                            problems.append({'a': a, 'b': b, 'op': '*', 'result': a * b})
            elif op_name == 'bolme':
                lo, hi = TIER_RANGES['bolme'][self.difficulty]
                for b in range(max(1, lo), hi + 1):
                    for result in range(lo, hi + 1):
                        a = b * result
                        key = ('/', a, b)
                        if key not in seen:
                            seen.add(key)
                            problems.append({'a': a, 'b': b, 'op': '/', 'result': result})
        return problems
