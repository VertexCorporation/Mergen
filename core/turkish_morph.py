# -*- coding: utf-8 -*-
"""
Mergen Türkçe Morfoloji & UTF-8 Güvencesi

Zeyrek (Python Zemberek portu) tercihli; yoksa kural tabanlı stemmer.

Sorumluluklar:
  1. Soru normalizasyonu — "Elmasın kimyası nedir?" → "elmas kimya"
  2. Lemmatizasyon — "elmasların" → "elmas"
  3. UTF-8 garantisi — stdout/stderr yeniden sarmalanır
  4. Terminal karakter hatasız çıktı
"""

import io
import re
import sys
from typing import List, Optional, Tuple, Any


# ──────────────────────────────────────────────────────────────────
#  UTF-8 STDOUT SARMALAMA — modül yüklendiğinde hemen çalışır
# ──────────────────────────────────────────────────────────────────

def _patch_utf8_streams():
    """
    stdout ve stderr'i UTF-8 + replace hata modu ile yeniden sarmala.
    Windows terminallerde (cp1254, cp850 vb.) karakter bozulmasını önler.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        buf = getattr(stream, "buffer", None)
        if buf is None:
            continue
        try:
            patched = io.TextIOWrapper(
                buf,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
            setattr(sys, stream_name, patched)
        except Exception:
            pass


_patch_utf8_streams()


def turkish_lower(text: str) -> str:
    """Türkçe karakter duyarlı küçük harfe dönüştürme fonksiyonu (I -> ı, İ -> i)."""
    if not text:
        return ""
    table = str.maketrans({
        'I': 'ı',
        'İ': 'i',
    })
    return text.translate(table).lower()


# ──────────────────────────────────────────────────────────────────
#  KURAL TABANLI TÜRKÇE STEMMER (Zeyrek yedeği)
# ──────────────────────────────────────────────────────────────────

# Yaygın Türkçe çekim, bildirme, ek-fiil ve zamir ekleri
_RAW_SUFFIXES = [
    # Çoğul + Durum / İyelik Kombinasyonları (Uzunlar)
    "lerinden", "larından", "leriyle", "larıyla", "lerinde", "larında",
    "lerinin", "larının", "leridir", "larıdır", "lerdir", "lardır",
    "lerden", "lardan", "lerine", "larına", "lerin", "ların", "lere", "lara",
    "leri", "ları", "ler", "lar", "ndaki", "ndeki", "lardaki", "lerdeki",
    
    # İyelik (Possessives)
    "imiz", "ımız", "umuz", "ümüz",
    "iniz", "ınız", "unuz", "ünüz",
    "inin", "ının", "unun", "ünün",
    "nin", "nın", "nun", "nün",
    "im", "ım", "um", "üm",
    "in", "ın", "un", "ün",
    "si", "sı", "su", "sü",
    "i", "ı", "u", "ü",
    
    # Durum (Cases)
    "nden", "ndan", "inden", "ından", "den", "dan", "ten", "tan",
    "nde", "nda", "inde", "ında", "de", "da", "te", "ta",
    "niyle", "nıyla", "iyle", "ıyla", "le", "la",
    "ne", "na", "ye", "ya", "e", "a",
    "ni", "nı", "nü", "nu", "yi", "yı", "yü", "yu",
    
    # Bildirme / Ek-Fiil / Zamir (Copulas / Pronouns)
    "dirler", "dırlar", "durler", "durlar", "dürler", "türler", "turlar", "tırlar", "tirler",
    "siniz", "sınız", "sunuz", "sünüz",
    "sin", "sın", "sun", "sün",
    "dir", "dır", "dur", "dur", "dür", "tür", "tur", "tır", "tir",
    "daki", "deki", "ki", "ken", "ince", "ınca"
]

# Uzundan kısaya doğru otomatik sırala
_TR_SUFFIXES: Tuple[str, ...] = tuple(sorted(list(set(_RAW_SUFFIXES)), key=len, reverse=True))

_MIN_STEM = 3   # Kök bu uzunluktan kısa olamaz


def _vocab_contains_with_mutation(stem: str, vocab: Any) -> Optional[str]:
    """Ünsüz yumuşaması durumunu restore ederek vocab kontrolü yapar (kitab -> kitap)."""
    if vocab.contains(stem):
        return stem
    if not stem:
        return None
    mutations = {
        'b': 'p',
        'c': 'ç',
        'd': 't',
        'ğ': 'k',
        'g': 'k'
    }
    last_char = stem[-1]
    if last_char in mutations:
        restored = stem[:-1] + mutations[last_char]
        if vocab.contains(restored):
            return restored
    return None


def _rule_stem(word: str, vocab: Optional[Any] = None) -> str:
    """Kural tabanlı Türkçe ek kırpıcı (yinelemeli, mutation ve vocab duyarlı)."""
    w = turkish_lower(word)
    if len(w) <= _MIN_STEM:
        return w

    # Eğer kelime zaten vocab içindeyse, hiç dokunma!
    if vocab is not None:
        matched_root = _vocab_contains_with_mutation(w, vocab)
        if matched_root is not None:
            return matched_root

    changed = True
    stripped_single_vowel = False
    while changed and len(w) > _MIN_STEM:
        changed = False
        for suffix in _TR_SUFFIXES:
            is_single = (len(suffix) == 1)
            if is_single:
                # Tek karakterli ekleri (ı, i, u, ü, a, e) sadece kelime boyu >= 4 ise, daha önce tekli silinmediyse
                # ve sadece yeni kök vocab'de bir karşılık buluyorsa sil!
                if stripped_single_vowel or len(w) < 4:
                    continue
                if vocab is None or _vocab_contains_with_mutation(w[:-1], vocab) is None:
                    continue

            if w.endswith(suffix):
                stem = w[: -len(suffix)]
                if len(stem) >= _MIN_STEM:
                    # Kök kelime dağarcığında bulunursa (yumuşama dahil) hemen dur ve onu dön!
                    if vocab is not None:
                        matched_root = _vocab_contains_with_mutation(stem, vocab)
                        if matched_root is not None:
                            return matched_root

                    w = stem
                    if is_single:
                        stripped_single_vowel = True
                    changed = True
                    break
    return w


# ──────────────────────────────────────────────────────────────────
#  SORU KALıPLARI — RAG sorgusunu temizlemek için
# ──────────────────────────────────────────────────────────────────

_QUESTION_PATTERNS = re.compile(
    r'\b(nedir|nelerdir|ne demek(tir)?|ne anlama gelir|nasildir?|nasıldır?|nasıl yapılır|'
    r'nasıl|nasil|nerede(dir)?|ne zaman(dır)?|kimdir|kim(ler)?|nicin|niçin|neden|niye|'
    r'kactir?|kaçtır?|kac(ane)?|kaç(ane)?|hangisi(dir)?|hangi|bir şey mi|mı|mi|mu|mü)\b',
    re.IGNORECASE | re.UNICODE,
)


# ──────────────────────────────────────────────────────────────────
#  ANA SINIF
# ──────────────────────────────────────────────────────────────────

class TurkishMorph:
    """
    Türkçe morfolojik analiz motoru.

    Öncelik sırası:
      1. Zeyrek (pip install zeyrek) — Python Zemberek portu
      2. Kural tabanlı stemmer — her zaman çalışır, kurulum gerektirmez

    Özellikler:
      - Lemmatizasyon: "elmasların" → "elmas"
      - Soru normalizasyonu: sorgu kavramlarını çıkar
      - UTF-8 güvenli yazdırma
    """

    def __init__(self, verbose: bool = False):
        self._analyzer = None
        self._backend  = "rule"
        self._init_zeyrek(verbose)

    def _init_zeyrek(self, verbose: bool):
        try:
            import zeyrek
            self._analyzer = zeyrek.MorphAnalyzer()
            self._backend  = "zeyrek"
            if verbose:
                print("[Mergen Morph] Zeyrek (Python Zemberek) aktif.")
        except ImportError:
            if verbose:
                print("[Mergen Morph] Zeyrek yüklü değil → kural tabanlı stemmer kullanılıyor.")
                print("               Kurulum: pip install zeyrek")
        except Exception as e:
            if verbose:
                print(f"[Mergen Morph] Zeyrek başlatılamadı ({e}) → kural tabanlı.")

    # ─────────────────────────────────────────
    #  LEMMATIZASYON
    # ─────────────────────────────────────────

    def lemmatize(self, word: str, vocab: Optional[Any] = None) -> str:
        """Kelimeyi kök formuna indirge."""
        word = turkish_lower(word.strip())
        if not word or len(word) < 2:
            return word

        if self._backend == "zeyrek" and self._analyzer is not None:
            try:
                results = self._analyzer.lemmatize(word)
                # [(surface, [lemma, ...]), ...]
                if results and results[0][1]:
                    lemma = results[0][1][0]
                    if vocab is not None and vocab.contains(lemma):
                        return lemma
                    return lemma
            except Exception:
                pass

        return _rule_stem(word, vocab=vocab)

    def lemmatize_words(self, words: List[str], vocab: Optional[Any] = None) -> List[str]:
        return [self.lemmatize(w, vocab=vocab) for w in words]

    # ─────────────────────────────────────────
    #  SORGU NORMALİZASYONU
    # ─────────────────────────────────────────

    def normalize_query(self, query: str, vocab: Optional[Any] = None) -> Tuple[str, List[str]]:
        """
        Ham sorguyu RAG için hazırla.

        Döner:
          - temiz_sorgu (str): soru ekleri kaldırılmış metin
          - kavramlar (List[str]): lemmatize edilmiş anahtar kelimeler
        """
        q = turkish_lower(query.strip())

        # Soru kalıplarını kaldır
        q = _QUESTION_PATTERNS.sub(" ", q)
        q = re.sub(r'\s+', ' ', q).strip()

        # Kavramları çıkar (3+ harfli kelimeler, sayılar veya kısa sözlük kelimeleri)
        raw_words = re.findall(r'\b[a-zçğışöü0-9]+\b', q, re.UNICODE)
        concepts = []
        for w in raw_words:
            lemma = self.lemmatize(w, vocab=vocab)
            if len(lemma) >= 3 or lemma.isdigit():
                concepts.append(lemma)
            elif vocab is not None and hasattr(vocab, 'contains') and vocab.contains(lemma):
                concepts.append(lemma)
            elif vocab is not None and hasattr(vocab, 'word_to_id') and lemma in vocab.word_to_id:
                concepts.append(lemma)
        concepts = list(dict.fromkeys(concepts))

        return q, concepts

    # ─────────────────────────────────────────
    #  YARDIMCI
    # ─────────────────────────────────────────

    def is_turkish(self, text: str) -> bool:
        """Metinde Türkçe karakterler var mı?"""
        return bool(re.search(r'[çğışöüÇĞİŞÖÜ]', text))

    @property
    def backend(self) -> str:
        return self._backend

    # ─────────────────────────────────────────
    #  UTF-8 GÜVENLİ ÇIKTI
    # ─────────────────────────────────────────

    @staticmethod
    def safe_print(text: str, end: str = "\n"):
        """
        Herhangi bir terminal encoding'inde güvenle yazdır.
        UnicodeEncodeError yerine '?' karakteri kullanır.
        """
        try:
            print(text, end=end, flush=True)
        except UnicodeEncodeError:
            fallback = text.encode("utf-8", errors="replace").decode("utf-8")
            print(fallback, end=end, flush=True)
        except Exception:
            print(repr(text), end=end, flush=True)
