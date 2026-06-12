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
from typing import List, Optional, Tuple


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


# ──────────────────────────────────────────────────────────────────
#  KURAL TABANLI TÜRKÇE STEMMER (Zeyrek yedeği)
# ──────────────────────────────────────────────────────────────────

# Yaygın Türkçe çekim ekleri — uzundan kısaya sıralı
_TR_SUFFIXES: Tuple[str, ...] = (
    "lerinden", "larından",
    "leriyle",  "larıyla",
    "lerinden", "larından",
    "lerinden", "larından",
    "lerinde",  "larında",
    "lerinin",  "larının",
    "leriyle",  "larıyla",
    "leridir",  "larıdır",
    "lerdir",   "lardır",
    "lerden",   "lardan",
    "leriyle",  "larıyla",
    "lerine",   "larına",
    "lerin",    "ların",
    "lere",     "lara",
    "leri",     "ları",
    "ler",      "lar",
    "nden",     "ndan",
    "nıyla",    "niyle",
    "nında",    "ninde",
    "nının",    "ninin",
    "ından",    "inden",
    "ında",     "inde",
    "ının",     "inin",
    "ınca",     "ince",
    "ndan",     "nden",
    "nda",      "nde",
    "nın",      "nin",
    "nun",      "nün",
    "na",       "ne",
    "nı",       "ni",
    "nu",       "nü",
    "dan",      "den",
    "da",       "de",
    "ya",       "ye",
    "yı",       "yi",
    "yu",       "yü",
    "la",       "le",
    "ın",       "in",
    "un",       "ün",
    "ı",        "i",
    "u",        "ü",
    "a",        "e",
)

_MIN_STEM = 3   # Kök bu uzunluktan kısa olamaz


def _rule_stem(word: str) -> str:
    """Kural tabanlı Türkçe ek kırpıcı."""
    w = word.lower()
    if len(w) <= _MIN_STEM:
        return w
    for suffix in _TR_SUFFIXES:
        if w.endswith(suffix):
            stem = w[: -len(suffix)]
            if len(stem) >= _MIN_STEM:
                return stem
    return w


# ──────────────────────────────────────────────────────────────────
#  SORU KALıPLARI — RAG sorgusunu temizlemek için
# ──────────────────────────────────────────────────────────────────

_QUESTION_PATTERNS = re.compile(
    r'\b(nedir|nelerdir|ne demek(tir)?|ne anlama gelir|nasıldır?|nasıl yapılır|'
    r'nerede(dir)?|ne zaman(dır)?|kimdir|kim(ler)?|niçin|neden|niye|'
    r'kaçtır?|kaç(ane)?|hangisi(dir)?|hangi|bir şey mi|mı|mi|mu|mü)\b',
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

    def lemmatize(self, word: str) -> str:
        """Kelimeyi kök formuna indirge."""
        word = word.strip().lower()
        if not word or len(word) < 2:
            return word

        if self._backend == "zeyrek" and self._analyzer is not None:
            try:
                results = self._analyzer.lemmatize(word)
                # [(surface, [lemma, ...]), ...]
                if results and results[0][1]:
                    return results[0][1][0]
            except Exception:
                pass

        return _rule_stem(word)

    def lemmatize_words(self, words: List[str]) -> List[str]:
        return [self.lemmatize(w) for w in words]

    # ─────────────────────────────────────────
    #  SORGU NORMALİZASYONU
    # ─────────────────────────────────────────

    def normalize_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Ham sorguyu RAG için hazırla.

        Döner:
          - temiz_sorgu (str): soru ekleri kaldırılmış metin
          - kavramlar (List[str]): lemmatize edilmiş anahtar kelimeler
        """
        q = query.strip().lower()

        # Soru kalıplarını kaldır
        q = _QUESTION_PATTERNS.sub(" ", q)
        q = re.sub(r'\s+', ' ', q).strip()

        # Kavramları çıkar (3+ harfli kelimeler)
        raw = re.findall(r'\b[a-zçğışöü]{3,}\b', q, re.UNICODE)
        concepts = [self.lemmatize(w) for w in raw]
        concepts = list(dict.fromkeys(c for c in concepts if len(c) >= 3))

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
