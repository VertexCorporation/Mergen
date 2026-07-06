# -*- coding: utf-8 -*-
"""
Mergen Bio Vektörleyici — Transformer-Free

Dikkat mekanizması (Attention) veya Transformer kullanmaz.
Tamamen yerel: karakter n-gram hash + rastgele projeksiyon.

Neden bu yaklaşım?
- Türkçe morfölojisine doğal uyum: "elmas", "elmasın", "elması"
  tüm ortak n-gram'ları paylaşır — ek temizlemeye gerek yok.
- Kelime haznesi gerektirmez (hash trick).
- Her çalışmada deterministik — aynı metin → aynı vektör.
- CPU'da milisaniyeler içinde çalışır.
"""

import hashlib
import numpy as np
from typing import List


class BioVectorizer:
    """
    Transformer-free, tamamen yerel vektörleyici.

    Pipeline:
        metin
          → karakter n-gram'ları (bigram..4-gram)
          → hash trick (HASH_DIM boyutlu seyrek vektör)
          → sabit rastgele projeksiyon (DIM boyutuna indir)
          → L2 normalizasyon
    """

    DIM      = 512       # Çıkış vektörü boyutu (ChromaDB ile uyumlu)
    HASH_DIM = 131072    # Hash uzayı — 2^17, çakışmayı minimize eder
    SEED     = 2024      # Üretim determinizmi

    def __init__(self, dim: int = None):
        d = dim or self.DIM
        self.dim = d

        # Sabit tohum — model dosyası gerektirmez, her çalışmada aynı
        rng = np.random.RandomState(self.SEED)
        # Johnson-Lindenstrauss lemmasına göre: proj / sqrt(d)
        self._proj = (rng.randn(self.HASH_DIM, d) / np.sqrt(d)).astype(np.float32)

    # ──────────────────────────────────────────────────────────
    #  KARAKTER N-GRAM ÇIKARICI
    # ──────────────────────────────────────────────────────────

    def _ngrams(self, text: str) -> List[str]:
        """
        Metni karakter n-gram'larına böl.
        \x02 (başlangıç) ve \x03 (bitiş) sınır belirteçleri eklenir —
        "ev" ile "sev"deki "ev" n-gram'larını ayırt eder.
        """
        normed = f"\x02{text.lower()}\x03"
        grams: List[str] = []
        for n in range(2, 5):          # bigram, trigram, 4-gram
            for i in range(len(normed) - n + 1):
                grams.append(normed[i : i + n])
        return grams

    # ──────────────────────────────────────────────────────────
    #  HASH ENCODING
    # ──────────────────────────────────────────────────────────

    def _hash_encode(self, text: str) -> np.ndarray:
        """
        Metin → HASH_DIM boyutlu seyrek vektör.

        Her n-gram için çift hash (h1, h2) hesaplanır:
        - h1 → hangi boyuta yazılacağını belirler
        - h2'nin yüksek biti → işareti belirler (±1)
        Bu, hash çakışmalarının birbirini iptal etmesini sağlar.

        BUG-08 FIX: Eski kod n-gram başına 2 adet MD5 çağrısı yapıyordu.
        Tek bir MD5 digest'i 32 hex karakter üretir; ilk 8 ve son 8 karakter
        bağımsız hash değerleri olarak kullanılır — matematiksel kalite aynı,
        işlem sayısı yarıya iner. 10.000 n-gram'lık bir belgede -10.000 çağrı.
        """
        vec = np.zeros(self.HASH_DIM, dtype=np.float32)
        for gram in self._ngrams(text):
            b = gram.encode("utf-8")
            # BUG-08 FIX: Tek MD5 çağrısı — digest'in iki farklı bölümü
            digest = hashlib.md5(b).hexdigest()   # 32 hex karakter
            h1 = int(digest[:8],  16)             # İlk 8 karakter  → konum 1
            h2 = int(digest[-8:], 16)             # Son 8 karakter  → konum 2 / işaret
            sign = 1.0 if (h2 & 1) else -1.0
            vec[h1 % self.HASH_DIM] += sign
            vec[h2 % self.HASH_DIM] += sign * 0.5   # İkincil sinyal

        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec /= norm
        return vec

    # ──────────────────────────────────────────────────────────
    #  ANA ENCODE METodu
    # ──────────────────────────────────────────────────────────

    def encode(
        self,
        texts:         List[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Metin listesi → (N, dim) float32 matris.

        Uzun metinleri cümlelere böler ve parça ortalamalarını alır —
        tek kelimeler ve paragraflar için eşit kalitede sonuç üretir.
        """
        results: List[np.ndarray] = []

        for i, text in enumerate(texts):
            if show_progress and i > 0 and i % 200 == 0:
                print(f"  [BioVec] {i}/{len(texts)} işlendi", flush=True)

            # Cümle/parça bölümleme: uzun metinde ortalama al
            parts = [s.strip() for s in text.replace("\n", ". ").split(". ")
                     if len(s.strip()) > 4]
            if not parts:
                parts = [text]

            # Her parçayı hash'le, sonra ortala
            part_vecs = [self._hash_encode(p) for p in parts[:10]]
            mean_hash = np.mean(part_vecs, axis=0).astype(np.float32)

            # Rastgele projeksiyon: HASH_DIM → dim
            projected = mean_hash @ self._proj

            # L2 normalize
            norm = float(np.linalg.norm(projected))
            if norm > 1e-9:
                projected /= norm

            results.append(projected)

        return np.array(results, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
