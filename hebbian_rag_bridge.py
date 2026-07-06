# -*- coding: utf-8 -*-
"""
Mergen Hebbian-RAG Köprüsü

RAG verisi indekslenirken aynı bağlamda geçen kavramların
sinaptik bağlarını güçlendirir.

Hebb Kuralı (1949):
  "Birlikte ateşlenen nöronlar birlikte bağlanır."
  dW_ij = η × pre_i × post_j

Uygulama:
  Bir metin parçasında birlikte geçen kelimeler (pencere içinde)
  brain.hebbian_trace vektörünü karşılıklı güçlendirir.
  Mesafe arttıkça etki azalır: Δ = η / (1 + d)
"""

import re
import threading
from typing import Any, List, Optional


class HebbianRAGBridge:
    """
    RAG indekslemesini Hebbian ağırlık güncellemesiyle köprüler.

    Teknik:
    - Metinden vocab kavram ID'leri çıkarılır.
    - Bağlam penceresi (WINDOW=6) içindeki her çift için
      brain.hebbian_trace güncellenir.
    - İşlem arka planda thread ile yürür — RAG indekslemeyi yavaşlatmaz.
    """

    WINDOW_SIZE = 6      # Karşılıklı etki penceresinin yarıçapı
    BASE_LR     = 0.006  # Temel öğrenme hızı — düşük tut, kararlılık için
    MAX_TRACE   = 5.0    # İz doyma sınırı

    def __init__(self, brain: Any, vocab: Any, verbose: bool = False,
                 brain_lock: Optional[Any] = None):
        self.brain   = brain
        self.vocab   = vocab
        self.verbose = verbose
        self._lock   = threading.Lock()
        # BUG-02 FIX: brain.hebbian_trace tensor'ına eş zamanlı erişimi
        # korumak için dışarıdan alınan merkezi lock.
        # None geçilirse kendi yerel lock'u oluşturur (backward compat).
        self._brain_lock = brain_lock if brain_lock is not None else threading.RLock()
        self._count  = 0

    # ──────────────────────────────────────────────────────────
    #  TOPLU GÜNCELLEME (arka plan thread)
    # ──────────────────────────────────────────────────────────

    def update_from_batch(
        self,
        texts:   List[str],
        source:  str = "rag",
        reward:  float = 0.8,
        wait:    bool = False,
        timeout: Optional[float] = None,
    ):
        """
        Metin listesini arka planda işle, sinaptik izleri güncelle.
        Thread-safe ve non-blocking.
        """
        result = {"updated": 0}

        def _worker():
            updated = 0
            for text in texts:
                try:
                    if self._update_chunk(text, reward):
                        updated += 1
                except Exception as e:
                    if self.verbose:
                        print(f"[Mergen Hebbian] Chunk update failed for {source}: {e}")
                    continue
            result["updated"] = updated
            if self.verbose and updated:
                print(
                    f"[Mergen Hebbian] {source}: {updated}/{len(texts)} chunk → "
                    f"toplam {self._count} sinaptik güncelleme"
                )

        t = threading.Thread(target=_worker, daemon=True, name="hebbian-rag")
        t.start()
        if wait:
            t.join(timeout=timeout)
            if t.is_alive():
                raise TimeoutError(f"Hebbian-RAG bridge update timed out for source={source}")
            return result["updated"]
        return t

    def update_single(self, text: str, reward: float = 1.0):
        """Tek metin parçasından senkron güncelleme (dosya öğrenme için)."""
        self._update_chunk(text, reward)

    # ──────────────────────────────────────────────────────────
    #  İÇ GÜNCELLEME MEKANİZMASI
    # ──────────────────────────────────────────────────────────

    def _update_chunk(self, text: str, reward: float) -> bool:
        """
        Bir metin parçasından Hebbian ağırlık güncellemesi yap.
        True döner eğer en az bir güncelleme gerçekleştiyse.
        """
        ids = self._extract_concept_ids(text)
        if len(ids) < 2:
            return False

        with self._lock:
            self._hebb_update(ids, reward)
            self._count += 1
        return True

    def _extract_concept_ids(self, text: str) -> List[int]:
        """
        Metinden vocab'taki kavram ID'lerini çıkar.
        Sadece kelime eşleşmesi — embedding gerektirmez.

        ARCH-05 FIX: Eski regex (r'\\b[a-zçğışöü]{3,}\\b') tire-bileşik kelimeleri
        (yapay-zeka, bağlı-liste), kısa büyük-harf kelimeleri (ML, AI, GPU) ve
        sayıları kaçırıyordu. Vocab'ın ~%20-30'u bu nedenle hiç güncellenmiyordu.
        """
        text_lower = text.lower()

        # Birincil: tire içerebilen bileşik kelimeler ve normal kelimeler
        # [a-zçğışöü0-9]+ → Türkçe/İngilizce harf ve rakam
        # (?:-[a-zçğışöü0-9]+)* → isteğe bağlı tire-ek(ler)
        compound_words = re.findall(
            r'\b[a-z\u00e7\u011f\u0131\u015f\u00f6\u00fc0-9]+(?:-[a-z\u00e7\u011f\u0131\u015f\u00f6\u00fc0-9]+)*\b',
            text_lower
        )

        # İkincil: büyük harf kısaltmalar (GPU, API, ML, AI vb.) — küçük harfe çevrilmiş
        abbreviations = re.findall(r'\b[a-z]{2,4}\b', text_lower)

        # Birleştir ve deduplike et (sıra korumalı)
        seen = set()
        candidates: List[str] = []
        for w in compound_words + abbreviations:
            if w not in seen:
                seen.add(w)
                candidates.append(w)

        h_dim = self._hdim()
        ids: List[int] = []
        for word in candidates:
            cid = self._word_to_id(word)
            if cid is not None and 0 <= cid < h_dim:
                ids.append(cid)

        return ids

    def _word_to_id(self, word: str) -> Optional[int]:
        """Kelimeyi vocab üzerinden ID'ye çevir — çeşitli vocab arayüzlerini dener."""
        try:
            if hasattr(self.vocab, 'word2id'):
                return self.vocab.word2id.get(word)
            if hasattr(self.vocab, 'get_id'):
                return self.vocab.get_id(word)
            if hasattr(self.vocab, 'concept_to_id'):
                return self.vocab.concept_to_id.get(word)
        except Exception as e:
            if self.verbose:
                print(f"[Mergen Hebbian] Vocab lookup failed for '{word}': {e}")
        return None

    def _hdim(self) -> int:
        """Brain Hebbian trace boyutunu güvenli şekilde al."""
        try:
            return self.brain.hebbian_trace.shape[0]
        except Exception as e:
            if self.verbose:
                print(f"[Mergen Hebbian] Hebbian trace size fallback: {e}")
            return 512

    def _hebb_update(self, concept_ids: List[int], reward: float):
        """
        Kavram çiftleri için Hebbian iz güncellemesi.

        dTrace[i] += η × reward / (1 + distance)
        dTrace[j] += η × reward / (1 + distance)

        Yakın kavramlar daha güçlü bağlanır.

        BUG-02 FIX: tüm tensor yazmaları _brain_lock ile korunıyor.
        Bu sayede arka plan RAG thread'i ile ana thread arasında
        data race oluşmayıyor.
        """
        import torch
        # BUG-02 FIX: hebbian_trace'ı okumadan önce lock al.
        with self._brain_lock:
            trace = self.brain.hebbian_trace
            h_dim = trace.shape[0]

            valid = [c for c in concept_ids if 0 <= c < h_dim]
            if len(valid) < 2:
                return

            with torch.no_grad():
                for i in range(len(valid)):
                    win_start = max(0, i - self.WINDOW_SIZE)
                    win_end   = min(len(valid), i + self.WINDOW_SIZE + 1)

                    for j in range(win_start, win_end):
                        if i == j:
                            continue
                        dist  = abs(i - j)
                        delta = self.BASE_LR * reward / (1.0 + dist)

                        c1, c2 = valid[i], valid[j]
                        trace[c1] = torch.clamp(trace[c1] + delta, -self.MAX_TRACE, self.MAX_TRACE)
                        trace[c2] = torch.clamp(trace[c2] + delta, -self.MAX_TRACE, self.MAX_TRACE)

    # ──────────────────────────────────────────────────────────
    #  İSTATİSTİK
    # ──────────────────────────────────────────────────────────

    @property
    def update_count(self) -> int:
        return self._count
