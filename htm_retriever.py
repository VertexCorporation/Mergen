# -*- coding: utf-8 -*-
"""
Mergen HTM Retriever — Biyolojik Dikkat Mekanizması

Transformer attention KULLANMAZ.
Numenta HTM-core prensipleri ile:

  1. Seyrek Dağıtık Temsil (SDR — Sparse Distributed Representation)
     Yoğun vektör → %2 aktif bit → ikili maske

  2. SDR Örtüşme Skoru
     AND(sdr_sorgu, sdr_aday) / min(|sdr_sorgu|, |sdr_aday|)
     Softmax olasılığı yok — saf örtüşme sayısı.

  3. Yayılımsal Aktivasyon (Spreading Activation)
     Yüksek eşleşme alan bir kayıt, komşularını da zayıfça aktive eder.
     Semantik yakınlık + yapısal benzerlik bileşimi.

  4. Yanal İnhibisyon (Lateral Inhibition)
     Kazanan nöron, çok benzer komşularını baskılar.
     Tekrarlı / örtüşen sonuçları eleme mekanizması.

Referans: Hawkins, J. et al. — "A Theory of How Columns in the
          Neocortex Enable Learning the Structure of the World" (2017)
"""

import numpy as np
from typing import List


class HTMRetriever:
    """
    Biyolojik SDR örtüşmesi ile aday yeniden sıralama.

    Kullanım:
        retriever = HTMRetriever()
        ranked_indices = retriever.rerank(
            query_vec, candidate_vecs, cosine_scores, top_k=5
        )
    """

    def __init__(
        self,
        sparsity:          float = 0.02,   # Aktif bit oranı — tipik korteks ~%2
        inhibition_thresh: float = 0.72,   # Bu benzerlik üstünde inhibisyon
        spread_decay:      float = 0.80,   # Yayılım her adımda azalma faktörü
        spread_steps:      int   = 2,      # Yayılım adım sayısı
    ):
        self.sparsity          = sparsity
        self.inhibition_thresh = inhibition_thresh
        self.spread_decay      = spread_decay
        self.spread_steps      = spread_steps

    # ──────────────────────────────────────────────────────────
    #  SDR DÖNÜŞÜMÜ
    # ──────────────────────────────────────────────────────────

    def to_sdr(self, vec: np.ndarray) -> np.ndarray:
        """
        Yoğun float vektör → Seyrek ikili maske.
        En büyük %sparsity değerleri aktif (True), geri kalanlar sıfır.
        """
        n_active = max(1, int(len(vec) * self.sparsity))
        sdr = np.zeros(len(vec), dtype=bool)
        top_idx = np.argpartition(vec, -n_active)[-n_active:]
        sdr[top_idx] = True
        return sdr

    # ──────────────────────────────────────────────────────────
    #  ÖRTÜŞME SKORU
    # ──────────────────────────────────────────────────────────

    def overlap(self, sdr_a: np.ndarray, sdr_b: np.ndarray) -> float:
        """
        İki SDR arasındaki örtüşme oranı.
        Daha küçük SDR'ye göre normalize edilir → asimetrik eşleşme.
        """
        n_overlap  = float(np.count_nonzero(sdr_a & sdr_b))
        n_min      = float(min(np.count_nonzero(sdr_a), np.count_nonzero(sdr_b)))
        return n_overlap / n_min if n_min > 0 else 0.0

    # ──────────────────────────────────────────────────────────
    #  YAYILIMSAL AKTİVASYON
    # ──────────────────────────────────────────────────────────

    def spreading_activation(
        self,
        scores:  np.ndarray,
        vecs:    np.ndarray,
    ) -> np.ndarray:
        """
        Yüksek skorlu kayıtlar, benzer kayıtlara zayıf aktivasyon yayar.
        Her adımda decay faktörü uygulanır.

        Biyolojik karşılık: korteks kolonları arasındaki yatay bağlantılar.
        """
        activated = scores.copy()
        n = len(vecs)
        if n < 2:
            return activated

        # L2 normalize edilmiş vektörlerle kosinüs benzerlik matrisi
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        normed = vecs / norms
        sim_mat = normed @ normed.T  # (n, n)
        np.fill_diagonal(sim_mat, 0.0)
        sim_mat = np.clip(sim_mat, 0, None)  # Sadece pozitif benzerlik

        decay = self.spread_decay
        for _ in range(self.spread_steps):
            spread = (sim_mat @ activated) * decay * 0.12
            activated = activated + spread
            decay *= self.spread_decay

        return activated

    # ──────────────────────────────────────────────────────────
    #  YANAL İNHİBİSYON
    # ──────────────────────────────────────────────────────────

    def lateral_inhibition(
        self,
        scores:   np.ndarray,
        vecs:     np.ndarray,
        winner_k: int = None,
    ) -> np.ndarray:
        """
        Kazanan kayıtlar, çok benzer komşularını baskılar.
        Tekrar eden / örtüşen bilgi parçalarını filtreler.

        Biyolojik karşılık: inhibitör internöronlar.
        """
        if len(scores) < 2:
            return scores

        k = winner_k or max(1, len(scores) // 2)
        winner_idx = set(np.argsort(scores)[-k:])

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        normed = vecs / norms

        suppressed = scores.copy()
        for wi in winner_idx:
            for i in range(len(scores)):
                if i in winner_idx:
                    continue
                sim = float(normed[wi] @ normed[i])
                if sim > self.inhibition_thresh:
                    suppressed[i] *= (1.0 - sim * 0.6)

        return suppressed

    # ──────────────────────────────────────────────────────────
    #  ANA YENIDEN SIRALAMA
    # ──────────────────────────────────────────────────────────

    def rerank(
        self,
        query_vec:        np.ndarray,
        candidate_vecs:   np.ndarray,
        cosine_scores:    np.ndarray,
        top_k:            int = 5,
        htm_weight:       float = 0.55,
    ) -> List[int]:
        """
        ChromaDB'nin kosinüs skorlarını HTM biyolojik dikkat ile yeniden sırala.

        Adımlar:
          1. Sorgu ve adayları SDR'ye dönüştür
          2. Her aday için SDR örtüşme skoru hesapla
          3. Yayılımsal aktivasyon uygula
          4. Yanal inhibisyon uygula
          5. Karma skor: (htm_weight × HTM) + ((1-htm_weight) × kosinüs)
          6. En iyi top_k indeksleri döndür
        """
        n = len(candidate_vecs)
        if n == 0:
            return []

        candidate_vecs = np.array(candidate_vecs, dtype=np.float32)
        cosine_scores  = np.array(cosine_scores,  dtype=np.float32)

        # 1. SDR dönüşümleri
        q_sdr = self.to_sdr(query_vec)

        # 2. Örtüşme skorları
        htm_scores = np.array([
            self.overlap(q_sdr, self.to_sdr(cv))
            for cv in candidate_vecs
        ], dtype=np.float32)

        # 3. Yayılımsal aktivasyon
        htm_scores = self.spreading_activation(htm_scores, candidate_vecs)

        # 4. Yanal inhibisyon
        htm_scores = self.lateral_inhibition(htm_scores, candidate_vecs)

        # 5. Normalize + karma skor
        def _norm01(arr: np.ndarray) -> np.ndarray:
            mx = arr.max()
            return arr / mx if mx > 1e-9 else arr

        htm_n = _norm01(htm_scores)
        cos_n = _norm01(cosine_scores)
        final = htm_weight * htm_n + (1.0 - htm_weight) * cos_n

        # 6. Sırala
        ranked = np.argsort(final)[::-1][:top_k]
        return ranked.tolist()
