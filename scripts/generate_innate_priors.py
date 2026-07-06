"""
CorticalColumn için Çok Katmanlı İnnate Priors Üreteci

Eski script: 768×1416 tek matris (HybridHebbianLearner)
Yeni script: L4(768×1024) + L23(1024×1024) + L5(1024×1416)

Strateji:
  1. Her vocab kelimesini SentenceTransformer ile 384-dim → dual-rail → 768-dim encode et
  2. L4: Rastgele ortogonal projeksiyon (768→1024) — Wernicke uzayını genişletir
  3. Vocab kelimelerini L4 üzerinden 1024-dim uzaya projekte et
  4. L5: Bu projeksiyonların transpozesi (1024×1416) — iç uzay→vocab eşlemesi
  5. L23: Birim matris + küçük gürültü (1024×1024) — başlangıçta identity

Sonuç: input("yercekimi") → L4 → L23 → L5 → en yüksek aktivasyon: "yercekimi" nöronu
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learning.cortical_column import CorticalColumn
from core.mergen_vocab import MergenVocab

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("pip install sentence-transformers gerekli.")
    sys.exit(1)


def main():
    # ── Config ──
    N_PRE = 768       # Wernicke dual-rail boyutu
    N_HIDDEN = 1024   # CorticalColumn iç katman genişliği
    OUTPUT_PATH = './mergen_cortical_priors.pt'
    LEGACY_PATH = './mergen_innate_priors.pt'  # Geriye dönük uyumluluk

    # ── 1. Vocab yükle ──
    vocab_path = './mergen_vocab.json'
    if not os.path.exists(vocab_path):
        print(f"Hata: {vocab_path} bulunamadı.")
        return
    vocab = MergenVocab.load(vocab_path)
    words = [vocab.id_to_word(i) for i in range(vocab.size())]
    n_post = vocab.size()
    print(f"[Priors] Vocab: {n_post} kelime")

    # ── 2. Semantic embedding üret ──
    print("[Priors] SentenceTransformer yükleniyor...")
    model = SentenceTransformer(
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        device='cpu'
    )
    print("[Priors] Kelimeler encode ediliyor...")
    embeddings = model.encode(words, convert_to_tensor=True, normalize_embeddings=True)
    # embeddings: (n_post, 384)

    # Dual-Rail: pos/neg ayrımı → 768 boyut
    pos = torch.relu(embeddings)
    neg = torch.relu(-embeddings)
    dual = torch.cat([pos, neg], dim=-1)  # (n_post, 768)
    print(f"[Priors] Dual-rail embeddings: {dual.shape}")

    # ── 3. L4 Priors: Ortogonal projeksiyon (768→1024) ──
    # QR decomposition ile yarı-ortogonal matris üret
    random_matrix_t = torch.randn(N_HIDDEN, N_PRE)
    Q_t, _ = torch.linalg.qr(random_matrix_t)
    # Q_t: (1024, 768)
    L4_weights = Q_t.T  # (768, 1024) — yarı-ortogonal projeksiyon

    # 0-1 aralığına normalize et (SNN ağırlıkları pozitif olmalı)
    L4_weights = (L4_weights - L4_weights.min()) / (L4_weights.max() - L4_weights.min())
    L4_weights *= 0.5  # Max 0.5 — başlangıç ağırlıkları ılımlı
    print(f"[Priors] L4 weights: {L4_weights.shape}, range: [{L4_weights.min():.3f}, {L4_weights.max():.3f}]")

    # ── 4. Vocab kelimelerini L4 uzayına projekte et ──
    # dual: (n_post, 768), L4: (768, 1024)
    # Her kelimenin 1024-dim iç temsili:
    projected = torch.matmul(dual, L4_weights)  # (n_post, 1024)
    # Normalize et
    projected = F.normalize(projected, p=2, dim=1)

    # ── 5. L5 Priors: Spiking katmanlarla uyumlu küçük rastgele değerler (1024→n_post) ──
    # FAZ 2 NOTU: Eski analog projeksiyon transpozesi spiking eşikleme altında
    # hatalı eşleşmelere yol açıyordu. Çıkış katmanı temiz bir başlangıç için 
    # küçük rastgele ağırlıklarla başlatılır; STDP eğitimde doğru bağlantıları kurar.
    L5_weights = torch.rand(N_HIDDEN, n_post) * 0.05
    print(f"[Priors] L5 weights: {L5_weights.shape}, range: [{L5_weights.min():.3f}, {L5_weights.max():.3f}]")

    # ── 6. L23 Priors: SOM-friendly küçük uniform random (1024→1024) ──
    # FAZ 2 NOTU: Identity matrix (eye*1.5) SOM topolojisini kilitler —
    # her nöron aynı giriş boyutunu güçlendirir, BMU seçimi anlamlı olmaz.
    # Küçük uniform random ile her nöron farklı başlangıç noktasında:
    # SOM öğrenmesi topolojik organizasyonu kendiliğinden oluşturabilir.
    L23_weights = torch.rand(N_HIDDEN, N_HIDDEN) * 0.3
    print(f"[Priors] L23 weights: {L23_weights.shape}, range: [{L23_weights.min():.3f}, {L23_weights.max():.3f}]")

    # ── 6b. L6 Priors: Geri besleme tahmini için küçük uniform random (1024→768) ──
    L6_weights = torch.rand(N_HIDDEN, N_PRE) * 0.05
    print(f"[Priors] L6 weights: {L6_weights.shape}, range: [{L6_weights.min():.3f}, {L6_weights.max():.3f}]")

    # ── 7. Doğrulama: Birkaç kelime test et ──
    print("\n[Priors] Doğrulama (top-3 aktivasyon):")
    # CorticalColumn oluştur ve ağırlıkları yükle (k-WTA ve spike eşiklerini kullanmak için)
    col = CorticalColumn(n_pre=N_PRE, n_post=n_post, n_hidden=N_HIDDEN, device='cpu')
    col.L4.weights.data = L4_weights.clone()
    col.L23.weights.data = L23_weights.clone()
    col.L5.weights.data = L5_weights.clone()
    col.L6.weights.data = L6_weights.clone()

    test_words = ['merhaba', 'yerçekimi', 'kitap', 'matematik', 'türkiye']
    for word in test_words:
        idx = vocab.word_to_id.get(word, None)
        if idx is None:
            print(f"  {word}: vocab'da yok, atlandı")
            continue
        # Simüle et: word → dual-rail → L4 → L23 → L5 (k-WTA aktif)
        word_embed = dual[idx]  # (768,)
        
        # word_embed'den spike üretelim (rate-like input simülasyonu)
        # Poisson spike yerine, direkt analog değerleri spiking-activation threshold'una göre besleyelim veya doğrudan forward pass çağıralım
        # CorticalColumn.forward() girdinin analog değerlerini ve surrogate spike'ları kullanıyor
        with torch.no_grad():
            out = col.forward(word_embed, spiking=False)
            
        top3 = torch.topk(out, 3)
        top3_words = [vocab.id_to_word(i) for i in top3.indices.tolist()]
        top3_vals = [f"{v:.3f}" for v in top3.values.tolist()]
        hit = "HIT" if idx in top3.indices.tolist() else "miss"
        print(f"  {word} [{hit}] -> {', '.join(f'{w}({v})' for w, v in zip(top3_words, top3_vals))}")

    # ── 8. Kaydet ──
    state = {
        'version': '2.0',
        'architecture': 'CorticalColumn',
        'n_pre': N_PRE,
        'n_hidden': N_HIDDEN,
        'n_post': n_post,
        'L4_weights': L4_weights,
        'L23_weights': L23_weights,
        'L5_weights': L5_weights,
        'L6_weights': L6_weights,
    }
    torch.save(state, OUTPUT_PATH)
    print(f"\n[Priors] Kaydedildi: {OUTPUT_PATH}")
    print(f"  L4:  {L4_weights.shape}")
    print(f"  L23: {L23_weights.shape}")
    print(f"  L5:  {L5_weights.shape}")
    print(f"  L6:  {L6_weights.shape}")

    # Geriye dönük uyumluluk: eski format L5 weights'i de kaydet
    # (brain.py'deki eski yükleme yolu bunu kontrol ediyor)
    torch.save(L5_weights, LEGACY_PATH)
    print(f"  Legacy (L5 only): {LEGACY_PATH}")


if __name__ == '__main__':
    main()
