"""
╔══════════════════════════════════════════════════════════════════════╗
║  MERGEN — Vocabulary SOM + STDP Training Script                      ║
║                                                                      ║
║  Bu script, Mergen'in 1416 kavramlik kelime dagarcigini (vocab)       ║
║  SNN motoru olan CorticalColumn uzerinde eğitir.                     ║
║                                                                      ║
║  Egitim Metodu:                                                      ║
║    1. Her kavram Wernicke dual-rail embedding (768-dim) ile temsil   ║
║       edilir.                                                        ║
║    2. L23 katmaninda SOM (Kohonen) topolojik organizasyonu saglanir.  ║
║    3. L4->L23->L5 katmanlari STDP + Dopamin kurali ile guclendirilir.║
║    4. Egitim sonunda model durunu 'mergen_weights.mx'e kaydedilir.    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import torch
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Proje kök dizinini sys.path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import MergenBrain_v7, MergenConfig
from core.mergen_vocab import MergenVocab

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("pip install sentence-transformers gerekli.")
    sys.exit(1)


def main():
    print("=" * 70)
    print("  MERGEN — SOM + STDP Kelime Dagarcigi Egitimi")
    print("=" * 70)

    # ── 1. Brain ve Konfigürasyon Yükle ──
    print("[Egitim] Brain orkestratoru baslatiliyor...")
    config = MergenConfig()
    # Egitim sirasinda CUDA varsa kullan
    config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Brain yukle (DMN thread'lerini baslatmamak icin wake_up() cagrilmayacak)
    mergen = MergenBrain_v7(config=config, verbose=True)
    
    # Once varsa mevcut .mx yukle
    loaded = False
    if hasattr(mergen, 'limbic') and mergen.limbic is not None:
        loaded = mergen.limbic.load_state()
        if loaded:
            print("[Egitim] Mevcut .mx bellek durumlari yuklendi.")
        else:
            print("[Egitim] Mevcut .mx bulunamadi, cortical priors uzerinden baslaniyor.")

    vocab = mergen.vocab
    vocab_size = vocab.size()
    words = [vocab.id_to_word(i) for i in range(vocab_size)]
    print(f"[Egitim] Vocab Boyutu: {vocab_size} kavram")

    # ── 2. Kelimeleri Dual-Rail Embedding Olarak Kodla ──
    print("[Egitim] SentenceTransformer yukleniyor...")
    model = SentenceTransformer(
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        device='cpu' # Bellek tasarrufu icin CPU'da encode et
    )
    print("[Egitim] Kelimeler encode ediliyor...")
    embeddings = model.encode(words, convert_to_tensor=True, normalize_embeddings=True)
    # Dual-rail (768-dim)
    pos = torch.relu(embeddings)
    neg = torch.relu(-embeddings)
    dual_embeddings = torch.cat([pos, neg], dim=-1).to(config.DEVICE)
    print(f"[Egitim] Dual-rail embeddings boyutu: {dual_embeddings.shape}")

    # ── 3. Egitim Parametreleri ──
    # L4 ve L23 katmanlarini donduruyoruz (stabil temsil saglamak icin)
    # L5 katmaninda ise yuksek STDP ogrenme hizi kullaniyoruz (A_ltp=0.1, A_ltd=0.01)
    # Boylece L5, L23'un seyrek spiking temsillerini vocab kelimelerine hizla esler.
    epochs = 8
    som_lr = 0.0  # L23 temsillerinin kaymasini (drift) onlemek icin SOM pasif
    stdp_reward = 1.0
    
    print(f"[Egitim] Parametreler:")
    print(f"  - Epochs: {epochs}")
    print(f"  - STDP Reward: {stdp_reward}")
    print(f"  - Cihaz: {config.DEVICE.upper()}")
    
    engine = mergen.hebbian_engine
    
    # Katman bazli ogrenme hizlarini ayarla ve dondurulmus katmanlarin normalizasyonunu kapat
    # (LTP olmadigi icin normalizasyonun agirliklari ezmesini engeller)
    engine.L4.A_ltp = 0.0
    engine.L4.A_ltd = 0.0
    engine.L4.scaling_speed = 0.0
    
    engine.L23.A_ltp = 0.0
    engine.L23.A_ltd = 0.0
    engine.L23.scaling_speed = 0.0
    
    engine.L5.A_ltp = 0.1
    engine.L5.A_ltd = 0.01
    engine.L5.scaling_speed = 0.001
    
    # Sıfırdan başlanıyorsa L5 ağırlıklarını küçük rastgele değerlerle ilklendir.
    # Priors'tan gelen L5 analog projeksiyon matrisi spiking katmanlarla uyumsuzdur.
    if not loaded:
        print("[Egitim] L5 agirliklari temiz bir baslangic icin kucuk rastgele degerlerle sifirlaniyor...")
        engine.L5.weights.data = torch.rand(1024, vocab_size, device=config.DEVICE) * 0.05
        
    engine.reset_traces()

    # ── 4. Egitim Dongusu ──
    print("\n[Egitim] Egitim basliyor...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Kelimeleri karistir
        indices = list(range(vocab_size))
        random.shuffle(indices)
        
        total_rpe = 0.0
        total_delta_w = 0.0
        
        for step, idx in enumerate(indices):
            # RESET TRACES BEFORE EACH WORD to prevent leakage!
            engine.reset_traces()
            
            word = words[idx]
            pre = dual_embeddings[idx] # (768,)
            
            # Target output (one-hot spike at index)
            post = torch.zeros(vocab_size, device=config.DEVICE)
            post[idx] = 1.0
            
            # Dopamin RPE adaptasyonunu engelle (her kelime ogretiminde RPE = reward olmali)
            if hasattr(engine, '_dopamine'):
                engine._dopamine.value_estimate = 0.0
                
            # CorticalColumn learning step (STDP + Dopamin + SOM)
            telemetry = engine.learning_step(
                pre_spikes=pre,
                post_spikes=post,
                reward=stdp_reward,
                som_lr=som_lr
            )
            
            total_rpe += telemetry.get('rpe', 0.0)
            total_delta_w += telemetry.get('delta_w', 0.0)
            
        epoch_duration = time.time() - epoch_start
        mean_rpe = total_rpe / vocab_size
        mean_dw = total_delta_w / vocab_size
        
        # Her epoch sonu bir telemetri bas
        print(f"  Epoch {epoch+1:2d}/{epochs:2d} | "
              f"Sure: {epoch_duration:.1f}s | "
              f"Ort. RPE: {mean_rpe:.4f} | "
              f"Ort. dW: {mean_dw:.6f}")

    training_duration = time.time() - start_time
    print(f"\n[Egitim] Egitim tamamlandi! Toplam Sure: {training_duration:.1f}s")

    # ── 5. Dogrulama (HIT Rate) ──
    print("\n[Dogrulama] Kararli durum testi (analog forward pass)...")
    hits = 0
    with torch.no_grad():
        for idx in range(vocab_size):
            engine.reset_traces()
            pre = dual_embeddings[idx]
            # Spiking=False ile graded analog aktivasyon kullanarak en yuksek degeri ara
            out = engine.forward(pre, spiking=False)
            pred_idx = torch.argmax(out).item()
            if pred_idx == idx:
                hits += 1
                
    hit_rate = hits / vocab_size
    print(f"  - Dogruluk (HIT Rate): {hits}/{vocab_size} ({hit_rate:.2%})")

    # ── 6. Kaydetme ──
    if hasattr(mergen, 'limbic') and mergen.limbic is not None:
        print(f"\n[Kaydetme] Yeni agirliklar '{config.MX_WEIGHTS_PATH}' dosyasina yaziliyor...")
        success = mergen.limbic.save_state()
        if success:
            print("  [OK] Model basariyla .mx formatinda kaydedildi!")
        else:
            print("  [FAIL] Model kayit hatasi!")
    else:
        print("\n[Kaydetme] Hata: Limbic layer bulunamadi, kaydedilemedi.")

    print("=" * 70)


if __name__ == '__main__':
    # Seed sabitle
    random.seed(42)
    torch.manual_seed(42)
    main()
