"""
Google Colab Yüksek Performanslı Büyük Eğitim Scripti (Otonom Parçalı I/O + Senkronize Rüya)

Bu script, 5 TB Google Drive veri kümesi üzerinde I/O darboğazı yaşamadan,
sıradaki Wikipedia parçasını (/content/ yerel diskine) indirerek eğitir,
VRAM taşmalarını önlemek için senkronize uyku döngülerini (DreamModule) yönetir
ve güncel ağırlıkları Google Drive checkpoint klasörüne yedekler.
"""

import os
import sys
import time
import shutil
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learning.cortical_column import CorticalColumn
from core.mergen_vocab import MergenVocab
from cognitive.wernicke_area import WernickeArea
from cognitive.dream import _xor_decrypt_mx, _xor_encrypt_mx, MX_MAGIC

def clean_turkish_text(text: str) -> str:
    """Türkçe karakterleri ve küçük harf dönüşümünü doğru şekilde yapar."""
    text = text.replace('İ', 'i').replace('I', 'ı')
    text = text.lower()
    cleaned = []
    for char in text:
        if char.isalnum() or char.isspace():
            cleaned.append(char)
        else:
            cleaned.append(' ')
    return "".join(cleaned)

def load_weights(mx_path: str, engine: CorticalColumn, device: str, user_id: str = 'default') -> bool:
    """Encrypted .mx ağırlık dosyasını CorticalColumn katmanlarına yükler."""
    if not os.path.exists(mx_path):
        print(f"[Loader] Checkpoint {mx_path} bulunamadı, sıfırdan/priors ile başlanıyor.")
        return False
    try:
        with open(mx_path, 'rb') as f:
            content = f.read()
        if content.startswith(MX_MAGIC):
            encrypted = content[len(MX_MAGIC):]
            json_bytes = _xor_decrypt_mx(encrypted, user_id)
            state = json.loads(json_bytes.decode('utf-8'))
            eng = state.get('engine', {})
            
            if 'L4_weights' in eng and eng['L4_weights'] is not None:
                engine.L4.weights.data = torch.tensor(eng['L4_weights'], device=device).float()
            if 'L23_weights' in eng and eng['L23_weights'] is not None:
                engine.L23.weights.data = torch.tensor(eng['L23_weights'], device=device).float()
            if 'weights' in eng and eng['weights'] is not None:
                engine.L5.weights.data = torch.tensor(eng['weights'], device=device).float()
            if 'L6_weights' in eng and eng['L6_weights'] is not None:
                engine.L6.weights.data = torch.tensor(eng['L6_weights'], device=device).float()
                
            if 'trace_pre' in eng and eng['trace_pre'] is not None:
                engine.trace_pre = torch.tensor(eng['trace_pre'], device=device).float()
            if 'trace_post' in eng and eng['trace_post'] is not None:
                engine.trace_post = torch.tensor(eng['trace_post'], device=device).float()
            if 'firing_rate_ema' in eng and eng['firing_rate_ema'] is not None:
                engine.firing_rate_ema = torch.tensor(eng['firing_rate_ema'], device=device).float()
                
            print(f"[Loader] Checkpoint {mx_path} başarıyla yüklendi.")
            return True
    except Exception as e:
        print(f"[Loader] Yükleme hatası: {e}")
    return False

def save_weights(mx_path: str, engine: CorticalColumn, user_id: str = 'default'):
    """CorticalColumn ağırlıklarını şifreli .mx formatında kaydeder."""
    state = {}
    if os.path.exists(mx_path):
        try:
            with open(mx_path, 'rb') as f:
                content = f.read()
            if content.startswith(MX_MAGIC):
                encrypted = content[len(MX_MAGIC):]
                json_bytes = _xor_decrypt_mx(encrypted, user_id)
                state = json.loads(json_bytes.decode('utf-8'))
        except:
            pass
            
    if not isinstance(state, dict):
        state = {}
        
    state['version'] = '2.0'
    state['timestamp'] = datetime.now().isoformat()
    
    if 'engine' not in state:
        state['engine'] = {}
        
    state['engine']['L4_weights'] = engine.L4.weights.data.cpu().tolist()
    state['engine']['L23_weights'] = engine.L23.weights.data.cpu().tolist()
    state['engine']['weights'] = engine.L5.weights.data.cpu().tolist()
    state['engine']['L6_weights'] = engine.L6.weights.data.cpu().tolist()
    state['engine']['trace_pre'] = engine.trace_pre.cpu().tolist()
    state['engine']['trace_post'] = engine.trace_post.cpu().tolist()
    state['engine']['firing_rate_ema'] = engine.firing_rate_ema.cpu().tolist()
    
    json_bytes = json.dumps(state).encode('utf-8')
    encrypted = _xor_encrypt_mx(json_bytes, user_id)
    
    tmp_path = mx_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        f.write(MX_MAGIC)
        f.write(encrypted)
        
    if os.path.exists(mx_path):
        os.remove(mx_path)
    os.rename(tmp_path, mx_path)
    print(f"[Saver] Ağırlıklar kaydedildi: {mx_path}")

def run_sync_dream(mx_path: str, cycles: int, device: str):
    """VRAM çökmesini önlemek için senkronize uyku/rüya konsolidasyonunu çalıştırır."""
    print(f"\n[Sleep-Sync] Senkronize uyku konsolidasyonu başlatılıyor ({cycles} döngü)...")
    if device == 'cuda':
        torch.cuda.empty_cache()
        
    from cognitive.dream import MergenDream
    try:
        # MergenDream, MX_WEIGHTS_PATH içindeki dosyayı okur/yazar
        dream = MergenDream(
            config_path="config.py",
            verbose=True,
            visualize=False
        )
        dream.sleep(cycles=cycles)
        print("[Sleep-Sync] Rüya konsolidasyonu tamamlandı. Checkpoint güncellendi.")
    except Exception as e:
        print(f"[Sleep-Sync] Uyku döngüsünde hata: {e}")
        
    if device == 'cuda':
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Mergen Google Colab Büyük Eğitim Scripti")
    parser.add_argument("--corpus_dir", type=str, required=True, help="Wikipedia .txt chunk dizini (Drive)")
    parser.add_argument("--local_dir", type=str, default="/content/", help="Yerel hızlı I/O dizini")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Checkpoint Drive dizini")
    parser.add_argument("--mx_path", type=str, default="./mergen_weights.mx", help="Weights dosya yolu")
    parser.add_argument("--vocab_path", type=str, default="./mergen_vocab.json", help="Vocab dosya yolu")
    parser.add_argument("--sleep_interval", type=int, default=500, help="Kaç cümlede bir uyku konsolidasyonu çalışsın")
    parser.add_argument("--sleep_cycles", type=int, default=1000, help="Uyku döngü sayısı")
    parser.add_argument("--som_lr", type=float, default=0.05, help="L23 SOM öğrenme hızı")
    parser.add_argument("--stdp_reward", type=float, default=1.0, help="STDP ödül çarpanı")
    parser.add_argument("--epochs_per_chunk", type=int, default=1, help="Her parça kaç kez dönülsün")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Cihaz (cuda/cpu)")
    args = parser.parse_args()

    # Klasörleri doğrula
    os.makedirs(args.local_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Vocab ve Model Yükle
    print(f"\n[Colab-Train] Model yükleniyor (Cihaz: {args.device.upper()})...")
    if not os.path.exists(args.vocab_path):
        print(f"Hata: {args.vocab_path} bulunamadı.")
        sys.exit(1)
        
    vocab = MergenVocab.load(args.vocab_path)
    vocab_size = vocab.size()
    print(f"[Colab-Train] Kelime Haznesi: {vocab_size} kavram")

    # Wernicke Alanı (Girdi Temsili)
    wernicke = WernickeArea(
        embedding_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        n_neurons=768,
        time_window=50,
        encoding='rate',
        device=args.device,
    )

    # Cortical Column (SNN Beyin)
    engine = CorticalColumn(n_pre=768, n_post=vocab_size, n_hidden=1024, device=args.device)

    # Katman bazlı öğrenme hızlarını ayarla
    engine.L4.A_ltp = 0.0
    engine.L4.A_ltd = 0.0
    engine.L4.scaling_speed = 0.0

    engine.L23.A_ltp = 0.0
    engine.L23.A_ltd = 0.0
    engine.L23.scaling_speed = 0.0

    # L5 & L6 Aktif Öğrenme (STDP + Homeostaz)
    engine.L5.A_ltp = 0.1
    engine.L5.A_ltd = 0.01
    engine.L5.scaling_speed = 0.001

    engine.L6.A_ltp = 0.05
    engine.L6.A_ltd = 0.005
    engine.L6.scaling_speed = 0.001

    # Faz 3: Prediktif geri beslemeyi aktif et
    engine.predictive_feedback = True
    print("[Colab-Train] Faz 3 Prediktif Geri Besleme Aktif Edildi. ✓")

    # Ağırlıkları yükle (yoksa priors'tan yükler)
    loaded = load_weights(args.mx_path, engine, args.device)
    if not loaded:
        # Priors yüklemeyi dene
        priors_path = Path('./mergen_cortical_priors.pt')
        if priors_path.exists():
            print("[Colab-Train] Priors yükleniyor...")
            state = torch.load(priors_path, map_location=args.device, weights_only=True)
            if 'L4_weights' in state:
                engine.L4.weights.data = state['L4_weights'].to(args.device)
            if 'L23_weights' in state:
                engine.L23.weights.data = state['L23_weights'].to(args.device)
            if 'L5_weights' in state:
                engine.L5.weights.data = state['L5_weights'].to(args.device)
            if 'L6_weights' in state:
                engine.L6.weights.data = state['L6_weights'].to(args.device)
        else:
            print("[Colab-Train] Priors bulunamadı. Rastgele ilklendirme yapılıyor.")
            engine.reset_all()

    # 2. Eğitim Durum Kontrolü (Drive)
    state_file_path = os.path.join(args.checkpoint_dir, "colab_training_state.json")
    processed_chunks = []
    if os.path.exists(state_file_path):
        try:
            with open(state_file_path, 'r') as sf:
                state_data = json.load(sf)
                processed_chunks = state_data.get("processed_chunks", [])
                print(f"[Colab-Train] Kaldığı yerden devam ediliyor. İşlenen parça sayısı: {len(processed_chunks)}")
        except Exception as se:
            print(f"[Colab-Train] Durum dosyası okuma hatası: {se}")

    # Chunk listesini al
    all_chunks = sorted([f for f in os.listdir(args.corpus_dir) if f.endswith(".txt")])
    unprocessed_chunks = [c for c in all_chunks if c not in processed_chunks]

    if not unprocessed_chunks:
        print("[Colab-Train] İşlenecek yeni parça kalmadı! Eğitim tamamlandı.")
        return

    print(f"[Colab-Train] Toplam parça: {len(all_chunks)}, Kalan parça: {len(unprocessed_chunks)}")

    # 3. Ana Parçalı I/O Eğitim Döngüsü
    sentence_counter = 0

    for chunk_name in unprocessed_chunks:
        chunk_drive_path = os.path.join(args.corpus_dir, chunk_name)
        chunk_local_path = os.path.join(args.local_dir, chunk_name)

        print(f"\n" + "="*70)
        print(f"[Chunk] Sıradaki parça kopyalanıyor: {chunk_name}")
        print("="*70)

        # Drive -> Yerel Disk I/O Transferi
        shutil.copy2(chunk_drive_path, chunk_local_path)

        # Parça verisini yükle
        with open(chunk_local_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]

        print(f"[Chunk] Yerel diske kopyalandı. Cümle sayısı: {len(sentences):,}")

        # Chunk içi epoch döngüsü
        for epoch in range(args.epochs_per_chunk):
            print(f"\n[Epoch] Parça: {chunk_name} | Epoch: {epoch+1}/{args.epochs_per_chunk}")
            random.shuffle(sentences)

            total_rpe = 0.0
            total_dw = 0.0
            processed_in_epoch = 0

            for idx, raw_sentence in enumerate(sentences):
                # Temiz Türkçe analizi
                cleaned = clean_turkish_text(raw_sentence)
                words = cleaned.split()
                if not words:
                    continue

                # Cümle içi kelime sıralı eğitim adımları
                engine.reset_traces()
                
                for word in words:
                    word_id = vocab.word_to_id.get(word, None)
                    if word_id is None:
                        continue  # Vocab'da yoksa atla
                        
                    # Wernicke perceive: (time_steps, 768)
                    pre_train = wernicke.perceive(word)
                    
                    # Target output (L5 target spike vector)
                    post = torch.zeros(vocab_size, device=args.device)
                    post[word_id] = 1.0

                    # Kelimeyi sun
                    for t in range(pre_train.shape[0]):
                        pre_t = pre_train[t]
                        
                        # Dopamin critic sıfırla (kelime başına RPE hesabı için)
                        if hasattr(engine, '_dopamine'):
                            engine._dopamine.value_estimate = 0.0

                        # SNN Learning Step (L4->L23->L5->L6)
                        telemetry = engine.learning_step(
                            pre_spikes=pre_t,
                            post_spikes=post,
                            reward=args.stdp_reward,
                            som_lr=args.som_lr
                        )
                        total_rpe += telemetry.get('rpe', 0.0)
                        total_dw += telemetry.get('delta_w', 0.0)
                        processed_in_epoch += 1

                sentence_counter += 1

                # 4. Senkronize Uyku Döngüsü (Auto-Sleep)
                if sentence_counter % args.sleep_interval == 0:
                    # 1. Ağırlıkları kaydet
                    save_weights(args.mx_path, engine)
                    # 2. Uyku konsolidasyonunu senkronize çalıştır
                    run_sync_dream(args.mx_path, args.sleep_cycles, args.device)
                    # 3. Konsolide edilmiş ağırlıkları geri yükle
                    load_weights(args.mx_path, engine, args.device)
                    
                    print(f"[Sleep-Sync] Eğitime devam ediliyor... (Cümle: {sentence_counter})")

                # Telemetri gösterimi (her 100 cümlede bir)
                if idx > 0 and idx % 100 == 0:
                    avg_rpe = total_rpe / max(1, processed_in_epoch)
                    avg_dw = total_dw / max(1, processed_in_epoch)
                    print(f"  İlerleme: {idx:>6}/{len(sentences):<6} | Ort. RPE: {avg_rpe:.4f} | Ort. dW: {avg_dw:.6f}")

        # 5. Yerel dosyayı temizle
        if os.path.exists(chunk_local_path):
            os.remove(chunk_local_path)
            print(f"[Chunk] Yerel dosya temizlendi: {chunk_local_path}")

        # 6. Checkpoint Yedekleme (Drive'a yedekle)
        drive_mx_backup = os.path.join(args.checkpoint_dir, f"mergen_weights_chunk_{chunk_name.replace('.txt', '')}.mx")
        shutil.copy2(args.mx_path, drive_mx_backup)
        # Güncel halini de ana ağırlık olarak Drive'da tut
        shutil.copy2(args.mx_path, os.path.join(args.checkpoint_dir, "mergen_weights.mx"))
        print(f"[Backup] Ağırlıklar Drive'a yedeklendi: {drive_mx_backup}")

        # Durumu güncelle ve kaydet
        processed_chunks.append(chunk_name)
        with open(state_file_path, 'w') as sf:
            json.dump({
                "processed_chunks": processed_chunks,
                "last_update": datetime.now().isoformat(),
                "total_sentences_trained": sentence_counter
            }, sf, indent=4)
        print(f"[Backup] Eğitim durumu güncellendi: {state_file_path}")

    print("\n[Colab-Train] Tüm unprocessed chunk'lar başarıyla tamamlandı! ☀️")

if __name__ == '__main__':
    main()
