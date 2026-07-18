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

def train_file(file_path: str, engine: CorticalColumn, wernicke: WernickeArea, vocab: MergenVocab, args, vocab_size: int, device: str) -> int:
    """Tek bir dosyayı (core curriculum veya wiki partı) eğitir."""
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"[Eğitim] Cümle sayısı: {len(sentences):,}")
    
    # Sözlük kelimelerini semantik çıkarım için önceden encode et
    vocab_words = [vocab.id_to_word(i) for i in range(vocab_size)]
    with torch.no_grad():
        vocab_embeddings = wernicke.encoder.encode(
            vocab_words, 
            convert_to_tensor=True, 
            normalize_embeddings=True,
            show_progress_bar=False
        ).to(device)

    sentence_counter = 0

    for epoch in range(args.epochs_per_chunk):
        print(f"\n{'='*60}")
        print(f"  EPOCH {epoch+1}/{args.epochs_per_chunk} | Dosya: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        random.shuffle(sentences)

        total_rpe = 0.0
        total_dw = 0.0
        processed_in_epoch = 0

        for idx, raw_sentence in enumerate(sentences):
            cleaned = clean_turkish_text(raw_sentence)
            words = cleaned.split()
            if not words:
                continue

            engine.reset_traces()

            word_details = []
            skipped_words = []

            for word in words:
                word_id = vocab.word_to_id.get(word, None)
                inference_made = False
                best_match_word = None
                best_sim = 0.0

                if word_id is None:
                    # Semantik çıkarım (kendi kendine eşleştirme)
                    with torch.no_grad():
                        word_emb = wernicke.encoder.encode(
                            [word],
                            convert_to_tensor=True,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        ).to(device).squeeze(0)
                        
                        similarities = torch.matmul(vocab_embeddings, word_emb)
                        best_idx = torch.argmax(similarities).item()
                        best_sim = similarities[best_idx].item()
                        
                        if best_sim >= 0.50:
                            word_id = best_idx
                            best_match_word = vocab_words[best_idx]
                            inference_made = True
                        else:
                            skipped_words.append((word, best_sim))
                            continue
                    
                # Ağırlık takibi için L5 ağırlıklarını kopyala
                w_before = engine.L5.weights.data.clone()

                pre_train = wernicke.perceive(word)
                post = torch.zeros(vocab_size, device=device)
                post[word_id] = 1.0

                word_rpe = 0.0
                steps_count = 0

                for t in range(pre_train.shape[0]):
                    pre_t = pre_train[t]
                    
                    if hasattr(engine, '_dopamine'):
                        engine._dopamine.value_estimate = 0.0

                    telemetry = engine.learning_step(
                        pre_spikes=pre_t,
                        post_spikes=post,
                        reward=args.stdp_reward,
                        som_lr=args.som_lr
                    )
                    word_rpe += telemetry.get('rpe', 0.0)
                    steps_count += 1
                    
                    total_rpe += telemetry.get('rpe', 0.0)
                    total_dw += telemetry.get('delta_w', 0.0)
                    processed_in_epoch += 1

                if steps_count > 0:
                    w_after = engine.L5.weights.data
                    max_dw = (w_after - w_before).abs().max().item()
                    max_ltp = (engine.L5.trace_pre.max() * engine.L5.A_ltp).item()
                    max_ltd = (engine.L5.trace_post.max() * engine.L5.A_ltd).item()
                    l23_rate = engine.L23.firing_rate_ema.mean().item() * 100.0
                    avg_w_rpe = word_rpe / steps_count
                    word_details.append({
                        'word': word,
                        'rpe': avg_w_rpe,
                        'dw': max_dw,
                        'ltp': max_ltp,
                        'ltd': max_ltd,
                        'rate': l23_rate,
                        'inference_made': inference_made,
                        'best_match_word': best_match_word,
                        'best_sim': best_sim
                    })

            sentence_counter += 1

            # ─── Her Cümle İçin Canlı Log Bloğu ───────────────────────────────
            avg_rpe_so_far = total_rpe / max(1, processed_in_epoch)
            avg_dw_so_far  = total_dw  / max(1, processed_in_epoch)

            print(f"\n┌─ Cümle {idx+1:>4}/{len(sentences)} │ Epoch {epoch+1}/{args.epochs_per_chunk} {'─'*25}")
            print(f"│ Girdi   : \"{raw_sentence}\"")

            if word_details:
                print(f"│ Öğrenme ({len(word_details)} kelime):")
                for wd in word_details:
                    surprise = " ⚡SÜRPRİZ" if wd['rpe'] > 0.4 else "          "
                    if wd['inference_made']:
                        word_str = f"\"{wd['word']}\"→\"{wd['best_match_word']}\"({wd['best_sim']:.2f})"
                    else:
                        word_str = f"\"{wd['word']}\""
                    print(f"│   {word_str:<42} RPE:{wd['rpe']:.3f}{surprise}  dW:{wd['dw']:.5f}  L23:{wd['rate']:.1f}%")
            else:
                print(f"│ Öğrenme : (bu cümlede öğrenilebilir kelime bulunamadı)")

            if skipped_words:
                skipped_strs = ', '.join(f'"{w}"(sim:{s:.2f})' for w, s in skipped_words)
                print(f"│ Sözlük Dışı (çıkarım başarısız): {skipped_strs}")

            # Cümle geneli kognitif yansıma
            if word_details:
                engine.reset_traces()
                try:
                    with torch.no_grad():
                        sent_spikes = wernicke.perceive(raw_sentence)
                        outputs = [engine.forward(sent_spikes[t], spiking=False) for t in range(sent_spikes.shape[0])]
                        mean_output = torch.stack(outputs).mean(dim=0)
                    top_vals, top_idxs = torch.topk(mean_output, 5)
                    fired = [f"\"{vocab.id_to_word(i)}\"({v:.2f})"
                             for v, i in zip(top_vals.tolist(), top_idxs.tolist()) if v > 0.005]
                    print(f"│ Zihin   : {', '.join(fired) if fired else '(henüz zayıf bağlantı)'}")
                except Exception as ex:
                    print(f"│ Zihin   : [hata: {ex}]")

            print(f"│ Kümül.  : Ort.RPE={avg_rpe_so_far:.4f}  Ort.dW={avg_dw_so_far:.6f}  İşlenen:{idx+1}/{len(sentences)}")
            print(f"└{'─'*57}")
            sys.stdout.flush()
            # ──────────────────────────────────────────────────────────────────

            # Senkronize Uyku Döngüsü (Auto-Sleep)
            if sentence_counter % args.sleep_interval == 0:
                save_weights(args.mx_path, engine)
                run_sync_dream(args.mx_path, args.sleep_cycles, args.device)
                load_weights(args.mx_path, engine, args.device)
                print(f"\n[Sleep-Sync] Uyku tamamlandı. Eğitime devam... (Toplam cümle: {sentence_counter})")

        # Epoch nihai özeti
        avg_rpe = total_rpe / max(1, processed_in_epoch)
        avg_dw = total_dw / max(1, processed_in_epoch)
        print(f"\n{'='*60}")
        print(f"  [Epoch {epoch+1} Bitti] {idx+1}/{len(sentences)} cümle | Ort.RPE:{avg_rpe:.4f} | Ort.dW:{avg_dw:.6f}")
        print(f"{'='*60}\n")

    return sentence_counter


class DualLogger(object):
    """Hem terminale hem de bir dosyaya eşzamanlı olarak çıktı yazar."""
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()

def main():
    parser = argparse.ArgumentParser(description="Mergen Google Colab Büyük Eğitim Scripti")
    parser.add_argument("--corpus_dir", type=str, default=None, help="Wikipedia .txt chunk dizini (Drive)")
    parser.add_argument("--local_dir", type=str, default="./tmp_local/", help="Yerel hızlı I/O dizini")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/", help="Checkpoint Drive dizini")
    parser.add_argument("--core_path", type=str, default="data/training/core_curriculum_v1.txt", help="Çekirdek müfredat dosya yolu")
    parser.add_argument("--mx_path", type=str, default="./mergen_weights.mx", help="Weights dosya yolu")
    parser.add_argument("--vocab_path", type=str, default="./mergen_vocab.json", help="Vocab dosya yolu")
    parser.add_argument("--sleep_interval", type=int, default=500, help="Kaç cümlede bir uyku konsolidasyonu çalışsın")
    parser.add_argument("--sleep_cycles", type=int, default=1000, help="Uyku döngü sayısı")
    parser.add_argument("--som_lr", type=float, default=0.05, help="L23 SOM öğrenme hızı")
    parser.add_argument("--stdp_reward", type=float, default=1.0, help="STDP ödül çarpanı")
    parser.add_argument("--epochs_per_chunk", type=int, default=1, help="Her parça kaç kez dönülsün")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Cihaz (cuda/cpu)")
    parser.add_argument("--embed_cache_size", type=int, default=8192, help="Wernicke gömme önbellek boyutu")
    parser.add_argument("--compile", action="store_true", help="PyTorch 2.0 compile özelliğini aktif et")
    args = parser.parse_args()

    # Klasörleri doğrula
    os.makedirs(args.local_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Çift günlük kaydediciyi etkinleştir (logs/ klasörü altında)
    os.makedirs("logs", exist_ok=True)
    log_file_path = os.path.join("logs", "training.log")
    sys.stdout = DualLogger(log_file_path)
    sys.stderr = DualLogger(log_file_path)

    print(f"\n[Logging] Eğitim logları hem konsola hem de dosyaya yazılıyor: {log_file_path}")

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
        embed_cache_size=args.embed_cache_size,
    )

    # Cortical Column (SNN Beyin)
    engine = CorticalColumn(n_pre=768, n_post=vocab_size, n_hidden=1024, device=args.device)
    if args.compile and hasattr(torch, 'compile'):
        print("[Colab-Train] PyTorch 2.0 derleme (torch.compile) etkinleştiriliyor...")
        engine = torch.compile(engine)

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
    core_completed = False
    total_sentences_trained = 0

    if os.path.exists(state_file_path):
        try:
            with open(state_file_path, 'r') as sf:
                state_data = json.load(sf)
                processed_chunks = state_data.get("processed_chunks", [])
                core_completed = state_data.get("core_completed", False)
                total_sentences_trained = state_data.get("total_sentences_trained", 0)
                print(f"[Colab-Train] Kaldığı yerden devam ediliyor.")
                print(f"  - Çekirdek Müfredat Tamamlandı: {core_completed}")
                print(f"  - İşlenen Wiki parça sayısı: {len(processed_chunks)}")
                print(f"  - Toplam eğitilen cümle sayısı: {total_sentences_trained}")
        except Exception as se:
            print(f"[Colab-Train] Durum dosyası okuma hatası: {se}")

    # 3. Çekirdek Müfredat Eğitimi (İlk Öncelik)
    if not core_completed:
        print(f"\n" + "="*70)
        print(f"[Curriculum] ÇEKİRDEK MÜFREDAT EĞİTİMİ BAŞLIYOR: {args.core_path}")
        print("="*70)
        
        if not os.path.exists(args.core_path):
            print(f"Hata: Çekirdek müfredat dosyası {args.core_path} bulunamadı.")
            sys.exit(1)

        # Yerel diske kopyala
        core_local_path = os.path.join(args.local_dir, "core_curriculum_v1.txt")
        shutil.copy2(args.core_path, core_local_path)
        
        # Eğit
        sentences_trained = train_file(core_local_path, engine, wernicke, vocab, args, vocab_size, args.device)
        total_sentences_trained += sentences_trained
        
        # Yerel dosyayı temizle
        if os.path.exists(core_local_path):
            os.remove(core_local_path)
            
        # İlk part/core bittiğinde rüya döngüsü ve kaydetme
        save_weights(args.mx_path, engine)
        run_sync_dream(args.mx_path, args.sleep_cycles, args.device)
        load_weights(args.mx_path, engine, args.device)
        
        # Drive'a yedekle
        drive_mx_core = os.path.join(args.checkpoint_dir, "mergen_weights_core_completed.mx")
        shutil.copy2(args.mx_path, drive_mx_core)
        shutil.copy2(args.mx_path, os.path.join(args.checkpoint_dir, "mergen_weights.mx"))
        print(f"[Backup] Çekirdek müfredat ağırlıkları Drive'a yedeklendi: {drive_mx_core}")
        
        # Durumu güncelle ve kaydet
        core_completed = True
        with open(state_file_path, 'w') as sf:
            json.dump({
                "core_completed": core_completed,
                "processed_chunks": processed_chunks,
                "last_update": datetime.now().isoformat(),
                "total_sentences_trained": total_sentences_trained,
                "last_processed_file": "core_curriculum_v1.txt"
            }, sf, indent=4)
        print(f"[Backup] Çekirdek müfredat tamamlandı durum kaydı yapıldı: {state_file_path}")

    # 4. Wikipedia Partları Eğitim Döngüsü
    if args.corpus_dir is None or not os.path.exists(args.corpus_dir):
        print("\n[Colab-Train] Wikipedia corpus dizini sağlanmadı veya bulunamadı.")
        print("[Colab-Train] Çekirdek müfredat eğitimi tamamlandı. Yerel duman testi BAŞARILI! ☀️")
        return

    all_chunks = sorted([f for f in os.listdir(args.corpus_dir) if f.endswith(".txt")])
    unprocessed_chunks = [c for c in all_chunks if c not in processed_chunks]

    if not unprocessed_chunks:
        print("[Colab-Train] İşlenecek yeni Wikipedia parçası kalmadı! Eğitim tamamlandı.")
        return

    print(f"[Colab-Train] Toplam Wiki parçası: {len(all_chunks)}, Kalan parça: {len(unprocessed_chunks)}")

    for chunk_name in unprocessed_chunks:
        chunk_drive_path = os.path.join(args.corpus_dir, chunk_name)
        chunk_local_path = os.path.join(args.local_dir, chunk_name)

        print(f"\n" + "="*70)
        print(f"[Wiki-Chunk] Sıradaki parça kopyalanıyor: {chunk_name}")
        print("="*70)

        # Drive -> Yerel Disk I/O Transferi
        shutil.copy2(chunk_drive_path, chunk_local_path)

        # Eğit
        sentences_trained = train_file(chunk_local_path, engine, wernicke, vocab, args, vocab_size, args.device)
        total_sentences_trained += sentences_trained

        # Yerel dosyayı temizle
        if os.path.exists(chunk_local_path):
            os.remove(chunk_local_path)
            print(f"[Chunk] Yerel dosya temizlendi: {chunk_local_path}")

        # Her part bittiğinde rüya (dream) döngüsü ve kaydetme
        save_weights(args.mx_path, engine)
        run_sync_dream(args.mx_path, args.sleep_cycles, args.device)
        load_weights(args.mx_path, engine, args.device)

        # Checkpoint Yedekleme (Drive'a yedekle)
        drive_mx_backup = os.path.join(args.checkpoint_dir, f"mergen_weights_chunk_{chunk_name.replace('.txt', '')}.mx")
        shutil.copy2(args.mx_path, drive_mx_backup)
        shutil.copy2(args.mx_path, os.path.join(args.checkpoint_dir, "mergen_weights.mx"))
        print(f"[Backup] Ağırlıklar Drive'a yedeklendi: {drive_mx_backup}")

        # Durumu güncelle ve kaydet
        processed_chunks.append(chunk_name)
        with open(state_file_path, 'w') as sf:
            json.dump({
                "core_completed": core_completed,
                "processed_chunks": processed_chunks,
                "last_update": datetime.now().isoformat(),
                "total_sentences_trained": total_sentences_trained,
                "last_processed_file": chunk_name
            }, sf, indent=4)
        print(f"[Backup] Eğitim durumu güncellendi: {state_file_path}")

    print("\n[Colab-Train] Tüm müfredat ve Wikipedia parçaları başarıyla tamamlandı! ☀️")

if __name__ == '__main__':
    main()
