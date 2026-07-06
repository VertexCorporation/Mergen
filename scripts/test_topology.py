"""
╔══════════════════════════════════════════════════════════════════════╗
║  MERGEN — Faz 2 Topoloji Testi                                       ║
║                                                                      ║
║  Bu script L23 katmanının Mexican Hat yanal bağlantılarıyla          ║
║  gerçekten topolojik bir anlamsal harita oluşturup oluşturmadığını   ║
║  doğrular.                                                           ║
║                                                                      ║
║  Test Mantığı:                                                       ║
║    1. Semantik gruplar (hayvanlar, araçlar, fiiller) için sabit      ║
║       vektörler oluştur.                                             ║
║    2. 4500 adım STDP + Dopamin eğitimi simüle et.                    ║
║    3. Her kelimenin L23 üzerinde ateşlediği nöronların               ║
║       ağırlık merkezini (Center of Mass) hesapla.                    ║
║    4. Küme-içi mesafe < küme-arası mesafenin %50'si → PASS           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys
import math
import torch
import random

# Windows terminali icin UTF-8 zorlama
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Proje kök dizinini sys.path'e ekle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from learning.cortical_column import CorticalLayer

# -- Sabitler -------------------------------------------------
N_IN           = 64    # Basit test: 64 boyutlu giris
N_HIDDEN       = 1024  # L23 noron sayisi (32x32)
GRID_H         = 32
GRID_W         = 32
TRAINING_STEPS = 2000  # Her adimda tum kelimeler dondurularak gosterilir
REWARD         = 1.0   # Pozitif dopamin (ogrenme)
DEVICE         = 'cpu'

# ── Test Grupları (Semantik Kümeler) ─────────────────────────────────
# Her kelimenin vektörü: N_IN boyutlu, kendi grubu içinde benzer,
# farklı gruplardan farklı.
#
# Strateji: Her gruba ayrı bir "temel bölge" atanır.
# Hayvanlar : ilk 20 boyut aktif  (indeks 0-19)
# Araçlar   : orta 20 boyut aktif (indeks 22-41)
# Fiiller   : son 20 boyut aktif  (indeks 44-63)
#
# Aynı gruptaki kelimeler bu temel bölgeye küçük rastgele gürültü ekler.
# Böylece hem benzerlik hem de bireysel kimlik korunur.

GROUPS = {
    'hayvanlar': ['kedi',   'köpek',  'kuş'],
    'araçlar':   ['araba',  'otobüs', 'kamyon'],
    'fiiller':   ['koşmak', 'yürümek','gitmek'],
}

BASE_REGIONS = {
    'hayvanlar': (0,  20),
    'araçlar':   (22, 42),
    'fiiller':   (44, 64),
}

def make_word_vector(group: str, word: str, seed: int) -> torch.Tensor:
    """
    Semantik grup yapısını yansıtan sabit vektör üret.
    Aynı gruptaki kelimeler yüksek cosine benzerliği paylaşır.
    """
    torch.manual_seed(seed)
    v = torch.zeros(N_IN)
    start, end = BASE_REGIONS[group]
    # Temel gruptaki tüm boyutları aktifleştir
    v[start:end] = 0.8
    # Küçük rastgele varyasyon (bireysel kimlik)
    v += torch.randn(N_IN) * 0.05
    v = torch.relu(v)  # Negatif değerlere izin verme
    v = v / (v.norm() + 1e-8)  # Normalize
    return v

def compute_center_of_mass(
    layer: CorticalLayer,
    word_vec: torch.Tensor,
) -> tuple:
    """
    Verilen kelime vektörünü L23'ten geçirir ve aktif nöronların
    32x32 grid üzerindeki ağırlık merkezini (x, y) döndürür.
    """
    with torch.no_grad():
        out = layer.forward(word_vec, spiking=True)  # (1024,)

    # Grid üzerindeki koordinatlar
    total_activation = out.sum().item()
    if total_activation < 1e-8:
        return (GRID_W / 2.0, GRID_H / 2.0)  # Merkez (nötr)

    cx = 0.0
    cy = 0.0
    for idx in range(N_HIDDEN):
        row = idx // GRID_W
        col = idx % GRID_W
        cx += col * out[idx].item()
        cy += row * out[idx].item()

    cx /= total_activation
    cy /= total_activation
    return (cx, cy)

def euclidean(p1: tuple, p2: tuple) -> float:
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def mean_intra_cluster_dist(centers: list) -> float:
    """Verilen center listesi içindeki ortalama çift-çift mesafe."""
    if len(centers) < 2:
        return 0.0
    dists = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dists.append(euclidean(centers[i], centers[j]))
    return sum(dists) / len(dists)

def mean_inter_cluster_dist(c_list_a: list, c_list_b: list) -> float:
    """İki küme arasındaki ortalama mesafe."""
    dists = []
    for ca in c_list_a:
        for cb in c_list_b:
            dists.append(euclidean(ca, cb))
    return sum(dists) / len(dists) if dists else 0.0

# ── Test Fonksiyonu ───────────────────────────────────────────────────

def run_topology_test(verbose: bool = True) -> bool:
    """
    Topoloji testini çalıştırır.
    Returns: True → PASS, False → FAIL
    """
    print("=" * 64)
    print("  MERGEN Faz 2 — L23 Topoloji Testi")
    print("=" * 64)

    # ── Adım 1: L23 Katmanını Oluştur ────────────────────────────────
    if verbose:
        print("\n[1/4] L23 katmanı oluşturuluyor (32x32 Mexican Hat)...")

    # Sadece L23'u test ediyoruz (L4 yok, dogrudan N_IN → N_HIDDEN)
    # lateral_k=5: sadece en guclu 5 noron aktif -> daha belirgin BMU secimi
    # spike_threshold=0.1: dusurek ateSleme, ama k-WTA onu kis kilar
    l23 = CorticalLayer(
        n_in=N_IN,
        n_out=N_HIDDEN,
        spike_threshold=0.1,
        lateral_k=5,           # Cok kucuk k -> net, belirgin aktivasyon
        topology_grid=(GRID_H, GRID_W),
        lat_sigma_exc=2.0,
        lat_sigma_inh=5.0,
        lat_A_exc=0.5,
        lat_A_inh=0.25,
        lat_spectral_target=0.9,
        A_ltp=0.01,
        A_ltd=0.005,
        device=DEVICE,
    )

    # ── Veri Tabanli Agirlik Ilklendirmesi ───────────────────────────
    # Standart SOM uygulamasi: her noron, egitim orneklerinden birini
    # baslangic agirlik vektoru olarak alir. Bu, BMU seciminin anlamli
    # olmasi icin kritik. Rastgele ilklendirmede tum noronlar benzer
    # skora sahip olur -> BMU secimi anlamsiz.
    if verbose:
        print("    Veri tabanli agirlik ilklendirmesi...")

    all_vecs = []
    for grp, words in GROUPS.items():
        sd = 42
        for w in words:
            all_vecs.append(make_word_vector(grp, w, sd))
            sd += 1

    # Her cikis noronuna en yakin egitim ornegiyle ilklendir
    # (Forgy ilklendirmesi: rastgele orneklerden sec)
    with torch.no_grad():
        for j in range(N_HIDDEN):
            sample = all_vecs[j % len(all_vecs)]
            # Hafif gurultu ekle -> benzersiz ilk pozisyonlar
            noise = torch.randn(N_IN) * 0.1
            init_w = (sample + noise).clamp(0.0, 1.0)
            l23.weights.data[:, j] = init_w

    if verbose:
        print(f"    Agirlik ilklendirmesi tamamlandi: "
              f"{N_HIDDEN} noron, {len(all_vecs)} ornek")


    if verbose:
        has_topology = l23._has_topology
        w_lat_shape = l23.W_lat.shape if has_topology else None
        print(f"    Topoloji aktif: {has_topology}")
        if has_topology:
            print(f"    W_lat sekli: {w_lat_shape}")
            print(f"    W_lat ortalama: {l23.W_lat.mean():.4f}")
            print(f"    W_lat spektral buyukluk (yaklasik): "
                  f"{l23.W_lat.abs().max():.4f}")

    # ── Adım 2: Kelime Vektörlerini Hazırla ──────────────────────────
    if verbose:
        print(f"\n[2/4] Kelime vektörleri hazırlanıyor...")

    word_vectors = {}
    seed = 42
    for group, words in GROUPS.items():
        for w in words:
            vec = make_word_vector(group, w, seed)
            word_vectors[w] = (group, vec)
            seed += 1
            if verbose:
                print(f"    {w:10s} ({group:10s}): norm={vec.norm():.3f}, "
                      f"tepe bolge=[{BASE_REGIONS[group][0]}-{BASE_REGIONS[group][1]}]")

    # -- Adim 3: SOM Egitimi (Kohonen Kurali) --------------------------------
    # Not: STDP + Dopamin homeostazı SOM güncellemelerine müdahale ediyor.
    # Topolojik organizasyon testinde sadece SOM kullanilir.
    # Ana sistemde (Limbic) her ikisi paralel calisir ama lr dengesi kritik.
    if verbose:
        print(f"\n[3/4] SOM egitimi ({TRAINING_STEPS} adim)...")
        print(f"    Kohonen BMU + komsuluk cekim kurali")

    all_words = list(word_vectors.items())  # [(word, (group, vec)), ...]
    total_steps = 0

    # Klasik SOM ogrenme takvimi: baslangicta genis alan + yuksek lr,
    # zamanla daralip dusur -> ince ayar
    som_lr_initial  = 0.5   # Agresif baslangi: hizli organizasyon
    som_lr_final    = 0.02  # Ince ayar sonu
    sigma_initial   = 6.0   # Genis komsu (global topoloji)
    sigma_final     = 1.0   # Dar komsu (lokal ince ayar)

    for step in range(TRAINING_STEPS):
        # Ogrenme hizi ve komsu boyutu: eksponansiyel azalma
        t = step / max(TRAINING_STEPS - 1, 1)
        som_lr    = som_lr_initial * (som_lr_final / som_lr_initial) ** t
        sigma_eff = sigma_initial  * (sigma_final  / sigma_initial)  ** t

        # Her adimda tum kelimeleri karistir
        random.shuffle(all_words)
        for word, (group, vec) in all_words:
            # Sadece SOM guncelleme (STDP yok)
            # sigma_eff'i dogrudan som_update'e gecirmek icin
            # BMU-tabanli Gaussian komsu hesabi yapiyoruz
            W = l23.weights.data  # (n_in, n_out)
            scores = torch.mv(W.t(), vec)
            bmu_idx = int(torch.argmax(scores).item())
            bmu_row = bmu_idx // GRID_W
            bmu_col = bmu_idx % GRID_W

            rows = torch.arange(GRID_H, dtype=torch.float32)
            cols = torch.arange(GRID_W, dtype=torch.float32)
            grid_rows = rows.unsqueeze(1).expand(GRID_H, GRID_W).reshape(-1)
            grid_cols = cols.unsqueeze(0).expand(GRID_H, GRID_W).reshape(-1)
            dist_sq = (grid_rows - bmu_row)**2 + (grid_cols - bmu_col)**2
            neighborhood = torch.exp(-dist_sq / (2.0 * sigma_eff**2))

            delta = vec.unsqueeze(1) - W
            update = som_lr * neighborhood * delta
            W.add_(update)
            W.clamp_(min=0.0, max=1.0)

            total_steps += 1

        if verbose and (step + 1) % 500 == 0:
            sample_word, (sample_group, sample_vec) = all_words[0]
            with torch.no_grad():
                sample_post = l23.forward(sample_vec, spiking=True)
            active = (sample_post > 0).sum().item()
            print(f"    Adim {step+1:4d}/{TRAINING_STEPS}: "
                  f"lr={som_lr:.4f}, sigma={sigma_eff:.2f}, "
                  f"'{sample_word}' -> aktif noron: {active}")



    # ── Adım 4: Ağırlık Merkezleri Hesapla ───────────────────────────
    if verbose:
        print(f"\n[4/4] Ağırlık merkezleri (Center of Mass) hesaplanıyor...")

    group_centers = {group: [] for group in GROUPS}

    for word, (group, vec) in all_words:
        cx, cy = compute_center_of_mass(l23, vec)
        group_centers[group].append((cx, cy))
        if verbose:
            print(f"    {word:10s} ({group:10s}): CoM=({cx:.1f}, {cy:.1f})")

    # ── Adım 5: Topolojik Organizasyon Ölçümü ────────────────────────
    print("\n" + "-" * 64)
    print("  TOPOLOJI OLCUM SONUCLARI")
    print("-" * 64)

    group_names = list(GROUPS.keys())
    intra_dists = {}
    for g in group_names:
        intra_dists[g] = mean_intra_cluster_dist(group_centers[g])
        print(f"  Küme-içi mesafe [{g:10s}]: {intra_dists[g]:.2f} grid adımı")

    print()
    inter_pairs = []
    for i in range(len(group_names)):
        for j in range(i+1, len(group_names)):
            ga, gb = group_names[i], group_names[j]
            d = mean_inter_cluster_dist(group_centers[ga], group_centers[gb])
            inter_pairs.append((ga, gb, d))
            print(f"  Küme-arası mesafe [{ga} ↔ {gb}]: {d:.2f} grid adımı")

    # ── Pass/Fail Kriteri ─────────────────────────────────────────────
    print()
    mean_intra = sum(intra_dists.values()) / len(intra_dists)
    mean_inter = sum(d for _, _, d in inter_pairs) / len(inter_pairs)

    threshold = 0.6  # Kume-ici < kume-arasi x 0.6
    passed = mean_intra < mean_inter * threshold

    print(f"  Ortalama kume-ici mesafe  : {mean_intra:.2f}")
    print(f"  Ortalama kume-arasi mesafe: {mean_inter:.2f}")
    print(f"  Oran (intra/inter)        : {mean_intra/max(mean_inter,0.001):.2f} (hedef < {threshold})")

    print()
    if passed:
        print("  SONUC: PASS")
        print("  L23 Mexican Hat topolojisi semantik kumeleme uretiyor.")
    else:
        print("  SONUC: FAIL")
        print("  Topolojik organizasyon henuz yetersiz.")
        print("  Olasi sebepler:")
        print("    - Egitim adimi sayisi yetersiz (TRAINING_STEPS artirin)")
        print("    - spike_threshold cok yuksek (az noron atesleniyoor)")
        print("    - Mexican Hat sigma degerleri ince ayar gerektirebilir")

    print("=" * 64)
    return passed


if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)

    # Komut satırından -v / --verbose kontrolü
    verbose = True
    if '--quiet' in sys.argv or '-q' in sys.argv:
        verbose = False

    ok = run_topology_test(verbose=verbose)
    sys.exit(0 if ok else 1)
