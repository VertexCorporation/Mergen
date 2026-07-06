# MERGEN

> *"Zeka statik bir eşleme fonksiyonu değil, yaşayan ve ritmik bir süreçtir."*

**Mergen**, günümüzün derin öğrenme (Deep Learning) paradigmalarının sınırlarını aşmak üzere tasarlanmış, biyolojik esintili deneysel bir bilişsel mimaridir. İsmini Türk mitolojisinde bilgelik, her şeyi bilme ve akıl tanrısı olan **Mergen**'den alır; derin bilgi ile mükemmel eylemin arasındaki köprüyü simgeler.

---

## Problem: Neden Transformer Değil?

Günümüzün baskın yapay zeka modelleri (Büyük Dil Modelleri, Transformer mimarileri) mühendislik harikalarıdır, fakat temelde **statiktirler**.
*   **Zamanı algılamazlar;** sadece dizileri işlerler.
*   **İçsel durumları (internal states) yoktur;** her token üretiminden sonra sistem sıfırlanır.
*   Devasa küresel matris çarpımları nedeniyle yüksek enerji tüketirler.
*   **Düşünmezler;** sadece istatistiksel olarak bir sonraki en olası kelimeyi tahmin ederler.

**Yapay Genel Zekanın (AGI)** sadece matris çarpımlarını ölçeklendirerek (scaling up) elde edilemeyeceğine inanıyoruz. Bu hedefe ulaşmak, beynin fiziksel yapısını ve sinaptik plastisitesini taklit eden sistemlere geçişi gerektirir.

---

## Çözüm: Mergen Mimarisi (V8.0)

Mergen sadece bir chatbot değildir. Modern donanımlar (GPU hızlandırmalı PyTorch) üzerinde çalışan bir **Dijital Beyin** simülasyonudur. Zihnin sadece çıktılarını değil, işlevsel dinamiklerini ve plastisitesini (öğrenme yeteneğini) kopyalamayı hedefler.

### Temel Felsefe
1.  **Sürekli Zaman Dinamikleri:** Mergen, zaman gecikmeleri, ritimler ve senkronizasyon (spike trenleri) ile sürekli zaman akışında çalışır.
2.  **Hebbian Plastisite & STDP:** Bilgi, Spike-Timing-Dependent Plasticity (STDP) ve Hebbian öğrenme kuralları ile lokal olarak sinapslarda güncellenir ve saklanır.
3.  **Default Mode Network (DMN):** Beyin boşta (idle) kaldığında, uyku döngüleri (rüya aşamaları) aracılığıyla kendi kendini düzenler ve bellek konsolidasyonu yapar.
4.  **Katı İzolasyon:** Sentetik çıkarımlar, halüsinasyon kirliliğini önlemek için gerçek dünyaya dayalı verilerden izole edilir.

---

## Bilişsel Mimari Genel Yapısı

Mergen V8, **4 Katmanlı bir Spiking Cortical Kolon** yapısını uygular:

### 1. Wernicke Alanı (Duyusal Algı)
Gelen metin dizilerini anlamsal gömme (embedding) temsiline ve zamansal rate spike trenlerine (768 boyutlu) çevirir.

### 2. Spiking Cortical Kolon (Sinaptik Bellek Çekirdeği)
PyTorch ile hızlandırılmış, ~4.05 milyon sinaps içeren 4 katmanlı bir neokorteks kolonu:
- **Katman 4 (L4 - Granüler Giriş):** Wernicke/Talamus'tan gelen duyusal girdileri alır ve L23'e yansıtır.
- **Katman 23 (L23 - Supragranüler İlişkisel):** Mexican Hat yanal bağlantıları kullanarak Kohonen SOM (Self-Organizing Map) tabanlı topolojik temsil haritalaması yapar (32x32 mekansal grid).
- **Katman 5 (L5 - İnfragranüler Çıkış):** İlişkisel katmandan gelen girdileri motor spike intent (niyet) vektörlerine (kelime dağarcığı eşleşmelerine) dönüştürür.
- **Katman 6 (L6 - Multiformis Geri Besleme):** L23'ten L4'e geri yansıtma yaparak **Öngörücü Kodlama (Predictive Coding)** gerçekleştirir; gelen duyusal girdiyi öngörü hatasına (residual/surprise) indirger.

### 3. Broca Alanı (Dil İfade Katmanı)
Dil üretiminden sorumludur. Ham motor niyet spike vektörlerinden anlamlı cümleler oluşturmak için kural tabanlı bir Türkçe Özne-Nesne-Yüklem (SOV) cümle oluşturucusu ve şablonlar kullanır.

### 4. Limbic & Yürütücü Kontrol Katmanı
Otonominin orkestratörüdür. Şu işlevleri yönetir:
- **Default Mode Network (DMN):** Sistem boşta kaldığında otomatik rüya görme döngülerini tetikler. Canlı sohbet esnasında VRAM çakışmalarını önleyen spin-wait senkronizasyon mekanizmasına sahiptir.
- **Uyku Borcu Takibi:** Uyku döngüleri, etkileşim yüküne ve veri sindirme oranına göre dinamik olarak ayarlanır.
- **Durum Kalıcılığı (.mx Protokolü):** Beyin durumunun tamamını (ağırlıklar, izler, düşünceler) XOR+Base64 ile şifreleyerek kaydeder.

### 5. Çift Koleksiyonlu RAG Motoru
Hızlı ve Transformer gerektirmeyen karakter n-gram tabanlı bir `BioVectorizer` ve HTM (Hierarchical Temporal Memory) yeniden sıralayıcı içeren yerel vektör veri tabanıdır. Bilgileri kesin olarak ayırır:
- `mergen_bilgi_bio`: Gerçek dünya girdileri ve doğrulanmış bilgi olguları.
- `mergen_ruya_bio`: REM uykusu sırasında sentezlenen spekülatif/sentetik ilişkiler (`reliability: synthetic` metaverisi ile işaretlenir).

---

## ⚡ Fark Yaratan Özellikler

| Özellik | Standart Transformer | **MERGEN Motoru** |
| :--- | :--- | :--- |
| **İletişim** | Yoğun Matris Çarpımı | **Seyrek Spike'lar ve Lokal Alanlar** |
| **Zaman** | Ayrık Adımlar (Tokenlar) | **Sürekli Akış (dt)** |
| **Bellek** | Sınırlı Bağlam Penceresi | **Hebbian Sinaptik Bellek + RAG** |
| **Öğrenme** | Yalnızca Backpropagation | **Lokal Plastisite + STDP + Dopamin** |
| **Durum** | Durumsuz (Stateless) | **Kalıcı Dinamik Durum (.mx)** |
| **Dinlenme** | Boşta (Sıfır Aktivite) | **Aktif Bellek Konsolidasyonu (DMN)** |

---

## 🚀 Başlangıç Kılavuzu

### Kurulum

```bash
git clone https://github.com/VertexCorporation/Mergen.git
cd Mergen
pip install -r requirements.txt
```

### Beyni Çalıştırma ve Test Etme

#### 1. İnteraktif CLI Sohbeti (`Mergen.py`)
Mergen ile sohbet etmek ve etkileşim döngüsünü başlatmak için:
```bash
python Mergen.py
```
*   **CLI Komutları:**
    *   `/stats` - Kelime haznesi boyutu, yanıt sayısı ve dopamin verimlilik metriklerini gösterir.
    *   `/introspect` - Mevcut iç ses kavramlarını ve dinlenme durumu parametrelerini listeler.
    *   `/clear` - Sohbet geçmişini temizler.
    *   `/exit` - Beyni kapatır, bellek konsolidasyonunu tetikler ve ağırlıkları `mergen_weights.mx` dosyasına kaydeder.
*   **Veri Okutma Sözdizimi:**
    *   `oku:dosya.txt` - Belirtilen metin dosyasını RAG veri tabanına ve bilgi havuzuna sindirir.

#### 2. Bulut Eğitim Döngüsü (`scripts/train_colab.py`)
Büyük ölçekli veri kümeleriyle bulut ortamlarında otonom eğitim yürütmek, yerel veri transferi adımlarıyla disk akışını optimize etmek ve kaynak kullanımını dengelemek amacıyla senkronize uyku konsolidasyonu döngülerini çalıştırmak için kullanılır:
```bash
python scripts/train_colab.py \
    --corpus_dir "./data/chunks/" \
    --checkpoint_dir "./checkpoints/" \
    --sleep_interval 500 \
    --sleep_cycles 1000
```

#### 3. Deneyim Sindirme ve Rüya Görme (`scripts/simulation_playground.py`)
Ham deneyim paragraflarını beynin anlamsal belleğine besler ve otomatik rüya konsolidasyonunu başlatır:
```bash
python scripts/simulation_playground.py --data ./data/simulation_texts.txt --dream-cycles 20
```

#### 4. Matematik ve Müfredat Eğitimi (`scripts/math_training.py`)
Mergen'in matematiksel kavramları aşamalı olarak (Tier 0 -> Tier 3) öğrenmesini sağlayan müfredat eğitimidir.
```bash
# Tek bir zorluk seviyesinde (Tier 0) eğitim:
python scripts/math_training.py --tier 0 --epochs 5 --split 0.80 --dream --dream-cycles 10

# Aşamalı Müfredat Eğitimi (Hata durumunda eğitimi durduran Hard Stop korumalı):
python scripts/math_training.py --curriculum --epochs 5 --dream --dream-cycles 5
```

#### 5. Kelime Dağarcığı Eğitimi (`scripts/train_vocabulary.py`)
1416 Türkçe temel kavramı neokorteks katmanlarına STDP ve dopamin ödül mekanizmalarıyla sıfırdan eğitir:
```bash
python scripts/train_vocabulary.py
```
- CUDA üzerinde eğitimi ~2 dakikada tamamlar.
- Eğitim başarısını (HIT oranı, hedef > %99) test eder ve ağırlıkları `mergen_weights.mx` olarak kaydeder.

#### 6. Innate Priors Üretimi (`scripts/generate_innate_priors.py`)
Katman boyutlarına (L4, L23, L5, L6) uygun varsayılan priors ağırlıklarını sıfırdan oluşturur:
```bash
python scripts/generate_innate_priors.py
```
- `mergen_cortical_priors.pt` ve legacy `mergen_innate_priors.pt` dosyalarını yazar.

#### 7. Topolojik Mexican Hat Testi (`scripts/test_topology.py`)
Katman 23'teki Mexican Hat yanal bağlantılarının SOM haritalama başarısını test eder:
```bash
python scripts/test_topology.py
```
- Küme-içi ve küme-dışı aktivasyon oranlarını (hedef < 0.6) raporlar.

#### 8. Bilişsel Sağlık Kontrolü (`scripts/verify_all_layers.py`)
Tüm biyolojik katmanların (Kelime Haznesi, Wernicke, Hebbian, Broca, Limbik ve DMN) kararlı çalışıp çalışmadığını uçtan uca doğrular:
```bash
python scripts/verify_all_layers.py
```

---

## 🛡️ Lisans

Bu proje **Apache License 2.0** ile lisanslanmıştır. 
Mergen, AGI araştırma topluluğuna sunulmuş açık kavramsal bir katkıdır.

---

<p align="center">
  <i>"Ben Mergen'im."</i>
</p>
