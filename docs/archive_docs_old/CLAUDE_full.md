# CLAUDE.md - Mergen Project Handoff

Bu belge, Mergen projesini yeni bir AI coding tool'unun sifirdan ve eksiksiz anlamasi icin hazirlanmis teknik devir dokumanidir. Bu projede temel oncelik "daha iyi chatbot" yapmak degildir; asil hedef, backpropagation ve harici LLM kullanmadan, biyolojiden ilham alan katmanli bir bilissel mimariyi olculebilir sekilde calistirmaktir.

Claude Code bu projede calismaya baslamadan once mutlaka bu dosyayi ve proje kokundeki `PROJECT_MEMORY_BANK.md` dosyasini okumalidir.

---

## 1. Projenin Amaci ve Mimarisi

### Temel Amac

Mergen, STDP ve Hebbian prensipleriyle calisan, biyolojiden ilham almis deneysel bir dijital beyin prototipidir. Projenin tasarim ilkeleri sunlardir:

- Cekirdek bilissel isleyiste harici LLM kullanilmaz.
- Cekirdek ogrenme hatti backpropagation kullanmaz.
- Ogrenme, yerel aktivite, kavram eslesmesi, Hebbian izler, STDP benzeri agirlik guncellemeleri, RAG tabanli hatirlama ve Dream konsolidasyonu uzerinden yurutulur.
- Amac sadece dogru cevap vermek degil; hangi katmanin davranisa nasil katkida bulundugunu olcebilir hale getirmektir.

### Aktif Mimari: Conversational Brain v7.0

Aktif sohbet/runtime mimarisi su giris noktasindan calisir:

- `Mergen.py`
- `brain.py` icindeki `MergenBrain_v7`

Canli sohbet akisinin yuksek seviye yolu:

1. Kullanici girdisi `Mergen.py` uzerinden `MergenBrain_v7.handle_command()` veya `respond()` hattina gelir.
2. `conversation_memory.py`, onceki baglami ve zamir/co-reference benzeri referanslari cozmeye calisir.
3. `intent_analyzer.py`, girdinin niyetini ve konu/subject bilgisini cikarir.
4. Limbic katman aktifse ic aktivasyon uretir; QA modunda bu katmanin serbest Broca cevabi kullanilmaz, sadece oncelik/valuation sinyali kullanilir.
5. `_recall_knowledge()` bilgi adaylarini toplar:
   - KB/fact recall
   - subject tabanli recall
   - RAG recall
   - Hebbian trace skoru
   - Limbic kavram ortusmesi skoru
   - Semantic fallback yalnizca ozel/debug veya son care modunda
6. Adaylar yapilandirilmis skorlarla siralanir:
   - `text`
   - `source`
   - `base_score`
   - `rag_score`
   - `hebbian_score`
   - `limbic_score`
   - `final_score`
   - `matched_concepts`
7. `response_generator.py`, en iyi adaylardan kisa, deterministik, konu merkezli cevap sentezler.
8. Aktif ogrenme guvenlik kapisi, soru cumlelerini bilgi olarak ezberlemeyi engellemeye calisir.
9. Etkilesim hafizasi ve ilgili state dosyalari guncellenir.

### Ana Bilissel Katmanlar

#### Wernicke / Algisal Katman

Ilgili dosyalar:

- `wernicke_area.py`
- `mergen_brain_wrapper.py`

Wernicke katmani metinleri embedding/spike temsillerine donusturmek ve algisal semantik sinyal uretmek icin kullanilir. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` modeline baglidir. Model bulunamazsa veya HF Hub erisimi sorunluysa runtime degrade edebilir; bu durumda sistemin tamamen cokmemesi beklenir.

Onemli not: Daha once semantic recall her soruda pahali embedding taramasi yapiyordu ve cevap hattini domine ediyordu. Bu nedenle varsayilan QA hattindan cikarildi. Semantic recall yeniden ana yola alinmamalidir.

#### Neural Core / Hebbian Katman

Ilgili dosyalar:

- `mergen_brain.py`
- `learning/hebbian_engine.py`
- `brain.py`

Metinlerden kelime/kavram iliskileri, fact kayitlari ve Hebbian izler olusturur. Son mimari degisikliklerden sonra Hebbian trace artik sadece sayaç degil, QA aday siralamasinda skor bileseni olarak kullanilir.

#### RAG / Hatirlama Katmani

Ilgili dosyalar:

- `rag_engine.py`
- `bio_vectorizer.py`
- `htm_retriever.py`
- `hebbian_rag_bridge.py`

RAG sistemi ChromaDB, biyolojik vectorizer ve HTM retriever bilesenleriyle calisir. 10k deneyimlik egitimde RAG kayitlari oldukca guclu calismistir. `hebbian_rag_bridge.py`, RAG kayitlariyla Hebbian izler arasinda kopru kurar.

Beklenen davranis: RAG kapatildiginda audit raporunda cevap siralamasi veya kaynak dagilimi degismelidir. Degismiyorsa RAG runtime davranisa baglanmiyor demektir.

#### Limbic Executive Layer

Ilgili dosyalar:

- `limbic_executive_layer.py`
- `emotional_reward_system.py`
- `homeostatic_controller.py`
- `drive_system.py`

Limbic katman, serbest metin uretici olarak guvenilir degildir. QA modunda rastgele Broca cevabi yerine valuation/gating sinyali olarak kullanilmalidir. `last_thought` veya ateslenen kavramlar, fact adaylariyla ortusurse `limbic_score` verir.

Limbic state dosyasi:

- `mergen_weights.mx`

Bu dosya `.mx` protokoluyle kaydedilen biyolojik state'i tutar. Silmeden once mutlaka arsivlenmelidir.

#### Broca / Dil Uretim Katmani

Ilgili dosyalar:

- `broca_area.py`
- `language_engine.py`
- `response_generator.py`

`broca_area.py` ve `language_engine.py` deneysel serbest uretim motorlari olarak durur. Aktif QA kalitesi bu serbest motorlara dayandirilmamalidir. Canli cevaplarda asil kontrol `response_generator.py` tarafindadir.

Son durum: `response_generator.py`, rastgele/dağinik ciktilar yerine kisa, deterministik, konu odakli sentez uretmek icin iyilestirildi. Yine de bu bir LLM degildir; ciktisi retrieval ve heuristik senteze dayanir.

#### Dream Konsolidasyonu

Ilgili dosya:

- `dream.py`

Dream, `.mx` protokoluyle uyumlu konsolidasyon katmanidir. Manuel komutlarla calisir:

- `dream:run`
- `dream:uyku`
- `dream:run <cycles>`
- `dream:uyku <cycles>`

Sinirlar:

- Varsayilan cycle: `100`
- Gecerli aralik: `1..1000`
- Otomatik DMN/idle Dream tetiklemesi su anda aktif mimariye baglanmamalidir.

Dream katkisi sadece "calisti" diye kabul edilmez; audit ile Dream oncesi/sonrasi cevap siralamasi, latency, copy ratio ve skor bilesenleri karsilastirilmalidir.

---

## 2. Klasor Yapisi ve Onemli Dosyalar

### Kok Dizin

- `Mergen.py`  
  Aktif sohbet uygulamasinin ana giris noktasi. Terminalden Mergen'i baslatir.

- `brain.py`  
  Aktif Conversational Brain v7.0 orkestratoru. Intent, recall, scoring, Limbic, Dream komutlari ve response hattini birlestirir.

- `response_generator.py`  
  Retrieval adaylarindan kontrollu cevap sentezler. Cevap kalitesi ve copy ratio acisindan kritik dosyadir.

- `mergen_brain.py`  
  Neural core, concept/fact ogrenme, Hebbian trace ve KB state yonetimi.

- `mergen_brain_wrapper.py`  
  Brain ile Wernicke/semantic katman arasinda ara katman.

- `wernicke_area.py`  
  Sentence-transformers tabanli algisal/semantik temsil katmani.

- `limbic_executive_layer.py`  
  Limbic state, biyolojik deneyim akisi, ic aktivasyon ve `.mx` state okuma/yazma.

- `dream.py`  
  Dream/NREM/REM konsolidasyon motoru.

- `rag_engine.py`  
  RAG motoru ve ChromaDB entegrasyonu.

- `bio_vectorizer.py`  
  Transformer kullanmadan biyolojik esinli local vectorizer.

- `htm_retriever.py`  
  HTM benzeri retrieval mantigi.

- `hebbian_rag_bridge.py`  
  RAG kayitlari ve Hebbian izler arasindaki kopru.

- `intent_analyzer.py`  
  Intent, subject ve soru turu analizi.

- `conversation_memory.py`  
  Konusma baglami, referans cozme ve kisa donem etkilesim hafizasi.

- `PROJECT_MEMORY_BANK.md`  
  Projenin kritik tarihcesi, onceki buglar, kararlar, yol haritasi ve mevcut durum. Her ciddi degisiklikten sonra guncellenmelidir.

- `requirements.txt`  
  Python bagimliliklari.

### `scripts/`

- `scripts/verify_all_layers.py`  
  Ana saglik kontrolu. Beklenen sonuc: `6/6 TESTS PASSED`.

- `scripts/simulation_playground.py`  
  Experience training harness. Veri dosyasindaki her bos olmayan satiri deneyim olarak isler; KB/RAG/Hebbian/Limbic/Dream hattini calistirir.

- `scripts/cognitive_audit.py`  
  Bilissel mimari gerceklik denetimi. Baseline ve ablation kosullarinda probe sorularini calistirir; latency, kaynak, skor bilesenleri, copy ratio ve katman etkilerini raporlar.

- `scripts/archive_playground_metrics.py`  
  Training loglarini olculebilir metrik arsivine donusturur.

### `data/`

- `data/simulation_texts.txt`  
  Playground icin varsayilan kucuk deneyim seti.

- `data/training/core_curriculum_v1.txt`  
  Birlesik buyuk egitim dataseti. Kullanici bunu 10k+ deneyime kadar buyutmustur.

- `data/training/core_physics_v1.txt`
- `data/training/core_biology_v1.txt`
- `data/training/core_ai_v1.txt`
- `data/training/core_reasoning_v1.txt`
- `data/training/mergen_identity_v1.txt`  
  Daha kucuk konu bazli egitim dosyalari. Son pratik karar, tek buyuk curriculum dosyasi kullanmaktir.

### `datasets/`

- `datasets/generators/math_teacher.py`  
  Aritmetik ogretimi icin deneysel veri/teacher modulu. Su anda aktif sohbet hattina tam entegre degildir. Geliştirme hedeflerinden biridir.

### `anatomy/`, `connectivity/`, `learning/`

Biyolojik esinli dusuk seviye moduller:

- neuron modelleri
- sinaps
- topology/connectome
- Hebbian engine
- plasticity kurallari
- homeostasis

Bu moduller aktif runtime'in alt katmanlarini besler. Bu alanlarda degisiklik yapmadan once `verify_all_layers.py` mutlaka calistirilmalidir.

### `docs/`

Proje dokumantasyonu. Bu dosya da burada yer alir:

- `docs/CLAUDE.md`

### State ve Artifact Dosyalari

- `mergen_knowledge.mx`  
  Brain/KB state.

- `mergen_weights.mx`  
  Limbic `.mx` state.

- `mergen_rag_db/`  
  ChromaDB persistent RAG veritabani.

- `mergen_vocab.json`  
  Kavram/vocabulary state.

- `mergen_innate_priors.pt`  
  Innate prior agirliklari.

- `mergen_conversation_memory.json`  
  Konusma hafizasi.

- `mergen_matrix_memory.json`  
  Ek matrix memory artifact'i.

- `dream_log.npz`  
  Dream konsolidasyon logu.

---

## 3. Teknoloji Yigini

### Dil ve Runtime

- Python 3.10+ onerilir.
- Proje Windows uzerinde PowerShell ile aktif gelistirilmistir.
- GPU kullanimi PyTorch tarafindan desteklenir; CPU fallback de beklenir.

### `requirements.txt`

Mevcut bagimliliklar:

```txt
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
requests>=2.28.0
chromadb>=0.4.0
zeyrek>=0.1.0
sentence-transformers
```

### Ana Kutuphaneler

- PyTorch  
  Tensor hesaplama, agirliklar, neural state ve device yonetimi.

- NumPy  
  Numeric state, Dream loglari, vector islemleri.

- ChromaDB  
  Persistent RAG store.

- sentence-transformers  
  Wernicke semantic encoder. Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

- zeyrek  
  Turkce morfolojik isleme icin kullanilir.

- matplotlib  
  Dream ve deneysel gorsellestirme icin kullanilabilir; komut entegrasyonunda `visualize=False` tercih edilir.

### Harici Servisler

Aktif core runtime icin zorunlu harici servis yoktur. Hugging Face model indirme gerekiyorsa internet/HF cache gerekebilir. HF uyari mesajlari normaldir:

- `HF_TOKEN` opsiyonel olarak rate limit ve download stabilitesi icin kullanilabilir.

Projede `openrouter_client.py` gibi deneysel dosyalar bulunabilir; bunlar core bilissel mimarinin parcasi olarak kabul edilmemelidir.

---

## 4. Kodlama Standartlari ve Kurallar

### Genel Kurallar

- Sessiz hata yutma yasaktir. `except: pass` eklenmemelidir.
- Hatali durumlar ya loglanmali ya da kullaniciya acik mesajla donmelidir.
- Degisiklikler kucuk, odakli ve olculebilir olmali.
- Mimari butunluk, performanstan once gelir.
- Dirty worktree varsayilmalidir; kullanici degisiklikleri geri alinmamalidir.
- State dosyalari silinmeden once arsivlenmelidir.

### Karar Hiyerarsisi

1. Mimari Butunluk
2. Guvenilirlik
3. Dogruluk
4. Surdurulebilirlik
5. Performans

### Isimlendirme

- Python fonksiyonlari ve degiskenleri: `snake_case`
- Siniflar: `PascalCase`
- Sabitler: `UPPER_CASE`
- CLI argumanlari: `--kebab-case`
- Dosya adlari: mevcut repo stili korunur; genelde `snake_case.py`

### Mimari Kurallar

- Aktif sohbet hatti `Mergen.py` + `brain.py` + `response_generator.py` merkezlidir.
- `main.py` ayri bir V3/headless aritmetik deney hattidir; aktif sohbet mimarisine dogrudan karistirilmamalidir.
- Semantic recall varsayilan QA yoluna geri alinmamalidir.
- Limbic QA modunda serbest cevap uretmemelidir; scoring/valuation sinyali vermelidir.
- RAG, Hebbian trace ve Limbic katkisi audit raporlarinda gorulebilir olmali.
- Dream sadece manuel veya explicit test komutlariyla calistirilmalidir; otomatik DMN/idle Dream sonraki faz konusudur.

### Test Standartlari

Kod degisikliklerinden sonra en azindan:

```powershell
python -m py_compile brain.py response_generator.py scripts/cognitive_audit.py limbic_executive_layer.py dream.py
```

Mimari degisikliklerden sonra:

```powershell
python scripts/verify_all_layers.py
```

QA/recall/response degisikliklerinden sonra:

```powershell
python scripts/cognitive_audit.py --max-probes 3 --conditions baseline,no_rag,no_limbic,no_semantic,no_hebbian_trace
```

Training harness degisikliklerinden sonra:

```powershell
python scripts/simulation_playground.py --dry-run --limit 2
```

---

## 5. Kurulum ve Cevre Degiskenleri

### Lokal Kurulum

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Opsiyonel Cevre Degiskenleri

Aktif runtime icin zorunlu `.env` anahtari yoktur. Ancak asagidakiler faydali olabilir:

```env
HF_TOKEN=...
HF_HOME=...
TRANSFORMERS_CACHE=...
```

Opsiyonel/deneysel dosyalar icin:

```env
OPENROUTER_API_KEY=...
```

`OPENROUTER_API_KEY`, core Mergen mimarisi icin zorunlu degildir ve harici LLM kullanan deneysel yollarla karistirilmamalidir.

### Ilk Calistirma Notlari

- Wernicke ilk calismada Hugging Face model indirebilir.
- HF token yoksa unauthenticated request uyarisi alinabilir.
- ChromaDB versiyon/path farklari nedeniyle bazi ortamlarda RAG kayit sayisi farkli gorunebilir.
- Kullanici ortaminda daha once 10k training sonrasinda RAG kayitlari 10k+ seviyesindeydi.

---

## 6. Veri ve State Yonetimi

### Kalici State Dosyalari

#### `mergen_knowledge.mx`

Brain/KB tarafindaki bilgi state'ini tutar. Fact kayitlari, learned facts ve neural core state bu dosyayla iliskilidir.

#### `mergen_weights.mx`

Limbic state dosyasidir. `.mx` protokolu uzerinden kaydedilir. Iceriginde agirliklar, epizodlar, trace'ler, odul bilgileri ve vocabulary baglamlari bulunabilir.

#### `mergen_rag_db/`

ChromaDB persistent RAG veritabani. Training harness RAG aktifse buraya kayit ekler.

#### `mergen_vocab.json`

Kavram sozlugu. Limbic ve language katmanlari icin onemlidir.

#### `mergen_conversation_memory.json`

Kisa/orta vadeli konusma hafizasi.

#### `dream_log.npz`

Dream konsolidasyon rapor/log artifact'i.

### Egitim Veri Akisi

Ana egitim hatti:

1. `scripts/simulation_playground.py` veri dosyasini okur.
2. Bos olmayan her satir bir `experience` olarak kabul edilir.
3. `mergen.brain.learn_from_text(...)` ile KB/Hebbian ogrenme calisir.
4. `mergen.limbic.respond(text, max_attempts=1)` ile biyolojik deneyim akisi tetiklenir.
5. RAG aktifse ayni satir RAG store'a eklenir.
6. Hebbian-RAG bridge aktifse batch update yapilir.
7. Sonunda brain state ve Limbic `.mx` state kaydedilir.
8. Istenirse manuel Dream konsolidasyonu calisir.

### Temiz Yeniden Egitim

Kullanici ileride `mergen_weights.mx` dahil state dosyalarini silip temiz, kaliteli dataset ile bastan egitmeyi planliyor. Dogru yaklasim:

1. Mevcut state dosyalarini silmeden once `archive/` veya timestamp'li bir klasore tasimak.
2. `mergen_weights.mx`, `mergen_knowledge.mx`, `mergen_rag_db/`, gerekirse `mergen_conversation_memory.json` ve ilgili memory artifactlerini birlikte ele almak.
3. Yeni dataset icin once `--dry-run` calistirmak.
4. Kucuk limit smoke testi yapmak.
5. Sonra tam training.
6. Training sonrasi cognitive audit calistirmak.

---

## 7. Sik Kullanilan Komutlar

### Sohbet Runtime

```powershell
python Mergen.py
```

Canli komutlar:

```text
/stats
/introspect
/clear
/exit
dream:run
dream:run 10
dream:uyku
dream:uyku 10
```

### Syntax Kontrolu

```powershell
python -m py_compile brain.py response_generator.py scripts/cognitive_audit.py limbic_executive_layer.py dream.py
```

### Tam Katman Saglik Kontrolu

```powershell
python scripts/verify_all_layers.py
```

Beklenen sonuc:

```text
6/6 TESTS PASSED
```

### Hızlı Cognitive Audit

```powershell
python scripts/cognitive_audit.py --max-probes 3 --conditions baseline,no_rag,no_limbic,no_semantic,no_hebbian_trace
```

### Tam Cognitive Audit

```powershell
python scripts/cognitive_audit.py --conditions baseline,no_rag,no_limbic,no_semantic,no_hebbian_trace
```

### Dream Katkisi Audit

```powershell
python scripts/cognitive_audit.py --dream-cycles 1 --conditions baseline,no_rag,no_limbic,no_semantic,no_hebbian_trace
```

### Training Harness Dry Run

```powershell
python scripts/simulation_playground.py --dry-run --limit 2
```

### Kisa Training Smoke Test

```powershell
python scripts/simulation_playground.py --limit 2 --dream-cycles 1
```

### Buyuk Dataset Training

```powershell
python scripts/simulation_playground.py --data data/training/core_curriculum_v1.txt --dream-cycles 25 --strict-eval --bridge-timeout 120
```

### Dream Kapali Training

```powershell
python scripts/simulation_playground.py --data data/training/core_curriculum_v1.txt --no-dream
```

### Metrics Arsivleme

```powershell
python scripts/archive_playground_metrics.py --input logs\run_001.txt --name run_001
```

Bu komut, training console logundan metrikleri ayiklayip `logs/metrics/` altinda JSON/MD olarak arsivlemek icin kullanilir.

### Ayrı Aritmetik Deney Hatti

```powershell
python main.py
```

Not: `main.py`, aktif sohbet mimarisinden ayri bir V3/headless aritmetik/SNN deney hattidir.

---

## 8. Bilinen Kisitlamalar ve Teknik Borclar

### Sistem Hala Deneysel

Mergen bir LLM degildir. Cevaplar retrieval, scoring, Hebbian trace, Limbic valuation ve heuristik response synthesis uzerinden uretilir. Dogal dil akiciligi sinirlidir.

### Limbic Katki Hala Zayif

Audit sonucunda Limbic katmanin cevap kalitesine etkisi sinirli kalmistir. Son mimaride Limbic metin uretmez, skor sinyali verir. Gelecekte Limbic'in daha guclu ve olculebilir gating/priority rolu kazanmasi gerekir.

### RAG ve KB Kalitesi Datasete Bagimli

Kotu veya parcali dataset, Mergen'in cevabini dogrudan bozar. Ornegin named entity sorularinda guvenilir fact yoksa sistem kontrollu "guvenilir bilgi bulamadim" cevabi vermelidir.

### Semantic Recall Varsayilan Disinda Kalmalidir

Onceki auditlerde semantic recall pahaliydi ve diger katmanlari maskeleyebiliyordu. Varsayilan cevap hattina tekrar alinmasi mimari gerceklik denetimini bozar.

### Wernicke/HF Bagimliligi

`sentence-transformers` modeli ilk calismada indirilebilir. Offline veya token'siz ortamda uyari verebilir. `brain.py`, Wernicke init hatalarinda degrade edebilmelidir.

### RAG DB Ortam Farki

Kullanici ortaminda 10k+ RAG kaydi gorulmustur. Farkli Python/ChromaDB ortamlarinda `mergen_rag_db/` okunamayabilir veya bos gorunebilir. Audit sonucunu yorumlarken ortam farki kontrol edilmelidir.

### `main.py` ile `Mergen.py` Karistirilmamalidir

Repo icinde iki farkli deneysel hat vardir:

- `Mergen.py`: aktif sohbet/bilissel mimari
- `main.py`: ayri arithmetic/headless V3 deney hatti

Bunlar ayni sey degildir.

### Aritmetik Ogretimi Tam Entegre Degil

`datasets/generators/math_teacher.py` vardir, fakat aktif sohbet hattina tam entegre degildir. Mergen'e aritmetik ogretmek icin yeni bir curriculum, arithmetic probe set, cevap dogrulama ve scoring baglantisi gerekir.

### Dokumanlarin Bir Kismi Eski Olabilir

`README.md` ve bazi eski dokumanlar, onceki mimari durumlari anlatabilir. En guncel teknik kaynaklar:

1. `PROJECT_MEMORY_BANK.md`
2. `docs/CLAUDE.md`
3. `brain.py`
4. `scripts/cognitive_audit.py`
5. `scripts/simulation_playground.py`

### Encoding/Mojibake

Bazi eski terminal ciktilarinda Turkce karakterler bozuk gorunebilir. Yeni dokuman ve kodlarda UTF-8 korunmalidir.

---

## 9. Mevcut Durum ve Yol Haritasi

### En Son Kalinan Nokta

Proje test edilebilir durumdadir. Son operasyonlarda:

- Semantic recall varsayilan QA hattindan cikarildi.
- RAG, Hebbian trace ve Limbic signal aday scoring'e baglandi.
- ResponseGenerator daha deterministik ve konu merkezli hale getirildi.
- Question residue filtreleri eklendi.
- Multi-word subject parsing iyilestirildi.
- `yerçekimi/kütleçekim/gravitasyonel` ve `rüya/dream/konsolidasyon` alias'lari eklendi.
- Named entity sorularinda guvenilir bilgi yoksa uydurma cevap yerine kontrollu bilinmezlik donulmesi saglandi.
- Dream manuel komutlari eklendi.
- Simulation Playground, training harness haline getirildi.
- Cognitive audit araci eklendi.
- Memory bank son kararlarla guncellendi.

### Ornek Son Davranis

Beklenen tarz:

```text
Sen > Kütleçekim nedir?
Mergen > Kütleçekim hakkinda: cisimler arasında çekim etkisi oluşturur. Ayrica kütleçekim hakkinda: uzay ve zamanı büker.
```

Guvenilir veri yoksa:

```text
Sen > Cengiz Han kimdir?
Mergen > Bu konuda guvenilir bir bilgi bulamadim.
```

Bu tercih bilerek yapilmistir. Uydurma cevap, arastirma prototipi icin basarisizlik kabul edilir.

### Kisa Vadeli Yol Haritasi

1. Tam test gecisi
   - `py_compile`
   - `verify_all_layers.py`
   - hizli cognitive audit
   - tam cognitive audit

2. Dataset kalite denetimi
   - konu basina temiz tanimlar
   - soru-cevap degil, tutarli deneyim cumleleri
   - meta-training cumlelerinin domain cevaplarini domine etmemesi

3. Temiz yeniden egitim
   - eski state dosyalarini arsivle
   - kaliteli curriculum ile bastan train et
   - Dream konsolidasyonunu kontrollu calistir
   - audit ile once/sonra karsilastir

4. Aritmetik entegrasyonu
   - `math_teacher.py` incelenecek
   - toplama/cikarma/carpma/bolme icin curriculum uretilecek
   - arithmetic probe set yazilacak
   - runtime cevap hattina guvenilir arithmetic path eklenecek
   - bu path LLM/backprop kullanmadan calismali

5. Dream katkisini gercekten olcmek
   - Dream oncesi/sonrasi ayni probe set
   - trace dagilimi
   - cevap siralamasi
   - latency
   - copy ratio

6. Limbic katkisini guclendirmek
   - `limbic_score` daha anlamli hale getirilecek
   - random kavram ateslemelerinin QA cevabini bozmasi engellenecek
   - Limbic kapali/acik audit farki olculecek

7. Dokuman temizligi
   - README guncellenecek
   - eski Architecture B/V3 ayrimi netlestirilecek
   - komutlar ve state dosyalari tek dokumanda toparlanacak

### Basari Kriterleri

Mergen'in basarili sayilmasi icin:

- Sadece guzel konusmasi yetmez.
- Katmanlarin davranisa nedensel katkisi audit ile gorunmelidir.
- RAG kapatilinca fark olusmalidir.
- Hebbian trace sifirlaninca bazi cevap siralamalari degismelidir.
- Limbic kapatilinca en azindan skor bilesenlerinde kontrollu fark gorulmelidir.
- Dream sonrasi farklar sadece logda degil, davranis metriklerinde de izlenmelidir.
- Verify saglik testi bozulmamalidir.

---

## 10. Claude Code Icin Calisma Protokolu

### Ilk Yapilacaklar

1. `docs/CLAUDE.md` dosyasini oku.
2. `PROJECT_MEMORY_BANK.md` dosyasini bastan sona oku.
3. `brain.py`, `response_generator.py`, `scripts/cognitive_audit.py`, `scripts/simulation_playground.py`, `limbic_executive_layer.py`, `rag_engine.py`, `dream.py` dosyalarini incele.
4. Herhangi bir mimari iddiada bulunmadan once audit veya test ciktilarina bak.

### Degisiklik Yaparken

- Once ilgili kodu oku.
- Kucuk ve gerekceli patch yap.
- Sessiz hata yutma ekleme.
- State dosyalarini silme veya yeniden olusturma isini kullanici onayi olmadan yapma.
- Semantic recall'u ana yola geri alma.
- Core mimariye harici LLM veya backprop ekleme.
- Eger bir ozellik sadece chatbot kalitesini artiriyor ama mimari olculebilirligi bozuyorsa, uygulamadan once tartis.

### Her Ciddi Degisiklikten Sonra

- `PROJECT_MEMORY_BANK.md` guncellenmeli.
- En azindan ilgili syntax/test komutlari calistirilmali.
- Cognitive davranisa etki eden degisiklikler audit ile dogrulanmali.

---

## 11. Claude Code Icin Ilk Hedef Onerisi

Projeyi devralan Claude Code'un ilk hedefi yeni ozellik eklemek olmamalidir. Ilk hedef:

1. Mevcut test durumunu dogrulamak.
2. Son mimari degisikliklerin calisma durumunu audit ile gormek.
3. State dosyalari ve RAG DB'nin mevcut ortamda okunup okunmadigini kontrol etmek.
4. Sonra sadece bir odak secmek:
   - temiz yeniden egitim plani
   - arithmetic/math teacher entegrasyonu
   - Limbic scoring guclendirme
   - dataset kalite denetimi

Bu proje icin dogru gelistirme tarzi: once olc, sonra bagla, sonra egit, sonra tekrar olc.

