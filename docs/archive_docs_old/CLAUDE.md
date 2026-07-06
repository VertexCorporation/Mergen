# CLAUDE.md - Mergen Quick Handoff

Bu dosya Claude Code'un her oturumda hizlica okuyacagi kisa proje rehberidir.

Detayli mimari icin: `docs/CLAUDE_full.md` dosyasini oku (sadece gerektiginde).

Proje hafizasi icin: `PROJECT_MEMORY_BANK.md` dosyasini oku.

---

## Proje Amaci

Mergen, backpropagation ve harici LLM kullanmadan, STDP ve Hebbian prensipleriyle calisan biyolojiden ilham almis deneysel bir dijital beyin prototipidir.

Hedef sadece daha iyi chatbot yapmak degildir. Asil hedef, Wernicke, RAG, Hebbian trace, Limbic, Broca/Response ve Dream katmanlarinin canli davranisa olculebilir katkisini gosteren bir bilissel mimari kurmaktir.

---

## Aktif Mimari

Aktif sohbet/runtime hatti:

- `Mergen.py`
- `brain.py` icindeki `MergenBrain_v7`
- `response_generator.py`

Yuksek seviye akis:

1. Kullanici girdisi `Mergen.py` ile gelir.
2. `conversation_memory.py` baglami ve referanslari cozmeye calisir.
3. `intent_analyzer.py` intent ve subject cikarir.
4. `brain.py` KB/RAG/Hebbian/Limbic adaylarini toplar ve skorlar.
5. `response_generator.py` kisa, deterministik cevap sentezler.
6. Gerekirse aktif ogrenme ve state kaydi calisir.

Onemli ayrim: `Mergen.py` aktif sohbet mimarisidir; `main.py` ayri V3/headless arithmetic/SNN deney hattidir.

---

## Kritik Dosyalar

- `brain.py`: ana orkestrator; intent, recall, scoring, Limbic, Dream komutlari.
- `response_generator.py`: retrieval adaylarindan kontrollu cevap uretir.
- `mergen_brain.py`: KB, concept/fact ogrenme, Hebbian trace.
- `wernicke_area.py`: semantic/spike temsil; HF sentence-transformers kullanabilir.
- `limbic_executive_layer.py`: Limbic state, `.mx`, ic aktivasyon.
- `rag_engine.py`, `bio_vectorizer.py`, `htm_retriever.py`: RAG ve retrieval altyapisi.
- `hebbian_rag_bridge.py`: RAG ile Hebbian trace koprusu.
- `dream.py`: manuel Dream konsolidasyonu.
- `scripts/simulation_playground.py`: experience training harness.
- `scripts/cognitive_audit.py`: baseline/ablation bilissel mimari denetimi.
- `scripts/verify_all_layers.py`: 6/6 katman saglik kontrolu.
- `PROJECT_MEMORY_BANK.md`: en guncel tarihce, kararlar, buglar ve yol haritasi.

---

## State ve Veri

Onemli state dosyalari:

- `mergen_knowledge.mx`: Brain/KB state.
- `mergen_weights.mx`: Limbic `.mx` state.
- `mergen_rag_db/`: ChromaDB RAG veritabani.
- `mergen_vocab.json`: vocabulary.
- `mergen_conversation_memory.json`: konusma hafizasi.
- `dream_log.npz`: Dream log artifact'i.

Egitim dataseti:

- `data/training/core_curriculum_v1.txt`
- `data/simulation_texts.txt`

State dosyalarini silme veya yeniden olusturma isini kullanici onayi olmadan yapma. Temiz yeniden egitim gerekiyorsa once arsivle.

---

## Cekirdek Kurallar

- Core runtime'a harici LLM veya backpropagation ekleme.
- Sessiz hata yutma ekleme; `except: pass` yasak.
- Semantic recall'u varsayilan QA yoluna geri alma.
- Limbic'i QA'da serbest cevap uretici yapma; scoring/valuation sinyali olarak kullan.
- RAG, Hebbian trace ve Limbic katkisi audit raporlarinda gorulebilir olmali.
- Dream otomatik DMN/idle tetikleme ile baglanmadi; manuel ve olculerek calistir.
- Dirty worktree varsay; kullanici degisikliklerini geri alma.
- Mimari butunluk, guvenilirlikten; guvenilirlik, performanstan once gelir.

---

## Sik Komutlar

```powershell
python Mergen.py
python -m py_compile brain.py response_generator.py scripts/cognitive_audit.py limbic_executive_layer.py dream.py
python scripts/verify_all_layers.py
python scripts/cognitive_audit.py --max-probes 3 --conditions baseline,no_rag,no_limbic,no_semantic,no_hebbian_trace
python scripts/simulation_playground.py --dry-run --limit 2
python scripts/simulation_playground.py --data data/training/core_curriculum_v1.txt --dream-cycles 25 --strict-eval --bridge-timeout 120
python scripts/math_training.py --dry-run --tier 1
python scripts/math_training.py --tier 0 --no-save
python scripts/math_training.py --tier 1 --dream --dream-cycles 10
```

Canli Dream komutlari:

```text
dream:run
dream:run 10
dream:uyku
dream:uyku 10
```

---

## Mevcut Durum

Proje test edilebilir durumdadir. Son mimari operasyonlarda semantic recall varsayilan QA hattindan cikarildi; RAG, Hebbian trace ve Limbic signal candidate scoring'e baglandi; `response_generator.py` daha deterministik ve konu merkezli hale getirildi.

Ek olarak multi-word subject ve alias esleme iyilestirildi, Dream manuel komutlari eklendi, Simulation Playground training harness haline getirildi ve Cognitive Audit araci eklendi.

2026-06-24 itibariyle tamamlanan mimariye oturtma isleri:
- response_generator.py sessiz except bloklari temizlendi (kural: except:pass yasak).
- Limbic scoring guclendirildi: _limbic_signal_concepts sorgu kavramlarini da okuyor; _limbic_candidate_score arousal_gate kaldirildi, soft match agirligina kaydi (0.55/0.35/0.10); final formul limbic 0.12->0.15, rag 0.18->0.15.
- Hebbian trace rank-based normallestirildi: max normalizasyonu yerine torch.searchsorted percentile rank.

Aritmetik egitim entegrasyonu tamamlandi (2026-06-24):
- math_teacher.py genisletildi: 4 islem (toplama/cikarma/carpma/bolme), difficulty tiers, enumerate_all(), format_fact().
- scripts/math_training.py eklendi: train/holdout split, per-operasyon recall degerlendirme, generalization gap raporu.
- Tier 0 (toplama): train=0.88, holdout=0.70, PASS.
- Tier 1 (toplama+cikarma): train=0.90, holdout=0.77, PASS.
- brain.py, response_generator.py, mergen_brain.py degistirilmedi.

Bir sonraki en mantikli is: temiz yeniden egitim (simulation_playground.py) veya tam curriculum egitim (tier 2-3 ile carpma/bolme testi).

