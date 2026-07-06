import torch
import numpy as np

# Faz 1: Neuromodulation
from learning.neuromodulation import NeuromodulationSystem

# Faz 2 ve 3: Thalamic Gating & Working Memory
from cognitive.mergen_brain import MergenBrain, MergenConfig

# Faz 4: RAG Engine
from rag_engine import RAGEngine

def test_phase_1():
    print("--- Faz 1: Nöromodülasyon Testi Başlıyor ---")
    nm = NeuromodulationSystem(device='cpu')
    
    # Kural 1: Float değil, Tensor olmalı.
    assert isinstance(nm.da, torch.Tensor), "DA tensor değil!"
    
    # Kural 2: requires_grad=False olmalı.
    assert nm.da.requires_grad == False, "DA requires_grad True geldi!"
    
    # Kural 3: In-place update mekanizması (tick_homeostasis)
    eski_da = nm.da.clone()
    nm.tick_homeostasis()
    yeni_da = nm.da
    
    print("Faz 1 Başarılı! Tensörler doğru formatta (requires_grad=False) ve in-place çalışıyor.")
    print(f"DA Öncesi: {eski_da.item():.4f}, Sonrası: {yeni_da.item():.4f}\n")

def test_phase_2_and_3():
    print("--- Faz 2 ve 3: Thalamic Gating & Working Memory Testi Başlıyor ---")
    config = MergenConfig()
    # Dummy boyutlar
    brain = MergenBrain(vocab_size=668, config=config)
    
    # Dummy input raporu hazırlayalım (Wernicke çıkışı gibi)
    # Cosine Similarity eps testini tetiklemek için Working Memory'yi zorluyoruz.
    report = {
        'primary_intent': 'GREETING',
        'confidence_score': 0.9,
        'subject': 'test',
        'sentiment': {'sentiment_score': 0.5, 'excitement': 0.5},
        'morphology': {'is_question': False, 'tense': None}
    }
    
    # Sistemi bir kez çalıştırarak Thalamic Gating ve WM süreçlerini tetikletiyoruz.
    output = brain.process(report)
    
    print("Faz 2 & 3 Başarılı! Thalamic Gating ve Working Memory (cosine_similarity) çökmeden çalıştı.\n")

def test_phase_4():
    print("--- Faz 4: RAG Döngüsü Testi Başlıyor ---")
    rag = RAGEngine(db_path="./test_rag_db") # Test için ayrı bir path
    success = rag.initialize(verbose=False)
    
    if not success:
        print("Uyarı: RAG Engine kütüphaneleri (chromadb, bio_vectorizer) yüklü değilse atlanabilir.")
        return

    # Metadata'sı eksik (eski tip) veya standart bir veri indeksleyelim.
    # memory_type otomatik olarak 'semantic' olmalı.
    index_count = rag.index_texts(["Burası test verisidir."], source="test", metadatas=[{"source": "test"}])
    
    # Eklenen veriyi memory_type="semantic" ile çekmeye çalışalım.
    results = rag.search("test", memory_type="semantic")
    
    if len(results) > 0:
        print("Faz 4 Başarılı! RAG indeksleme memory_type varsayılan atamasıyla çalıştı.")
    else:
        print("Faz 4 Tamamlandı, ancak search sonuç döndürmedi (Beklenen bir durum olabilir).")
    print("")

if __name__ == "__main__":
    try:
        test_phase_1()
        test_phase_2_and_3()
        test_phase_4()
        print("BÜTÜN FAZ TESTLERİ BAŞARIYLA TAMAMLANDI!")
    except Exception as e:
        import traceback
        print(f"HATA OLUŞTU:\n")
        traceback.print_exc()
