# -*- coding: utf-8 -*-
"""
Tests for Gelişmiş Rüyalar & Bellek Konsolidasyonu (Sleep/DMN)
"""
import sys
import os
import shutil
import json
import torch

sys.path.append('.')

from rag_engine import RAGEngine
from cognitive.wernicke_area import WernickeArea
from learning.hebbian_engine import HybridHebbianLearner, NUM_MASKED_TOKENS
from cognitive.broca_area import BrocaArea
from cognitive.limbic_executive_layer import LimbicExecutiveLayer
from cognitive.response_generator import ResponseGenerator
from cognitive.dream import MergenDream

# Setup temporary test directories
DB_PATH = "./test_mergen_rag_db"
WEIGHTS_PATH = "./test_mergen_weights.mx"
DIARY_PATH = "./dream_diary.json"

if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
if os.path.exists(WEIGHTS_PATH):
    os.remove(WEIGHTS_PATH)
if os.path.exists(DIARY_PATH):
    os.remove(DIARY_PATH)

def test_dream_system():
    print("=" * 60)
    print(" STARTING DREAM CONSOLIDATION TEST SUITE (TDD)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Initialize core components
    print("\n[Test 1] Initializing components...")
    vocab_size = 1416
    vocab_words = ["kelime"] * vocab_size # dummy vocab
    
    # Initialize RAG Engine
    rag = RAGEngine(db_path=DB_PATH)
    success = rag.initialize(verbose=True)
    assert success, "RAGEngine initialization failed"
    
    # Initialize Wernicke
    wernicke = WernickeArea(
        n_neurons=768,
        embedding_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        encoding='rate',
        time_window=10,
        device=device
    )
    
    # Initialize Hebbian Engine
    hebbian = HybridHebbianLearner(
        n_pre=768,
        n_post=vocab_size,
        device=device
    )
    
    # Initialize Broca
    broca = BrocaArea(
        n_neurons=vocab_size,
        concept_vocabulary=vocab_words,
        device=device
    )
    
    # Initialize Limbic
    limbic = LimbicExecutiveLayer(
        mergen_engine=hebbian,
        broca=broca,
        wernicke=wernicke,
        mx_path=WEIGHTS_PATH,
        user_id="test_user"
    )
    
    # Save a baseline weights file
    limbic.save_state()
    assert os.path.exists(WEIGHTS_PATH), "Weights file not created"

    # 2. Test RAG Dream Collection & Isolation
    print("\n[Test 2] Testing RAG separate dream collection & isolation...")
    # Add a normal fact
    rag.index_texts(["Güneş sıcak bir yıldızdır. Güneş Sistemi'nin merkezinde yer alır."], source="input")
    # Add a dream fact (should be implemented in step 2)
    assert hasattr(rag, "add_dream_fact"), "RAGEngine is missing add_dream_fact method"
    rag.add_dream_fact("Elma kuantumu etkiler. Kuantum fiziğinde elmanın rolü büyüktür.", confidence=0.2)
    
    # Verify isolation: normal query should not return the dream fact
    normal_results = rag.search("kuantum", top_k=5)
    # Check if "Elma kuantumu etkiler" is in normal results
    for r in normal_results:
        assert "Elma kuantumu etkiler" not in r.get('text', ''), "Dream fact leaked into normal query!"
    print("  [OK] Dream fact successfully isolated from normal query.")
    
    # Query dream collection specifically
    assert hasattr(rag, "query_dream"), "RAGEngine is missing query_dream method"
    dream_results = rag.query_dream("kuantum", top_k=5)
    found_dream = False
    for r in dream_results:
        if "Elma kuantumu etkiler" in r.get('text', ''):
            found_dream = True
            assert r.get('source') == 'DREAM', "Incorrect source metadata"
            assert r.get('reliability') == 'synthetic', "Incorrect reliability metadata"
            assert r.get('confidence') == 0.2, "Incorrect confidence metadata"
    assert found_dream, "Dream fact not found in dream collection query!"
    print("  [OK] Dream fact successfully retrieved from dream collection with proper metadata.")

    # 3. Test NREM text memory consolidation
    print("\n[Test 3] Testing targeted NREM text memory consolidation...")
    dream = MergenDream(
        config_path="config.py",
        verbose=True
    )
    # Inject config override for tests
    dream.config['MX_WEIGHTS_PATH'] = WEIGHTS_PATH
    dream.load_memory()
    
    assert hasattr(dream, "consolidate_text_memory"), "MergenDream is missing consolidate_text_memory method"
    # We will test consolidating a low-confidence fact into weights
    w_before = dream.weights.clone()
    dream.consolidate_text_memory("Kuantum parçacık dalga fonksiyonu ile çalışır.", wernicke=wernicke)
    w_after = dream.weights.clone()
    assert not torch.equal(w_before, w_after), "Weights did not change after consolidation"
    print("  [OK] NREM text memory consolidation successfully updated weights.")

    # 4. Test REM active concept extraction & synthesis
    print("\n[Test 4] Testing REM active concept extraction & synthesis...")
    assert hasattr(dream, "get_active_dream_concepts"), "MergenDream is missing get_active_dream_concepts method"
    # Artificially set weights of a couple concepts high to simulate co-activation
    dream.weights[100, 5] = 2.0
    dream.weights[101, 10] = 2.0
    
    active_concepts = dream.get_active_dream_concepts(vocab_words, top_n=2)
    assert len(active_concepts) >= 2, "Failed to extract active dream concepts"
    print(f"  [OK] Extracted active dream concepts: {active_concepts}")

    # 5. Test Limbic integration (Dream Diary & Dynamic Sleep Cycles)
    print("\n[Test 5] Testing Limbic integration (Dream Diary)...")
    # Simulate Limbic dreaming and diary generation
    # We will call a method that writes the diary
    assert hasattr(limbic, "write_dream_diary"), "LimbicExecutiveLayer is missing write_dream_diary method"
    limbic.write_dream_diary(["kuantum", "parçacık", "dalga"])
    assert os.path.exists(DIARY_PATH), "dream_diary.json was not created"
    
    with open(DIARY_PATH, 'r', encoding='utf-8') as f:
        diary_data = json.load(f)
    assert "concepts" in diary_data
    assert "kuantum" in diary_data["concepts"]
    print("  [OK] Dream diary written and validated.")

    # 6. Test ResponseGenerator for dream query response
    print("\n[Test 6] Testing ResponseGenerator for dream query response...")
    resp_gen = ResponseGenerator(vocab=None, brain=limbic)
    
    # We ask about the dream
    response = resp_gen.generate(
        query="ne rüyası gördün",
        intent="INQUIRY",
        subject=None,
        knowledge_facts=[]
    )
    assert "rüya" in response.lower() or "kuantum" in response.lower(), f"Unexpected response: {response}"
    print(f"  [OK] ResponseGenerator returned dream response: {response}")

    print("\n" + "=" * 60)
    print(" ALL DREAM CONSOLIDATION TESTS PASSED SUCCESSFULLY! [OK]")
    print("=" * 60)

# Cleanup
def cleanup():
    try:
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
    except Exception as e:
        print(f"Warning: Failed to cleanup DB directory {DB_PATH}: {e}")

    try:
        if os.path.exists(WEIGHTS_PATH):
            os.remove(WEIGHTS_PATH)
    except Exception as e:
        print(f"Warning: Failed to cleanup weights file {WEIGHTS_PATH}: {e}")

    try:
        if os.path.exists(DIARY_PATH):
            os.remove(DIARY_PATH)
    except Exception as e:
        print(f"Warning: Failed to cleanup diary path {DIARY_PATH}: {e}")

if __name__ == "__main__":
    try:
        test_dream_system()
    finally:
        cleanup()

