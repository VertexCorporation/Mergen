"""
MERGEN BIOLOGICAL CORE — FULL SYSTEM VERIFICATION
===================================================
Tests every layer of the cognitive architecture end-to-end:
  1. Vocabulary
  2. Wernicke (Perception)
  3. Hebbian Engine + Innate Priors
  4. Limbic Executive Layer (Dual-Pathway)
  5. STDP + Dopamine Plasticity
  6. DMN (Default Mode Network) Spontaneous Firing
"""
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mergen_vocab import MergenVocab
from cognitive.wernicke_area import WernickeArea
from learning.cortical_column import CorticalColumn
from cognitive.limbic_executive_layer import LimbicExecutiveLayer
from cognitive.broca_area import BrocaArea


def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title} ")
    print("=" * 60)


def main():
    print_header("MERGEN BIOLOGICAL CORE - FULL SYSTEM VERIFICATION")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[System] Active Computation Device: {device.upper()}")

    passed = 0
    total = 6

    # ---------------------------------------------------------
    # 1. Vocabulary Verification
    # ---------------------------------------------------------
    print("\n[1/6] Verifying Vocabulary...")
    vocab_path = './mergen_vocab.json'
    try:
        vocab = MergenVocab.load(vocab_path)
        all_words = vocab.all_words
        print(f"  [OK] Loaded vocabulary: {vocab.size()} tokens")
        print(f"  [OK] Sample words: {all_words[6:11]}")
        assert vocab.size() > 10, "Vocab too small"
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Vocabulary loading failed: {e}")
        return

    # ---------------------------------------------------------
    # 2. Wernicke Area Verification (Perception)
    # ---------------------------------------------------------
    print("\n[2/6] Verifying Wernicke Area (Layer 1: Perceive)...")
    try:
        wernicke = WernickeArea(
            n_neurons=768,
            embedding_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            encoding='rate',
            time_window=10,
            device=device
        )
        test_sentence = "Kuantum parçacık dalga fonksiyonu ile çalışır."
        spike_train = wernicke.perceive(test_sentence)
        print(f"  [OK] Spike train shape: {spike_train.shape}")
        assert spike_train.shape == (10, 768), f"Shape mismatch: {spike_train.shape}"

        # Verify spike activity exists
        spike_rate = (spike_train > 0).float().mean().item()
        print(f"  [OK] Mean spike rate: {spike_rate:.2%}")
        # A 0.8% - 2.0% spike rate is normal due to L2 normalization + dual rail ReLU sparsity
        assert spike_rate > 0.001, "Spike train is effectively silent"
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Wernicke Area failed: {e}")
        return

    # ---------------------------------------------------------
    # 3. Cortical Column & Innate Priors (Memory/Weights)
    # ---------------------------------------------------------
    print("\n[3/6] Verifying Cortical Column & Cortical Priors...")
    try:
        hebbian_engine = CorticalColumn(
            n_pre=768,
            n_post=vocab.size(),
            n_hidden=1024,
            lateral_k=50,
            device=device
        )
        priors_state = torch.load(
            './mergen_cortical_priors.pt',
            map_location=device,
            weights_only=True
        )
        assert isinstance(priors_state, dict) and priors_state.get('version') == '2.0', \
            "Priors format mismatch (expected version 2.0 dict)"
        
        hebbian_engine.L4.weights.data = priors_state['L4_weights'].to(device)
        hebbian_engine.L23.weights.data = priors_state['L23_weights'].to(device)
        hebbian_engine.L5.weights.data = priors_state['L5_weights'].to(device)
        
        print(f"  [OK] Cortical Priors loaded successfully")
        print(f"  [OK] L4 shape:  {hebbian_engine.L4.weights.shape}")
        print(f"  [OK] L23 shape: {hebbian_engine.L23.weights.shape}")
        print(f"  [OK] L5 shape:  {hebbian_engine.L5.weights.shape}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Cortical Column / Priors failed: {e}")
        return

    # ---------------------------------------------------------
    # 4. Limbic Executive Layer Verification (Dual-Pathway)
    # ---------------------------------------------------------
    print("\n[4/6] Verifying Limbic Executive Layer (Dual-Pathway)...")
    try:
        # BUG FIX: BrocaArea expects `concept_vocabulary=` not `vocab=`
        broca = BrocaArea(
            n_neurons=vocab.size(),
            concept_vocabulary=all_words,
            device=device
        )
        limbic = LimbicExecutiveLayer(
            mergen_engine=hebbian_engine,
            broca=broca,
            wernicke=wernicke
        )

        print("  Testing Dual-Pathway Thought Generation...")
        response = limbic.respond(test_sentence, max_attempts=1)
        print(f"  [OK] Fired Neurons (Intent): {limbic.last_thought}")
        print(f"  [OK] Broca Response: {response}")
        assert limbic.last_thought is not None and len(limbic.last_thought) > 0, \
            "No thought generated"

        # Verify thoughts contain real words, not word_NNN placeholders
        for concept in limbic.last_thought.split(" -> "):
            assert not concept.startswith("word_"), \
                f"Placeholder concept detected: {concept}"
        print("  [OK] All concepts are real vocabulary words.")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Limbic Executive Layer failed: {e}")
        return

    # ---------------------------------------------------------
    # 5. Dopamine & STDP Plasticity Verification
    # ---------------------------------------------------------
    print("\n[5/6] Verifying STDP & Dopamine Modulator...")
    try:
        # Generate a fresh learning signal so eligibility isn't empty
        fresh_spike_train = wernicke.perceive("Dopamin ödül sinyalidir.")
        for t in range(fresh_spike_train.shape[0]):
            pre = fresh_spike_train[t]
            post = hebbian_engine.forward(pre)
            hebbian_engine.update_traces(pre, post)

        w_before = hebbian_engine.weights.data.clone().mean().item()
        hebbian_engine.apply_dopamine(reward=1.0)
        w_after = hebbian_engine.weights.data.mean().item()
        delta_w = abs(w_after - w_before)

        telemetry = hebbian_engine.get_telemetry()
        print(f"  [OK] Mean weight before reward: {w_before:.6f}")
        print(f"  [OK] Mean weight after reward:  {w_after:.6f}")
        print(f"  [OK] Weight change (delta_w):   {delta_w:.6f}")
        print(f"  [OK] RPE: {telemetry['rpe']:.4f}")
        print(f"  [OK] LTP magnitude: {telemetry['ltp']:.6f}")
        print(f"  [OK] LTD magnitude: {telemetry['ltd']:.6f}")
        assert delta_w > 0, "Dopamine did not change weights — plasticity broken"
        print("  [OK] Plasticity mechanisms functional.")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] STDP / Dopamine failed: {e}")
        return

    # ---------------------------------------------------------
    # 6. DMN Spontaneous Firing Verification
    # ---------------------------------------------------------
    print("\n[6/6] Verifying Default Mode Network (Spontaneous Firing)...")
    try:
        dmn_before = limbic.dmn_cycles
        limbic._spontaneous_fire()
        dmn_after = limbic.dmn_cycles
        print(f"  [OK] DMN cycles before: {dmn_before}")
        print(f"  [OK] DMN cycles after:  {dmn_after}")
        assert dmn_after == dmn_before + 1, "DMN cycle counter not incremented"

        # Verify a spontaneous thought was generated
        if limbic.internal_thoughts:
            last_thought = list(limbic.internal_thoughts)[-1]
            print(f"  [OK] Spontaneous thought: \"{last_thought.get('thought', '?')}\"")
        print("  [OK] Default Mode Network functional.")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] DMN Spontaneous Firing failed: {e}")
        return

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print_header(f"RESULT: {passed}/{total} TESTS PASSED")
    if passed == total:
        print("  [OK] ALL BIOLOGICAL SYSTEMS VERIFIED AND OPERATIONAL")
    else:
        print(f"  ✗ {total - passed} test(s) failed — review output above")


if __name__ == "__main__":
    main()
