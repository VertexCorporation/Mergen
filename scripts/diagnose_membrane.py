"""
Diagnostic: Measure the actual membrane potential distribution
when innate priors process sentences through the Hebbian engine.
"""
import os, sys, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mergen_vocab import MergenVocab
from learning import HybridHebbianLearner

def main():
    vocab = MergenVocab.load('./mergen_vocab.json')
    
    engine = HybridHebbianLearner(
        n_pre=768,
        n_post=vocab.size(),
        spike_threshold=1.0,  # default
        device='cpu'
    )
    
    # Load innate priors
    priors = torch.load('./mergen_innate_priors.pt', map_location='cpu')
    if priors.shape == engine.weights.shape:
        engine.weights.data = priors
        print(f"Loaded innate priors: {priors.shape}")
    else:
        print(f"Shape mismatch: priors={priors.shape}, weights={engine.weights.shape}")
        return

    # Load Wernicke
    from cognitive.wernicke_area import WernickeArea
    wernicke = WernickeArea(
        n_neurons=768,
        embedding_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        encoding='rate',
        time_window=10,
        device='cpu'
    )

    sentences = [
        "Kuantum parcacik dalga fonksiyonu ile calisir.",
        "Kutlecekim uzay ve zamani bukerek kara delikleri olusturur.",
        "Agaclar ormanda buyur, yapraklari entropiyi azaltir.",
        "Yapay zeka sinir aglari ve makine ogrenmesi kullanir.",
        "Kozmos galaksiler, yildizlar ve gezegenlerle doludur.",
    ]

    for sent in sentences:
        spike_train = wernicke.perceive(sent)
        
        # Accumulate neural intent over time steps
        membrane_accum = torch.zeros(vocab.size())
        for t in range(spike_train.shape[0]):
            pre = spike_train[t]
            membrane = torch.matmul(pre, engine.weights.data)
            membrane_accum += membrane
        
        # Skip special tokens
        vals = membrane_accum[6:]
        
        top5_vals, top5_idx = torch.topk(vals, 5)
        top5_idx += 6  # offset for special tokens
        
        print(f"\n--- {sent[:50]} ---")
        print(f"  Membrane stats: min={vals.min():.4f}, max={vals.max():.4f}, "
              f"mean={vals.mean():.4f}, std={vals.std():.4f}")
        print(f"  Neurons > 1.0: {(vals > 1.0).sum().item()}")
        print(f"  Neurons > 0.5: {(vals > 0.5).sum().item()}")
        print(f"  Neurons > 0.3: {(vals > 0.3).sum().item()}")
        print(f"  Neurons > 0.1: {(vals > 0.1).sum().item()}")
        print(f"  Top-5 concepts:")
        for v, i in zip(top5_vals, top5_idx):
            word = vocab.id_to_word(i.item())
            print(f"    [{i.item():3d}] {word:20s} membrane={v:.4f}")

if __name__ == "__main__":
    main()
