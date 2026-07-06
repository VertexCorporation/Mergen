import sys
import json
import torch

sys.stdout.reconfigure(encoding='utf-8')

p = torch.load('./mergen_innate_priors.pt', map_location='cpu', weights_only=True)
print(f"New priors shape: {p.shape}  (expected: torch.Size([768, 1136]))")

with open('mergen_vocab.json', 'r', encoding='utf-8') as f:
    v = json.load(f)

print(f"JSON all_words count   : {len(v['all_words'])}")
print(f"JSON word_categories   : {len(v['word_categories'])}")

missing = [w for w in v['all_words'] if w not in v['word_categories']]
print(f"Words missing category : {missing}")

print("\nCategory ranges:")
for cat, (start, end) in v['category_ranges'].items():
    sample = v['all_words'][start:start+3]
    print(f"  {cat:25s}: [{start:4d}, {end:4d})  size={end-start:3d}  sample={sample}")
