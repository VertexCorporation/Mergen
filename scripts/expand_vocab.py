import json

with open('mergen_vocab.json', 'r', encoding='utf-8') as f:
    v = json.load(f)

words = set(v['all_words'])
words.update([
    'ağaç', 'orman', 'yaprak', 'güneş', 'ısı', 'yaşam', 'yapay', 'zeka', 
    'insan', 'dünya', 'su', 'hava', 'toprak', 'hayvan', 'doğa', 'bitki', 
    'ışık', 'gece', 'gündüz', 'şehir', 'yıldız', 'evren', 'bilim', 'sanat', 
    'kalp', 'beyin', 'zihin', 'duygu', 'düşünce', 'ev', 'yol', 'deniz', 
    'okyanus', 'dağ'
])

# Ensure special tokens are at the beginning
special = ['<bos>', '<eos>', '<pad>', '<unk>', '<sep>', '<cls>']
words = [w for w in words if w not in special]
v['all_words'] = special + sorted(words)
v['word_to_id'] = {w: i for i, w in enumerate(v['all_words'])}

with open('mergen_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(v, f, ensure_ascii=False, indent=2)

print(f"Vocab size updated to {len(v['all_words'])}")
