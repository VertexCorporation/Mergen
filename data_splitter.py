from pathlib import Path

def ucur_beni():
    # Klasör ve dosya yolları
    hedef_klasor = Path("data") / "training"
    orjinal_dosya = hedef_klasor / "mergen_memory_test_50000.txt"
    
    part1_dosya = hedef_klasor / "mergen_memory_test_50000_part1.txt"
    part2_dosya = hedef_klasor / "mergen_memory_test_50000_part2.txt"

    if not orjinal_dosya.exists():
        print(f"Hata: {orjinal_dosya} bulunamadı.")
        return

    # 1. Hamle: Tüm dosyayı tek seferde RAM'e oku
    with open(orjinal_dosya, 'r', encoding='utf-8') as f:
        satirlar = f.readlines()

    # 2. Hamle: Bellek (RAM) üzerinde listeyi ikiye dilimle (Slicing)
    # Python'ın C-seviyesindeki optimizasyonu sayesinde bu işlem mikrosaniyeler sürer.
    part1_icerik = "".join(satirlar[:25000])
    part2_icerik = "".join(satirlar[25000:])

    # 3. Hamle: Tek bir write operasyonuyla diske fırlat
    with open(part1_dosya, 'w', encoding='utf-8') as f1:
        f1.write(part1_icerik)
        
    with open(part2_dosya, 'w', encoding='utf-8') as f2:
        f2.write(part2_icerik)

    print("İşlem jet hızıyla tamamlandı!")

if __name__ == "__main__":
    ucur_beni()