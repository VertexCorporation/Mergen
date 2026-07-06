# Mergen Training Curriculum

Bu klasor Mergen icin kontrollu experience egitim setlerini tutar.

## Dosyalar

- `core_curriculum_v1.txt`: Cekirdek kavram egitimi. Her bos olmayan satir tek bir experience olarak tasarlanmistir.

## Kullanim

Kucuk smoke egitimi:

```powershell
python scripts/simulation_playground.py --data data/training/core_curriculum_v1.txt --limit 10 --dream-cycles 1
```

Kademeli egitim:

```powershell
python scripts/simulation_playground.py --data data/training/core_curriculum_v1.txt --limit 50 --dream-cycles 5 --strict-eval --bridge-timeout 30
python scripts/simulation_playground.py --data data/training/core_curriculum_v1.txt --limit 100 --dream-cycles 10 --strict-eval --bridge-timeout 30
python scripts/simulation_playground.py --data data/training/core_curriculum_v1.txt --dream-cycles 25 --strict-eval --bridge-timeout 60
```

Her buyuk egitimden sonra:

```powershell
python scripts/verify_all_layers.py
```

## Veri Kurallari

- Her satir tek ana fikir tasimali.
- Soru cumleleri egitim verisine konmamali.
- Celiskili bilgi ayni dosyada bulunmamali.
- Onemli kavramlar farkli cumlelerle tekrar edilmeli.
- Yeni alanlar ayri dosyada tutulmali ve once kucuk limitlerle denenmeli.
