"""
╔══════════════════════════════════════════════════════════════════════╗
║         MERGEN V3 — MERGEN VOCAB (Intellectual Depth)               ║
║                                                                      ║
║  "A mind is only as rich as the words it knows."                    ║
║                                                                      ║
║  800+ words across: Physics, Software, Philosophy, Daily Life       ║
║  Author:  Vertex Corporation — Mergen Project                       ║
║  License: Apache-2.0                                                ║
╚══════════════════════════════════════════════════════════════════════╝

DESIGN PRINCIPLES:
━━━━━━━━━━━━━━━━━━
1. Categorized Growth: Words organized by domain for balanced learning
2. Bilingual: Turkish + English roots where meaningful
3. Neural-Aligned: OUTPUT_SIZE = len(all_words) — automatic sync
4. No Hardcoded Dimensions: All downstream layers read from vocab.size()
5. Dynamic Expansion: Add words at runtime without breaking matrices

INTEGRATION RULE:
━━━━━━━━━━━━━━━━
Every nn.Linear that outputs to vocabulary must use:
    nn.Linear(HIDDEN_DIM, vocab.size())   ✓ correct
    nn.Linear(HIDDEN_DIM, 100)            ✗ WRONG — will break on expansion
"""

from typing import List, Dict, Optional, Set
from pathlib import Path
import json


class MergenVocab:
    """
    Rich categorized vocabulary for Mergen's language engine.

    Usage:
        vocab = MergenVocab()
        print(f"Total words: {vocab.size()}")  # ~800+

        # In neural architecture:
        self.mx2 = nn.Linear(config.HIDDEN_DIM, vocab.size())

        # Lookups
        idx = vocab.word_to_id['kuantum']
        word = vocab.id_to_word(idx)

        # Dynamic growth
        vocab.add_word('yeni_kelime', category='technical')
        self.mx2 = nn.Linear(config.HIDDEN_DIM, vocab.size())  # auto-resize
    """

    # ═════════════════════════════════════════════════════════
    #  SPECIAL TOKENS (always at the beginning — fixed IDs)
    # ═════════════════════════════════════════════════════════
    SPECIAL_TOKENS = ['<bos>', '<eos>', '<pad>', '<unk>', '<sep>', '<cls>']

    # ═════════════════════════════════════════════════════════
    #  CATEGORY 1: PHYSICS & MATHEMATICS TERMINOLOGY
    # ═════════════════════════════════════════════════════════
    PHYSICS_TERMS = [
        # Quantum & Particle Physics
        'kuantum', 'dalga', 'parçacık', 'foton', 'elektron', 'proton',
        'nötron', 'atom', 'molekül', 'spin', 'kuark', 'lepton', 'boson',
        'higgs', 'süperpozisyon', 'dolanıklık', 'tünelleme', 'ışınım',
        # Relativity & Cosmology
        'rölativite', 'uzay', 'zaman', 'uzay-zaman', 'kütleçekim',
        'karadelik', 'galaksi', 'nötron-yıldızı', 'singülarite',
        'evren', 'çokluevren', 'paralel', 'boyut', 'eğrilik',
        # Thermodynamics & Statistical Mechanics
        'entropi', 'enerji', 'entalpi', 'ısı', 'sıcaklık', 'basınç',
        'yoğunluk', 'frekans', 'rezonans', 'osilasyon', 'dalgaboyu',
        # Classical & Analytical Mechanics
        'kuvvet', 'ivme', 'hız', 'momentum', 'moment', 'vektör',
        'skaler', 'tensör', 'alan', 'potansiyel', 'kinetik',
        # Mathematics
        'integral', 'türev', 'limit', 'matris', 'determinant', 'vektör',
        'fonksiyon', 'denklem', 'türbülans', 'logaritma', 'eksponansiyel',
        'topoloji', 'manifold', 'eğri', 'yüzey', 'hiperbolik', 'fraktal',
        'stokastik', 'olasılık', 'dağılım', 'varyans', 'korelasyon',
        'gradyan', 'divergens', 'rotasyon', 'diferansiyel', 'lineer',
        'nonlineer', 'kaos', 'attraktör', 'bifurkasyon', 'salınım',
        # English equivalents
        'quantum', 'entropy', 'relativity', 'singularity', 'manifold',
        'gradient', 'tensor', 'eigenvalue', 'eigenvector', 'laplacian',
    ]

    # ═════════════════════════════════════════════════════════
    #  CATEGORY 2: SOFTWARE & COMPUTING TERMINOLOGY
    # ═════════════════════════════════════════════════════════
    SOFTWARE_TERMS = [
        # Programming concepts
        'derleme', 'yorumlama', 'değişken', 'fonksiyon', 'sınıf', 'nesne',
        'metod', 'parametre', 'argüman', 'döngü', 'koşul', 'özyineleme',
        'iteratif', 'algoritma', 'veri-yapısı', 'dizi', 'liste', 'sözlük',
        'küme', 'ağaç', 'graf', 'yığın', 'kuyruk', 'bağlı-liste',
        # AI / ML / Neuroscience
        'yapay-zeka', 'makine-öğrenmesi', 'derin-öğrenme', 'sinir-ağı',
        'nöron', 'sinaps', 'ateşleme', 'spike', 'aktivasyon', 'eşik',
        'ağırlık', 'öğrenme-oranı', 'gradyan-inişi', 'geriyayılım',
        'ileriyayılım', 'epoch', 'batch', 'optimizasyon', 'adam', 'sgd',
        'overfitting', 'regularizasyon', 'dropout', 'embedding',
        'transformer', 'attention', 'recurrent', 'konvolüsyon',
        'hebbian', 'stdp', 'plastisite', 'konsolidasyon',
        # Systems & Infrastructure
        'işlemci', 'gpu', 'bellek', 'önbellek', 'disk', 'ağ', 'protokol',
        'sunucu', 'istemci', 'bulut', 'konteyner', 'sanal', 'paralel',
        'eşzamanlı', 'asenkron', 'thread', 'process', 'mutex',
        # Data & APIs
        'veri', 'veri-seti', 'veri-tabanı', 'sorgu', 'indeks', 'json',
        'api', 'endpoint', 'token', 'kimlik-doğrulama', 'şifreleme',
        'hash', 'blockchain', 'özet',
        # Web / UI
        'arayüz', 'bileşen', 'render', 'dom', 'stil', 'olay',
        # English equivalents
        'compile', 'recursion', 'polymorphism', 'abstraction',
        'encapsulation', 'inheritance', 'framework', 'library',
        'repository', 'branch', 'merge', 'commit', 'debugging',
        'deployment', 'pipeline', 'microservice',
    ]

    # ═════════════════════════════════════════════════════════
    #  CATEGORY 3: PHILOSOPHY & COGNITION
    # ═════════════════════════════════════════════════════════
    PHILOSOPHY_TERMS = [
        # Consciousness & Mind
        'bilinç', 'zihin', 'düşünce', 'farkındalık', 'deneyim', 'algı',
        'nitel', 'nicel', 'sübjektif', 'objektif', 'fenomen', 'nous',
        'qualia', 'epifenomen', 'idealizm', 'materyalizm', 'dualizm',
        # Epistemology
        'bilgi', 'gerçek', 'hakikat', 'kanıt', 'sav', 'önerme', 'aksiyom',
        'teorem', 'paradoks', 'çelişki', 'tutarlılık', 'tümevarım',
        'tümdengelim', 'kıyas', 'analiz', 'sentez',
        # Ethics & Values
        'değer', 'ahlak', 'etik', 'erdem', 'iyi', 'kötü', 'adalet',
        'özgürlük', 'sorumluluk', 'irade', 'niyet',
        # Metaphysics
        'varlık', 'olmak', 'olmayan', 'yokluk', 'öz', 'töz', 'biçim',
        'madde', 'ruh', 'evrim', 'oluş', 'değişim', 'devinim',
        # Concepts
        'anlam', 'amaç', 'neden', 'sonuç', 'ilke', 'yasa', 'düzen',
        'kaos', 'denge', 'uyum', 'çelişki', 'diyalektik', 'sentez',
        'tez', 'antitez', 'aşkınlık', 'içkinlik', 'mutlak', 'göreli',
        # English equivalents
        'consciousness', 'sentience', 'awareness', 'existence',
        'essence', 'ontology', 'epistemology', 'phenomenology',
        'teleology', 'dialectic', 'paradigm', 'a-priori', 'a-posteriori',
    ]

    # ═════════════════════════════════════════════════════════
    #  CATEGORY 4: DAILY LIFE & SOCIAL
    # ═════════════════════════════════════════════════════════
    DAILY_TERMS = [
        # Pronouns & identity
        'ben', 'sen', 'o', 'biz', 'siz', 'onlar', 'kendim', 'kendin',
        'kendisi', 'kendimiz', 'kendiniz', 'kendileri',
        # Common nouns
        'gün', 'gece', 'sabah', 'akşam', 'saat', 'dakika', 'saniye',
        'hafta', 'ay', 'yıl', 'yüzyıl', 'an', 'şimdi', 'önce', 'sonra',
        'ev', 'okul', 'iş', 'şehir', 'ülke', 'dünya', 'kitap', 'kalem',
        'su', 'yemek', 'hava', 'ateş', 'toprak',
        # People & relations
        'insan', 'kişi', 'arkadaş', 'aile', 'anne', 'baba', 'çocuk',
        'öğretmen', 'öğrenci', 'doktor', 'bilim-insanı', 'mühendis',
        # Greetings & social
        'merhaba', 'selam', 'selamlar', 'günaydın', 'teşekkür',
        'rica', 'lütfen', 'evet', 'hayır', 'belki', 'tamam', 'peki',
        # English equivalents
        'hello', 'hi', 'yes', 'no', 'maybe', 'please', 'thanks',
        'today', 'yesterday', 'tomorrow', 'morning', 'night',
    ]

    # ═════════════════════════════════════════════════════════
    #  CATEGORY 5: ADJECTIVES (Sıfatlar)
    # ═════════════════════════════════════════════════════════
    ADJECTIVES = [
        # Dynamic & systemic
        'dinamik', 'statik', 'stabil', 'kararsız', 'dengeli', 'dengesiz',
        'hibrit', 'saf', 'karışık', 'homojen', 'heterojen', 'modüler',
        'eşzamanlı', 'asenkron', 'paralel', 'seri', 'ardışık', 'rastgele',
        # Scale & complexity
        'basit', 'karmaşık', 'derin', 'yüzeysel', 'minimal', 'kapsamlı',
        'mikroskobik', 'makroskobik', 'büyük', 'küçük', 'sonsuz', 'sonlu',
        'kesintili', 'sürekli', 'ayrık', 'kesişen', 'ayrı',
        # Physics-flavored
        'rölativistik', 'kuantize', 'klasik', 'termal', 'elektromanyetik',
        'manyetik', 'elektriksel', 'gravitasyonel', 'nükleer', 'atomik',
        'fraktal', 'kaotik', 'düzenli', 'düzensiz', 'periyodik',
        # Cognitive
        'bilinçli', 'bilinçsiz', 'sezgisel', 'analitik', 'rasyonel',
        'emosyonel', 'mantıksal', 'paradoksal', 'tutarlı', 'çelişkili',
        'apaçık', 'örtük', 'açık', 'kapalı', 'gizli', 'belirgin',
        # Quality
        'doğru', 'yanlış', 'kesin', 'belirsiz', 'olası', 'imkansız',
        'zorunlu', 'isteğe-bağlı', 'kritik', 'marjinal', 'merkezi',
        'optimal', 'suboptimal', 'yeterli', 'yetersiz', 'fazla', 'az',
        # English equivalents
        'dynamic', 'chaotic', 'stable', 'hybrid', 'relativistic',
        'complex', 'simple', 'synchronous', 'parallel', 'fractal',
        'quantum', 'emergent', 'recursive', 'iterative', 'adaptive',
        'robust', 'fragile', 'sparse', 'dense',
    ]

    # ═════════════════════════════════════════════════════════
    #  CATEGORY 6: VERBS / ACTIONS (Eylemler)
    # ═════════════════════════════════════════════════════════
    VERBS = [
        # Cognitive
        'düşünmek', 'anlamak', 'bilmek', 'tanımak', 'hatırlamak',
        'unutmak', 'öğrenmek', 'öğretmek', 'keşfetmek', 'sorgulamak',
        'çıkarsamak', 'yorumlamak', 'algılamak', 'fark-etmek',
        'hissetmek', 'sezmek', 'tahmin-etmek', 'hesaplamak',
        # Process
        'sentezlemek', 'analiz-etmek', 'optimize-etmek', 'simüle-etmek',
        'modellemek', 'koordine-etmek', 'entegre-etmek', 'ayrıştırmak',
        'ivmelenmek', 'yavaşlamak', 'hızlanmak', 'evrilmek', 'dönüşmek',
        'büyümek', 'küçülmek', 'genişlemek', 'daralmak', 'uyum-sağlamak',
        'adapte-olmak', 'normalize-etmek', 'parametrelemek',
        # Communication
        'söylemek', 'anlatmak', 'açıklamak', 'tanımlamak', 'betimlemek',
        'sormak', 'cevaplamak', 'tartışmak', 'onaylamak', 'reddetmek',
        'kabul-etmek', 'çelişmek', 'doğrulamak', 'yanlışlamak',
        'önermek', 'iddia-etmek', 'kanıtlamak', 'göstermek',
        # Action
        'yapmak', 'etmek', 'oluşturmak', 'üretmek', 'yaratmak', 'inşa-etmek',
        'yıkmak', 'değiştirmek', 'korumak', 'geliştirmek', 'iyileştirmek',
        'düzenlemek', 'çözmek', 'bağlamak', 'ayırmak', 'birleştirmek',
        # Existence
        'olmak', 'bulunmak', 'var-olmak', 'gerçekleşmek', 'ortaya-çıkmak',
        'kaybolmak', 'belirmek', 'görünmek',
        # English equivalents
        'synthesize', 'optimize', 'simulate', 'iterate', 'evolve',
        'accelerate', 'decelerate', 'converge', 'diverge', 'propagate',
        'infer', 'deduce', 'induce', 'observe', 'compute', 'encode',
        'decode', 'learn', 'forget', 'recall',
    ]

    # ═════════════════════════════════════════════════════════
    #  CATEGORY 7: CONNECTIVES & ADVERBS (Bağlaçlar, Zarflar)
    # ═════════════════════════════════════════════════════════
    CONNECTIVES = [
        # Logical connectives
        've', 'veya', 'ama', 'fakat', 'lakin', 'ancak', 'ya-da', 'ne-de',
        'hem', 'hem-de', 'ise', 'oysa', 'halbuki', 'rağmen',
        # Causal
        'çünkü', 'zira', 'dolayısıyla', 'bu-yüzden', 'bu-nedenle',
        'sonuç-olarak', 'dolayısıyla', 'böylece', 'dolayısı',
        'sebebiyle', 'ötürü', 'varolarak',
        # Temporal
        'şimdi', 'önce', 'sonra', 'ardından', 'eşzamanlı', 'aynı-anda',
        'sürekli', 'ara-sıra', 'bazen', 'daima', 'hiçbir-zaman',
        'başlangıçta', 'sonunda', 'halen', 'henüz',
        # Modal adverbs
        'muhtemelen', 'kesinlikle', 'şüphesiz', 'belki', 'olasılıkla',
        'kuşkusuz', 'kesin', 'gerçekten', 'aslında', 'özellikle',
        'genel-olarak', 'özel-olarak', 'temelde', 'esasen',
        'teorik-olarak', 'pratik-olarak', 'yapısal-olarak',
        'fonksiyonel-olarak', 'matematiksel-olarak', 'fiziksel-olarak',
        # Degree
        'çok', 'az', 'biraz', 'oldukça', 'gayet', 'son-derece',
        'tamamen', 'kısmen', 'neredeyse', 'hemen-hemen',
        # English equivalents
        'and', 'or', 'but', 'however', 'therefore', 'thus', 'hence',
        'because', 'since', 'although', 'despite', 'whereas',
        'meanwhile', 'simultaneously', 'consequently',
        'probably', 'certainly', 'perhaps', 'definitely', 'theoretically',
        'practically', 'structurally', 'fundamentally', 'essentially',
    ]

    # ═════════════════════════════════════════════════════════
    #  CATEGORY 8: PUNCTUATION & SYMBOLS
    # ═════════════════════════════════════════════════════════
    PUNCTUATION = [
        '.', ',', '?', '!', ':', ';', '-', '—', '(', ')', '"', "'",
        '...', '…', '/',
    ]

    # ═════════════════════════════════════════════════════════
    #  CATEGORY 9: GÜNLÜK TÜRKÇE (Common Turkish)
    # ═════════════════════════════════════════════════════════
    TURKISH_COMMON = [
        # Doğa ve çevre
        'yol', 'taş', 'toprak', 'kum', 'kar', 'yağmur', 'bulut', 'güneş',
        'ay', 'yıldız', 'gök', 'deniz', 'göl', 'nehir', 'dağ', 'orman',
        'ağaç', 'çiçek', 'yaprak', 'kök', 'dal', 'tohum', 'bitki',
        'hayvan', 'kuş', 'balık', 'böcek', 'kurt', 'aslan', 'at', 'köpek',
        # Mekan ve yapı
        'köprü', 'bina', 'kapı', 'pencere', 'tavan', 'zemin', 'duvar',
        'masa', 'sandalye', 'yatak', 'mutfak', 'bahçe', 'park', 'cadde',
        'sokak', 'mahalle', 'köy', 'kasaba', 'ilçe', 'bölge', 'sınır',
        'hastane', 'kütüphane', 'müze', 'fabrika', 'çarşı', 'pazar',
        # Zaman ve süreç
        'kış', 'yaz', 'bahar', 'sonbahar', 'mevsim', 'dönem', 'çağ',
        'geçmiş', 'gelecek', 'tarih', 'süreç', 'aşama', 'adım', 'sıra',
        'sabah', 'öğle', 'akşam', 'gece', 'yarın', 'dün', 'bugün',
        # Nesne ve araç
        'araba', 'tren', 'uçak', 'gemi', 'bisiklet', 'motor',
        'telefon', 'ekran', 'kamera', 'mikrofon', 'hoparlör', 'pil',
        'kalem', 'defter', 'kağıt', 'kitap', 'çanta', 'şişe', 'kutu',
        'anahtar', 'kilit', 'ip', 'tel', 'boru', 'kablo', 'vida',
        # Yiyecek ve içecek
        'ekmek', 'et', 'sebze', 'meyve', 'çorba', 'pilav', 'kahve',
        'çay', 'süt', 'tuz', 'şeker', 'yağ', 'un', 'bal', 'peynir',
        # Para ve ekonomi
        'para', 'banka', 'fiyat', 'ücret', 'maliyet', 'kazanç', 'zarar',
        'borç', 'kredi', 'faiz', 'vergi', 'bütçe', 'gelir', 'gider',
        # İnsan ve toplum
        'toplum', 'halk', 'vatandaş', 'lider', 'yönetici', 'işçi',
        'sanatçı', 'sporcu', 'asker', 'polis', 'hakim', 'avukat',
        'gazeteci', 'politikacı', 'bilim-adamı', 'ressam', 'yazar',
        # Sağlık
        'sağlık', 'hastalık', 'ağrı', 'yara', 'ateş', 'grip', 'ameliyat',
        'tedavi', 'ilaç', 'vitamin', 'bağışıklık', 'kronik', 'akut',
        # Duygular ve psikoloji
        'sevgi', 'nefret', 'korku', 'merak', 'şaşkınlık', 'sevinç',
        'keder', 'öfke', 'kıskançlık', 'gurur', 'utanç', 'pişmanlık',
        'umut', 'hayal', 'arzu', 'istek', 'ihtiyaç', 'amaç', 'hedef',
        # Eğitim ve bilgi
        'eğitim', 'öğretim', 'ders', 'sınav', 'not', 'diploma', 'mezun',
        'araştırma', 'deney', 'sonuç', 'rapor', 'sunum', 'tartışma',
        'proje', 'görev', 'kural', 'yöntem', 'teknik', 'strateji',
    ]

    # ═════════════════════════════════════════════════════════
    #  CATEGORY 10: DOĞA BİLİMLERİ (Natural Sciences)
    # ═════════════════════════════════════════════════════════
    NATURAL_SCIENCE = [
        # Biyoloji
        'biyoloji', 'hücre', 'gen', 'dna', 'rna', 'protein', 'enzim',
        'bakteri', 'virüs', 'mantar', 'alg', 'bitki', 'hayvan', 'memeli',
        'sürüngen', 'omurgalı', 'omurgasız', 'evrim', 'adaptasyon',
        'mutasyon', 'seleksiyon', 'popülasyon', 'tür', 'nesil', 'kalıtım',
        'fenotip', 'genotip', 'kromozom', 'mitoz', 'mayoz', 'metabolizma',
        'fotosentez', 'solunum', 'sindirim', 'dolaşım', 'sinir', 'kas',
        'iskelet', 'beyin', 'nöron', 'sinaps', 'hormon', 'bağışıklık',
        'ekosistem', 'biyosfer', 'habitat', 'besin-zinciri',
        # Kimya
        'kimya', 'element', 'bileşik', 'karışım', 'molekül', 'iyon',
        'asit', 'baz', 'tuz', 'oksidasyon', 'redüksiyon', 'reaksiyon',
        'katalizör', 'çözünürlük', 'konsantrasyon', 'denge', 'entalpi',
        'entropi', 'orbital', 'bağ', 'polar', 'nonpolar', 'organik',
        'inorganik', 'polimer', 'monomer', 'karbohidrat', 'lipit',
        # Çevre
        'iklim', 'atmosfer', 'troposfer', 'ozon', 'sera-etkisi',
        'karbon', 'oksijen', 'azot', 'kirlilik', 'sürdürülebilir',
        'yenilenebilir', 'fosil', 'enerji-dönüşümü',
    ]

    # ═════════════════════════════════════════════════════════
    #  CATEGORY 11: SOSYAL BİLİMLER (Social Sciences)
    # ═════════════════════════════════════════════════════════
    SOCIAL_SCIENCE = [
        # Siyaset ve yönetim
        'demokrasi', 'cumhuriyet', 'monarşi', 'diktatörlük', 'seçim',
        'oy', 'parti', 'anayasa', 'yasa', 'hükümet', 'meclis', 'mahkeme',
        'yürütme', 'yasama', 'yargı', 'bağımsızlık', 'egemenlik',
        'insan-hakları', 'özgürlük', 'eşitlik', 'adalet', 'barış',
        # Ekonomi
        'ekonomi', 'piyasa', 'arz', 'talep', 'enflasyon', 'deflasyon',
        'büyüme', 'kriz', 'yatırım', 'sermaye', 'emek', 'üretim',
        'tüketim', 'ihracat', 'ithalat', 'ticaret', 'rekabet', 'tekel',
        'küreselleşme', 'liberalizm', 'sosyalizm', 'kapitalizm',
        # Sosyoloji ve kültür
        'kültür', 'medeniyet', 'uygarlık', 'gelenek', 'görenek', 'norm',
        'değer', 'kimlik', 'cinsiyet', 'ırk', 'etnik', 'din', 'inanç',
        'ideoloji', 'milliyetçilik', 'küreselleşme', 'göç', 'azınlık',
        'çoğunluk', 'dayanışma', 'çatışma', 'uzlaşma', 'diyalog',
        # Tarih
        'tarih', 'antik', 'ortaçağ', 'rönesans', 'aydınlanma', 'devrim',
        'reform', 'sömürgecilik', 'bağımsızlık', 'modernleşme', 'sanayi',
    ]

    # ═════════════════════════════════════════════════════════
    #  CATEGORY 12: GENİŞLETİLMİŞ FİİLLER (Extended Verbs)
    # ═════════════════════════════════════════════════════════
    EXTENDED_VERBS = [
        # Hareket
        'gitmek', 'gelmek', 'dönmek', 'geçmek', 'girmek', 'çıkmak',
        'inmek', 'çıkmak', 'atlamak', 'düşmek', 'kalkmak', 'oturmak',
        'uzanmak', 'eğilmek', 'dönmek', 'çevrilmek', 'sarılmak',
        # Algı
        'görmek', 'bakmak', 'izlemek', 'gözlemek', 'fark-etmek',
        'duymak', 'dinlemek', 'hissetmek', 'dokunmak', 'tatmak',
        'koklamak', 'algılamak', 'sezmek', 'kavramak',
        # İletişim
        'konuşmak', 'söylemek', 'anlatmak', 'açıklamak', 'sormak',
        'cevaplamak', 'tartışmak', 'müzakere', 'ikna', 'uyarmak',
        'önermek', 'istemek', 'emretmek', 'yasaklamak', 'izin',
        'paylaşmak', 'bildirmek', 'duyurmak', 'yayınlamak',
        # Zihin
        'düşünmek', 'planlamak', 'tasarlamak', 'hayal', 'hatırlamak',
        'unutmak', 'öğrenmek', 'anlamak', 'bilmek', 'tahmin',
        'hesaplamak', 'çözmek', 'karar', 'seçmek', 'değerlendirmek',
        # Üretim ve değişim
        'yapmak', 'üretmek', 'yaratmak', 'inşa', 'onarmak', 'bozmak',
        'dönüştürmek', 'geliştirmek', 'iyileştirmek', 'kötüleştirmek',
        'artırmak', 'azaltmak', 'genişletmek', 'daraltmak', 'birleştirmek',
        'ayırmak', 'bölmek', 'çarpmak', 'toplamak', 'çıkarmak',
    ]

    # ═════════════════════════════════════════════════════════
    #  CATEGORY 13: GENİŞLETİLMİŞ SIFATLAR (Extended Adjectives)
    # ═════════════════════════════════════════════════════════
    EXTENDED_ADJECTIVES = [
        # Fiziksel özellik
        'güzel', 'çirkin', 'uzun', 'kısa', 'geniş', 'dar', 'ağır',
        'hafif', 'hızlı', 'yavaş', 'sıcak', 'soğuk', 'ıslak', 'kuru',
        'sert', 'yumuşak', 'keskin', 'parlak', 'mat', 'şeffaf',
        'renkli', 'renksiz', 'pürüzlü', 'düzgün', 'oval', 'kare',
        # Kalite ve değer
        'iyi', 'kötü', 'doğru', 'yanlış', 'güvenilir', 'şüpheli',
        'kesin', 'belirsiz', 'net', 'bulanık', 'açık', 'kapalı',
        'gerçek', 'sahte', 'orijinal', 'kopya', 'yeni', 'eski',
        'modern', 'antik', 'ilkel', 'gelişmiş', 'basit', 'karmaşık',
        # Durum
        'aktif', 'pasif', 'hazır', 'hazırsız', 'tamamlanmış', 'eksik',
        'başarılı', 'başarısız', 'etkili', 'etkisiz', 'verimli', 'verimsiz',
        'güçlü', 'zayıf', 'sağlam', 'kırılgan', 'dayanıklı', 'geçici',
        'kalıcı', 'zorunlu', 'isteğe-bağlı', 'olası', 'imkansız',
        # Sosyal
        'sosyal', 'yalnız', 'dostane', 'düşmanca', 'resmi', 'gayriresmi',
        'kamusal', 'özel', 'bireysel', 'kolektif', 'ulusal', 'evrensel',
        'yerel', 'küresel', 'geleneksel', 'modern', 'liberal', 'muhafazakar',
    ]

    # ═════════════════════════════════════════════════════════
    #  CATEGORY 14: A1 TÜRKÇESİ (Günlük Temel Kelimeler)
    #  Kaynak: turkce_a1_kelimeler.txt — filtrelenmiş ve tekilleştirilmiş
    # ═════════════════════════════════════════════════════════
    A1_TURKISH = [
        # Kişiler ve ilişkiler
        'adam', 'kadın', 'erkek', 'kız', 'oğlan', 'bebek', 'komşu', 'misafir',
        'koca', 'eş', 'abla', 'ağabey', 'teyze', 'amca', 'hala', 'dayı',
        'büyükanne', 'büyükbaba', 'hemşire', 'aşçı', 'şoför', 'gazeteci',
        'ressam', 'müzisyen', 'şair', 'oyuncu', 'sporcu', 'fotoğrafçı',
        # Mekanlar
        'restoran', 'eczane', 'otel', 'sinema', 'tiyatro', 'havaalanı',
        'istasyon', 'market', 'fırın', 'kasap', 'manav', 'berber',
        'meydan', 'mahalle', 'ilçe', 'bölge', 'daire', 'apartman',
        'balkon', 'çatı', 'merdiven', 'asansör', 'garaj', 'bodrum',
        # Ulaşım
        'otobüs', 'tramvay', 'taksi', 'motosiklet', 'kamyon', 'minibüs',
        'metro', 'vapur', 'kavşak', 'trafik', 'durak', 'terminal',
        # Vücut ve sağlık
        'kalp', 'kafa', 'göz', 'kulak', 'burun', 'ağız', 'diş', 'dil',
        'saç', 'sakal', 'el', 'parmak', 'kol', 'bacak', 'ayak', 'omuz',
        'göğüs', 'sırt', 'karın', 'baş', 'boyun', 'yüz', 'alın', 'çene',
        'öksürük', 'ateş', 'grip', 'alerji', 'reçete', 'ameliyat',
        'muayene', 'ambulans', 'sigorta', 'iyileşme', 'tedavi',
        # Giyim ve aksesuar
        'elbise', 'gömlek', 'pantolon', 'etek', 'ceket', 'palto', 'kazak',
        'tişört', 'ayakkabı', 'çorap', 'şapka', 'eldiven', 'atkı', 'kemer',
        'yüzük', 'kolye', 'bilezik', 'gözlük', 'şemsiye', 'çanta',
        # Yiyecek ve içecek (A1 düzeyi - eksik olanlar)
        'ekmek', 'peynir', 'zeytin', 'domates', 'biber', 'patates', 'soğan',
        'salata', 'çorba', 'pilav', 'makarna', 'yumurta', 'tereyağı',
        'şeker', 'tuz', 'kahvaltı', 'meyve', 'elma', 'armut', 'muz',
        'portakal', 'çilek', 'üzüm', 'karpuz', 'kavun', 'sebze',
        'havuç', 'fasulye', 'nohut', 'mercimek', 'lahana', 'ıspanak',
        'pasta', 'kek', 'kurabiye', 'çikolata', 'dondurma', 'bisküvi',
        'ayran', 'limonata', 'portakal-suyu',
        # Renkler
        'kırmızı', 'mavi', 'yeşil', 'sarı', 'beyaz', 'siyah', 'kahverengi',
        'mor', 'turuncu', 'gri', 'pembe', 'altın', 'gümüş',
        # Sayılar (sözel)
        'sıfır', 'bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz',
        'dokuz', 'on', 'yirmi', 'otuz', 'kırk', 'elli', 'yüz', 'bin',
        # Yönler ve konumlar
        'sol', 'sağ', 'yukarı', 'aşağı', 'ileri', 'geri', 'iç', 'dış',
        'ön', 'arka', 'yan', 'karşı', 'üst', 'alt', 'orta', 'kenar',
        # Günler
        'pazartesi', 'salı', 'çarşamba', 'perşembe', 'cuma', 'cumartesi', 'pazar',
        # Mevsimler (ek)
        'ilkbahar',
        # Duygular (eksik olanlar)
        'nefret', 'şaşkınlık', 'sevinç', 'keder', 'öfke', 'kıskançlık',
        'gurur', 'utanç', 'pişmanlık', 'arzu', 'heyecan', 'hüzün', 'neşe',
        'dostluk', 'özgürlük', 'saygı', 'cesaret', 'sabır', 'güven',
        # Okul ve eğitim
        'tahta', 'tebeşir', 'silgi', 'cetvel', 'makas', 'tutkal', 'sıralar',
        'müdür', 'matematik', 'tarih', 'coğrafya', 'fizik', 'kimya',
        'biyoloji', 'resim', 'müzik', 'beden', 'tatil', 'ödev',
        # Ev eşyaları (eksik)
        'dolap', 'yatak', 'halı', 'perde', 'ayna', 'lamba', 'tablo',
        'çekmece', 'buzdolabı', 'ocak', 'kablo', 'pil', 'şişe', 'kutu',
        # Doğa olayları
        'fırtına', 'şelale', 'deprem', 'yangın', 'sel', 'kasırga',
        # Spor ve aktivite
        'futbol', 'basketbol', 'voleybol', 'tenis', 'yüzme', 'koşu',
        'güreş', 'boks', 'kayak', 'sörf', 'dalış', 'tırmanış', 'kamp',
        # Sanat ve müzik
        'şarkı', 'türkü', 'melodi', 'ritim', 'nota', 'gitar', 'piyano',
        'keman', 'flüt', 'davul', 'koro', 'orkestra', 'besteci',
        # Film ve medya
        'dizi', 'belgesel', 'komedi', 'dram', 'aksiyon', 'polisiye',
        # Hayvanlar (eksik)
        'kedi', 'köpek', 'at', 'inek', 'koyun', 'keçi', 'tavşan',
        'karınca', 'kelebek', 'örümcek', 'kurbağa', 'kertenkele', 'penguen',
        # Din ve toplum
        'cami', 'kilise', 'ibadet', 'dua', 'bayram', 'düğün', 'cenaze',
    ]

    # ═════════════════════════════════════════════════════════
    #  INITIALIZATION
    # ═════════════════════════════════════════════════════════

    def __init__(
        self,
        include_categories: Optional[List[str]] = None,
        custom_words: Optional[List[str]] = None,
    ):
        """
        Args:
            include_categories: Subset of categories to include.
                Default: all. Options: physics, software, philosophy,
                daily, adjectives, verbs, connectives, punctuation.
            custom_words: Additional words to merge in.
        """
        self._categories_map = {
            'punctuation':       self.PUNCTUATION,
            'physics':           self.PHYSICS_TERMS,
            'software':          self.SOFTWARE_TERMS,
            'philosophy':        self.PHILOSOPHY_TERMS,
            'daily':             self.DAILY_TERMS,
            'adjectives':        self.ADJECTIVES,
            'verbs':             self.VERBS,
            'connectives':       self.CONNECTIVES,

            'turkish_common':    self.TURKISH_COMMON,
            'natural_science':   self.NATURAL_SCIENCE,
            'social_science':    self.SOCIAL_SCIENCE,
            'extended_verbs':    self.EXTENDED_VERBS,
            'extended_adjectives': self.EXTENDED_ADJECTIVES,
            'a1_turkish':        self.A1_TURKISH,
        }


        if include_categories is None:
            include_categories = list(self._categories_map.keys())

        # Build vocabulary: special tokens FIRST (stable IDs)
        self.all_words: List[str] = list(self.SPECIAL_TOKENS)
        self.word_categories: Dict[str, str] = {
            tok: 'special' for tok in self.SPECIAL_TOKENS
        }
        self.category_ranges: Dict[str, tuple] = {
            'special': (0, len(self.SPECIAL_TOKENS))
        }

        # Add each category, track ranges for masking/analysis
        seen: Set[str] = set(self.all_words)
        for cat_name in include_categories:
            if cat_name not in self._categories_map:
                continue
            start = len(self.all_words)
            for word in self._categories_map[cat_name]:
                w = word.lower().strip()
                if w and w not in seen:
                    self.all_words.append(w)
                    self.word_categories[w] = cat_name
                    seen.add(w)
            end = len(self.all_words)
            self.category_ranges[cat_name] = (start, end)

        # Custom words
        if custom_words:
            start = len(self.all_words)
            for word in custom_words:
                w = word.lower().strip()
                if w and w not in seen:
                    self.all_words.append(w)
                    self.word_categories[w] = 'custom'
                    seen.add(w)
            end = len(self.all_words)
            if end > start:
                self.category_ranges['custom'] = (start, end)

        # Build reverse index
        self.word_to_id: Dict[str, int] = {
            w: i for i, w in enumerate(self.all_words)
        }

    # ═════════════════════════════════════════════════════════
    #  CORE API
    # ═════════════════════════════════════════════════════════

    def size(self) -> int:
        """
        CRITICAL: Use this for ALL matrix dimensions.

        Usage:
            self.mx2 = nn.Linear(HIDDEN_DIM, vocab.size())
            logits = torch.zeros(vocab.size())
        """
        return len(self.all_words)

    def __len__(self) -> int:
        return len(self.all_words)

    def id_to_word(self, idx: int) -> str:
        """Look up word by index. Returns <unk> for out-of-range."""
        if 0 <= idx < len(self.all_words):
            return self.all_words[idx]
        return '<unk>'

    def get_id(self, word: str, default: Optional[int] = None) -> int:
        """Look up word index, default to <unk> id if not found."""
        w = word.lower().strip()
        if w in self.word_to_id:
            return self.word_to_id[w]
        return default if default is not None else self.word_to_id['<unk>']

    def contains(self, word: str) -> bool:
        return word.lower().strip() in self.word_to_id

    def category_of(self, word: str) -> Optional[str]:
        return self.word_categories.get(word.lower().strip())

    def words_in_category(self, category: str) -> List[str]:
        if category not in self.category_ranges:
            return []
        start, end = self.category_ranges[category]
        return self.all_words[start:end]

    def category_mask(self, category: str,
                      device: Optional[str] = None) -> Optional[object]:
        """
        Return a boolean mask (torch tensor) selecting only words
        in the given category. Useful for constrained decoding.
        """
        if category not in self.category_ranges:
            return None
        try:
            import torch
            mask = torch.zeros(len(self.all_words), dtype=torch.bool)
            start, end = self.category_ranges[category]
            mask[start:end] = True
            if device:
                mask = mask.to(device)
            return mask
        except ImportError:
            start, end = self.category_ranges[category]
            return [start <= i < end for i in range(len(self.all_words))]

    def add_word(self, word: str, category: str = 'custom') -> int:
        """
        Dynamic vocabulary growth. Returns the new word's ID.

        After calling this, any nn.Linear using vocab.size() will need
        to be rebuilt. Example:

            new_id = vocab.add_word('zeta', 'physics')
            model.mx2 = nn.Linear(HIDDEN_DIM, vocab.size())
        """
        w = word.lower().strip()
        if w in self.word_to_id:
            return self.word_to_id[w]
        idx = len(self.all_words)
        self.all_words.append(w)
        self.word_to_id[w] = idx
        self.word_categories[w] = category
        return idx

    def stats(self) -> Dict:
        """Vocabulary breakdown by category."""
        counts = {}
        for cat, (start, end) in self.category_ranges.items():
            counts[cat] = end - start
        return {
            'total_size': len(self.all_words),
            'category_counts': counts,
            'special_tokens': self.SPECIAL_TOKENS,
        }

    # ═════════════════════════════════════════════════════════
    #  PERSISTENCE
    # ═════════════════════════════════════════════════════════

    def save(self, path: str = './mergen_vocab.json'):
        """Serialize vocabulary for persistence."""
        data = {
            'all_words': self.all_words,
            'word_categories': self.word_categories,
            'category_ranges': {k: list(v) for k, v in self.category_ranges.items()},
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    @classmethod
    def load(cls, path: str = './mergen_vocab.json') -> 'MergenVocab':
        """Load vocabulary from disk."""
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        vocab = cls(include_categories=[])  # empty init
        vocab.all_words = data['all_words']
        vocab.word_categories = data['word_categories']
        vocab.category_ranges = {
            k: tuple(v) for k, v in data['category_ranges'].items()
        }
        vocab.word_to_id = {w: i for i, w in enumerate(vocab.all_words)}
        return vocab

    def __repr__(self):
        stats = self.stats()
        parts = [f"MergenVocab(total={stats['total_size']}, "]
        for cat, n in stats['category_counts'].items():
            parts.append(f"  {cat}: {n}")
        return "\n".join(parts) + "\n)"


# ════════════════════════════════════════════════════════════════════
#  STANDALONE DEMO
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  MERGEN V3 — MergenVocab (Vocabulary Expansion)")
    print("=" * 65)

    vocab = MergenVocab()
    print(f"\n{vocab}\n")

    print(f"  vocab.size() = {vocab.size()}  ← USE THIS FOR nn.Linear\n")

    # Example neural matrix setup (documented, not run)
    print("  CORRECT USAGE IN MODEL:")
    print("    self.mx2 = nn.Linear(config.HIDDEN_DIM, vocab.size())")
    print("                                             ^^^^^^^^^^^^")
    print("    Never hardcode the output size!\n")

    # Category examples
    print("  Sample words from each category:")
    for cat in ['physics', 'software', 'philosophy', 'adjectives', 'verbs']:
        words = vocab.words_in_category(cat)[:5]
        print(f"    {cat:15s}: {words}")

    # Dynamic growth
    print(f"\n  Before add_word: vocab.size() = {vocab.size()}")
    new_id = vocab.add_word('kuantumetrik', 'physics')
    print(f"  After  add_word: vocab.size() = {vocab.size()} "
          f"(added 'kuantumetrik' at id={new_id})")

    # Save/load test
    vocab.save('/tmp/mergen_vocab_test.json')
    loaded = MergenVocab.load('/tmp/mergen_vocab_test.json')
    print(f"\n  Persisted & reloaded: {loaded.size()} words ✓")

    print("\n" + "=" * 65)
