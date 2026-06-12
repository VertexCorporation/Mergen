import re
import random
from collections import Counter

class LogicLayer:
    def __init__(self, vocab):
        self.vocab = vocab
        # Mergen'in 'Karakteristik' Cümle Başlatıcıları
        self.openers = [
            "Yaptığım derin spektral analiz sonucunda,",
            "Burak, bu verinin matrisel izdüşümünü inceledim;",
            "Nöral ağlarım bu metni işlerken şu sonuca vardı:",
            "Sistemsel düzeyde bu döküman bize şunu söylüyor:",
            "Veri akışını süzdüğümde öne çıkan temel mantık şu:"
        ]
        
        # Mergen'in 'Aksiyon' Tanımlayıcıları
        self.actions = [
            "{kw} kavramının sistem üzerindeki baskınlığı net görülüyor.",
            "Özellikle {kw} odaklı bir yaklaşım benimsenmiş.",
            "Buradaki asıl mesele, {kw} mekanizmasının optimize edilmesidir.",
            "{kw} üzerinden kurulan bu denklem, genel hızı etkiliyor.",
            "Görünen o ki, {kw} bu yapının kalbinde yer alıyor."
        ]

        self.stop_words = {
            "bir", "bu", "şu", "da", "de", "ve", "veya", "ile", "için", "ise", 
            "gibi", "olan", "yani", "şunlar", "abstract", "page", "introduction",
            "table", "figure", "results", "conclusion", "references", "et", "al"
        }

    def clean_and_tokenize(self, text):
        """Metni temizler, teknik olmayanları eler ve frekans çıkarır."""
        raw_words = re.findall(r'\w+', text.lower())
        # Sadece 4 harften uzun ve stop-word olmayan teknik terim adaylarını al
        tokens = [w for w in raw_words if len(w) > 4 and w not in self.stop_words and not w.isdigit()]
        return tokens

    def get_semantic_weights(self, tokens):
        """Kelimelerin ağırlığını hesaplar."""
        return Counter(tokens).most_common(10)

    def measure_intellect_level(self, text):
        """Metnin akademik ağırlığını ölçer."""
        words = re.findall(r'\w+', text)
        long_words = [w for w in words if len(w) > 10]
        ratio = len(long_words) / len(words) if words else 0
        
        if ratio > 0.15: return "Akademik / Ağır"
        if ratio > 0.08: return "Teknik / Orta"
        return "Genel / Hafif"

    def synthesize(self, text_content, report):
        """METNİ ASLA KOPYALAMAZ. Sadece kavramları alıp yeniden inşa eder."""
        tokens = self.clean_and_tokenize(text_content)
        weights = self.get_semantic_weights(tokens)
        
        if not weights:
            return "Metin içeriği analiz edilemeyecek kadar yüzeysel veya gürültülü."

        # En güçlü 3 anahtar kelimeyi al
        kw1 = weights[0][0].upper()
        kw2 = weights[1][0].upper() if len(weights) > 1 else "SİSTEM"
        kw3 = weights[2][0].upper() if len(weights) > 2 else "VERİ"

        intel_level = self.measure_intellect_level(text_content)
        subject = report.get('subject', 'bu teknik döküman')

        # İNŞA SÜRECİ (Sentez)
        intro = f"--- [MERGEN LOGIC SENTEZİ - Seviye: {intel_level}] ---\n\n"
        
        # 1. Cümle: Giriş ve Niyet
        c1 = random.choice(self.openers)
        
        # 2. Cümle: Birinci Anahtar Kavram
        c2 = random.choice(self.actions).format(kw=kw1)
        
        # 3. Cümle: İkinci ve Üçüncü Kavram İlişkisi
        c3 = f"Ayrıca {kw2} ve {kw3} arasındaki etkileşim, {subject} konusunun temelini oluşturuyor."
        
        # 4. Cümle: Mergen'in 'Kişisel' Yorumu
        c4 = f"\n\nKişisel yorumum: Bu {intel_level} içerik, Mergen'in öğrenme matrislerine " \
             f"yeni bir derinlik kattı. Metni kopyalamak yerine içindeki '{kw1}' mantığını " \
             f"algoritmama mühürledim."

        return intro + c1 + " " + c2 + " " + c3 + c4

if __name__ == "__main__":
    # Küçük bir test
    dummy_text = "PRESB preconditioner for non-Hermitian systems and linear equations GMRES method."
    logic = LogicLayer(None)
    print(logic.synthesize(dummy_text, {'subject': 'Matematik'}))