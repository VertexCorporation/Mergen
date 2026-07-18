import bz2
import re
import hashlib
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Optional
import mwparserfromhell
from tqdm import tqdm

# ==========================================
# LOGGING VE CONFIGURATION SYSTEM
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mergen_preprocessing.log", encoding="utf-8")
    ]
)

class Config:
    INPUT_FILE: str = "trwiki-latest-pages-articles.xml.bz2"
    OUTPUT_FILE: str = "mergen_memory_test_10000.txt"
    MAX_EXPERIENCES: int = 50000
    
    # Kelime Sınırları (Semantik yoğunluk için ideal aralık)
    MIN_WORDS: int = 7
    MAX_WORDS: int = 18
    
    # Karakter Filtreleri
    MIN_ALPHA_RATIO: float = 0.75  # Cümlenin %75'i harflerden oluşmalı (Tablo/Matematik temizliği)
    MAX_DIGITS: int = 5            # Koordinat ve yoğun sayısal çöpleri engellemek için
    
    # Bağlamsal Bağımsızlık Filtreleri (Cümle başı yasaklı kelimeler)
    BAD_STARTERS: Set[str] = {
        "bu", "o", "şu", "bunlar", "şunlar", "onlar", "ayrıca", "ancak", 
        "fakat", "ama", "çünkü", "zira", "nitekim", "bununla", "bundan",
        "ve", "veya", "oysa", "halbuki", "bununla birlikte", "bunun yanı sıra"
    }
    
    # Wikipedia Genel Çöp Kelime Listesi
    BAD_PATTERNS: List[str] = [
        "url", "erişim tarihi", "kaynakça", "dış bağlantılar", "nobelprize",
        "wikipedia", "britannica", "doi:", "isbn", "sayfa:", "(ingilizce)",
        "küçükresim", "thumb", "rowspan", "colspan", "|-", "||", "{{", "}}",
        "listesi", "doğumlar", "ölümler", "olaylar", "kategori:", "resim:", "dosya:"
    ]

# ==========================================
# METİN TEMİZLEME VE NLP ENGINE
# ==========================================
class MergenTextProcessor:
    def __init__(self):
        self.seen_hashes: Set[str] = set()
        # Yaygın Türkçe kısaltmalar (Cümle bölünmesini engellemek için)
        self.tr_abbreviations: List[str] = [
            "prof.", "dr.", "doç.", "av.", "hz.", "m.ö.", "m.s.", "v.b.", 
            "vb.", "bkz.", "yy.", "sf.", "bknz.", "mad.", "cad.", "sok."
        ]
        
    def clean_wiki_text(self, text: Optional[str]) -> str:
        if not text or not text.strip():
            return ""
        
        # Hızlı ön filtreleme: Eğer sayfa bir yönlendirme veya boş şablonsa mwparser'a sokma
        if text.startswith("#YÖNLENDİRME") or text.startswith("#REDIRECT"):
            return ""

        try:
            parsed = mwparserfromhell.parse(text)
            text = parsed.strip_code(normalize=True)
        except Exception:
            # mwparserfromhell'in çökme ihtimaline karşı fallback regex temizliği
            text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.S)
            text = re.sub(r"\[\[.*?\]\]", "", text)

        # Regex Temizlik Hattı (Performans için optimize edildi)
        text = re.sub(r"<.*?>", "", text)  # HTML etiketleri
        text = re.sub(r"\[\d+\]", "", text)  # Kaynak numaraları [1], [2]
        text = re.sub(r"https?://\S+", "", text)  # URL'ler
        text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.S)  # Wiki tabloları
        text = re.sub(r"\[\[(Dosya|File|Resim|Image):.*?\]\]", "", text, flags=re.I)
        
        # Satır içi çoklu boşlukları ve tabları tek boşluğa indirge
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def split_sentences_advanced(self, text: str) -> List[str]:
        """
        Kısaltma ve sayı korumalı gelişmiş Türkçe cümle bölücü (Sentence Tokenizer).
        Nokta içeren kısaltmaları geçici olarak maskeler.
        """
        protected_text = text
        
        # 1. Kısaltmaları koruma altına al
        for abbrev in self.tr_abbreviations:
            placeholder = abbrev.replace(".", "«DOT»")
            protected_text = re.sub(r'\b' + re.escape(abbrev), placeholder, protected_text, flags=re.IGNORECASE)
            
        # 2. "20. yüzyıl" veya "1. Ordu" gibi sıra sayılarını koruma altına al
        protected_text = re.sub(r'\b(\d+)\.', r'\1«DOT»', protected_text)
        
        # 3. Baş harf noktalarını koru (Örn: Mustafa K. Atatürk -> K«DOT»)
        protected_text = re.sub(r'\b([A-ZÇĞİÖŞÜ])\.', r'\1«DOT»', protected_text)

        # 4. Gerçek cümle sınırlarından böl (Büyük harfle başlayan kelime takibi şartı)
        raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ])', protected_text)
        
        sentences = []
        for s in raw_sentences:
            # Maskeleri geri çöz
            s_restored = s.replace("«DOT»", ".")
            if s_restored.strip():
                sentences.append(s_restored.strip())
                
        return sentences

    def is_valid_experience(self, sentence: str) -> bool:
        sentence = sentence.strip()

        # Temel Yapı Kontrolleri
        if not sentence.endswith("."):
            return False
        if sentence.startswith("(") or sentence.startswith("-") or sentence.startswith("*"):
            return False
        
        # İlk karakter kontrolü (Büyük harfle başlamalı)
        if not sentence[0].isupper():
            return False

        words = sentence.split()
        word_count = len(words)
        
        # Uzunluk Filtresi
        if not (Config.MIN_WORDS <= word_count <= Config.MAX_WORDS):
            return False

        # Bağlamsal Bağımsızlık Testi (Anaphoric check)
        # "Bu", "O dönemde" gibi ifadelerle başlayan cümleler dış dünyadan izole bir hafıza olamaz.
        first_word = words[0].lower().rstrip(",.:;")
        if first_word in Config.BAD_STARTERS:
            return False
        
        # İki kelimeli başlangıç kontrolü (Örn: "Bu durum", "O yüzden")
        if word_count > 1:
            two_words = f"{words[0]} {words[1]}".lower()
            if any(two_words.startswith(starter) for starter in Config.BAD_STARTERS):
                return False

        # Çöp İfade ve Karakter Taraması
        lower_sentence = sentence.lower()
        if any(bad in lower_sentence for bad in Config.BAD_PATTERNS):
            return False
        if "|" in sentence or "=" in sentence or "*" in sentence:
            return False

        # Semantik Yoğunluk ve Karakter Dağılım Testi
        # Yoğun formül, koordinat veya bozuk UTF karakterlerini eler.
        alpha_chars = len(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]", sentence))
        if len(sentence) > 0 and (alpha_chars / len(sentence)) < Config.MIN_ALPHA_RATIO:
            return False

        # Aşırı Sayısal Bilgi Filtresi (Kronoloji ve istatistik listelerini eler)
        digit_chars = len(re.findall(r"\d", sentence))
        if digit_chars > Config.MAX_DIGITS:
            return False

        # Parantez Bütünlüğü Kontrolü
        if sentence.count("(") != sentence.count(")") or sentence.count("[") != sentence.count("]"):
            return False

        # Tekilleştirme (De-duplication via MD5)
        h = hashlib.md5(sentence.encode("utf-8")).hexdigest()
        if h in self.seen_hashes:
            return False
        
        self.seen_hashes.add(h)
        return True

# ==========================================
# PIPELINE VE DATASET BUILDER Engine
# ==========================================
class MergenDatasetBuilder:
    def __init__(self):
        self.processor = MergenTextProcessor()
        self.stats = {
            "total_pages_processed": 0,
            "non_main_ns_skipped": 0,
            "total_sentences_extracted": 0,
            "valid_experiences_saved": 0
        }

    def build(self) -> None:
        logging.info("Mergen Hafıza Hattı Başlatılıyor...")
        
        try:
            with bz2.open(Config.INPUT_FILE, "rt", encoding="utf-8") as xml_file, \
                 open(Config.OUTPUT_FILE, "w", encoding="utf-8") as output_file:

                # XML iterparse'ı akış (stream) modunda başlatıyoruz
                context = ET.iterparse(xml_file, events=("start", "end"))
                
                # O(1) RAM kullanımı için kök düğümü (root) yakalıyoruz
                event, root = next(context)
                
                pbar = tqdm(desc="Processed Experiences", total=Config.MAX_EXPERIENCES)
                
                for event, elem in context:
                    if self.stats["valid_experiences_saved"] >= Config.MAX_EXPERIENCES:
                        break

                    # Sadece sayfa elementi bittiğinde işleme al
                    if event == "end" and elem.tag.endswith("page"):
                        self.stats["total_pages_processed"] += 1
                        
                        # 1. Namespace Kontrolü (Sadece Ana Maddeler -> ns = 0)
                        # Kategori, Şablon ve Kullanıcı sayfalarını anında eler.
                        ns_elem = elem.find(".//{*}ns")
                        if ns_elem is not None and ns_elem.text != "0":
                            self.stats["non_main_ns_skipped"] += 1
                            elem.clear()
                            root.clear() # RAM birikmesini önleyen kritik komut
                            continue

                        # 2. Metin İçeriğini Çek
                        text_elem = elem.find(".//{*}text")
                        text = text_elem.text if text_elem is not None else ""
                        
                        if text:
                            # Wiki işaret dillerini temizle
                            cleaned_text = self.processor.clean_wiki_text(text)
                            # Akıllı cümle bölme uygulayarak parçala
                            sentences = self.processor.split_sentences_advanced(cleaned_text)
                            
                            for sentence in sentences:
                                if self.stats["valid_experiences_saved"] >= Config.MAX_EXPERIENCES:
                                    break
                                
                                self.stats["total_sentences_extracted"] += 1
                                
                                # Mergen kriterlerine uygunluk testi
                                if self.processor.is_valid_experience(sentence):
                                    output_file.write(sentence + "\n")
                                    self.stats["valid_experiences_saved"] += 1
                                    pbar.update(1)

                        # RAM Sızıntısını Önleme: İşlenen elementi bellekten temizle
                        elem.clear()
                        root.clear()

                pbar.close()
                
        except FileNotFoundError:
            logging.error(f"Hata: {Config.INPUT_FILE} dosyası bulunamadı. Lütfen dump dosyasını dizine ekleyin.")
            return
        except Exception as e:
            logging.error(f"Pipeline sırasında beklenmeyen bir hata oluştu: {str(e)}")
            return

        self._print_report()

    def _print_report(self) -> None:
        """Süreç sonu mühendislik raporu"""
        logging.info("=" * 40)
        logging.info("MERGEN PIPELINE TAMAMLANDI RAPORU")
        logging.info("=" * 40)
        logging.info(f"İşlenen Toplam Wikipedia Sayfası : {self.stats['total_pages_processed']}")
        logging.info(f"Elenen Yan Veri (Namespace != 0)  : {self.stats['non_main_ns_skipped']}")
        logging.info(f"Analiz Edilen Ham Cümle Sayısı   : {self.stats['total_sentences_extracted']}")
        logging.info(f"Üretilen Kaliteli Deneyim (Exp)   : {self.stats['valid_experiences_saved']}")
        logging.info(f"Çıktı Dosyası                    : {Config.OUTPUT_FILE}")
        logging.info("=" * 40)

if __name__ == "__main__":
    builder = MergenDatasetBuilder()
    builder.build()