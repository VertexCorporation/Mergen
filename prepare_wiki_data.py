import os
import bz2
import re
import hashlib
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Optional
from pathlib import Path
import argparse
import mwparserfromhell
from tqdm import tqdm

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("prepare_wiki_preprocessing.log", encoding="utf-8")
    ]
)

class WikiConfig:
    MIN_WORDS: int = 7
    MAX_WORDS: int = 35
    MIN_ALPHA_RATIO: float = 0.75
    MAX_DIGITS: int = 15
    SENTENCES_PER_PART: int = 15000

    BAD_STARTERS: Set[str] = {
        "şu", "bunlar", "şunlar", "onlar", "ayrıca", "ancak", 
        "fakat", "ama", "çünkü", "zira", "nitekim", "bununla", "bundan",
        "ve", "veya", "oysa", "halbuki", "bununla birlikte", "bunun yanı sıra"
    }

    BAD_PATTERNS: List[str] = [
        "url", "erişim tarihi", "kaynakça", "dış bağlantılar", "nobelprize",
        "wikipedia", "britannica", "doi:", "isbn", "sayfa:", "(ingilizce)",
        "küçükresim", "thumb", "rowspan", "colspan", "|-", "||", "{{", "}}",
        "listesi", "doğumlar", "ölümler", "olaylar", "kategori:", "resim:", "dosya:"
    ]

class WikiTextProcessor:
    def __init__(self):
        self.seen_hashes: Set[str] = set()
        self.tr_abbreviations: List[str] = [
            "prof.", "dr.", "doç.", "av.", "hz.", "m.ö.", "m.s.", "v.b.", 
            "vb.", "bkz.", "yy.", "sf.", "bknz.", "mad.", "cad.", "sok."
        ]
        
    def clean_wiki_text(self, text: Optional[str]) -> str:
        if not text or not text.strip():
            return ""
        if text.startswith("#YÖNLENDİRME") or text.startswith("#REDIRECT"):
            return ""

        try:
            parsed = mwparserfromhell.parse(text)
            text = parsed.strip_code(normalize=True)
        except Exception:
            text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.S)
            text = re.sub(r"\[\[.*?\]\]", "", text)

        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.S)
        text = re.sub(r"\[\[(Dosya|File|Resim|Image):.*?\]\]", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def split_sentences_advanced(self, text: str) -> List[str]:
        protected_text = text
        for abbrev in self.tr_abbreviations:
            placeholder = abbrev.replace(".", "«DOT»")
            protected_text = re.sub(r'\b' + re.escape(abbrev), placeholder, protected_text, flags=re.IGNORECASE)
            
        protected_text = re.sub(r'\b(\d+)\.', r'\1«DOT»', protected_text)
        protected_text = re.sub(r'\b([A-ZÇĞİÖŞÜ])\.', r'\1«DOT»', protected_text)
        raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ])', protected_text)
        
        sentences = []
        for s in raw_sentences:
            s_restored = s.replace("«DOT»", ".")
            if s_restored.strip():
                sentences.append(s_restored.strip())
        return sentences

    def is_valid_experience(self, sentence: str) -> bool:
        sentence = sentence.strip()
        if not sentence.endswith("."):
            return False
        if sentence.startswith("(") or sentence.startswith("-") or sentence.startswith("*"):
            return False
        if not sentence[0].isupper():
            return False

        words = sentence.split()
        word_count = len(words)
        if not (WikiConfig.MIN_WORDS <= word_count <= WikiConfig.MAX_WORDS):
            return False

        first_word = words[0].lower().rstrip(",.:;")
        if first_word in WikiConfig.BAD_STARTERS:
            return False
        
        if word_count > 1:
            two_words = f"{words[0]} {words[1]}".lower()
            if any(two_words.startswith(starter) for starter in WikiConfig.BAD_STARTERS):
                return False

        lower_sentence = sentence.lower()
        if any(bad in lower_sentence for bad in WikiConfig.BAD_PATTERNS):
            return False
        if "|" in sentence:
            return False

        alpha_chars = len(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]", sentence))
        if len(sentence) > 0 and (alpha_chars / len(sentence)) < WikiConfig.MIN_ALPHA_RATIO:
            return False

        digit_chars = len(re.findall(r"\d", sentence))
        if digit_chars > WikiConfig.MAX_DIGITS:
            return False

        if sentence.count("(") != sentence.count(")") or sentence.count("[") != sentence.count("]"):
            return False

        h = hashlib.md5(sentence.encode("utf-8")).hexdigest()
        if h in self.seen_hashes:
            return False
        
        self.seen_hashes.add(h)
        return True

class WikiDatasetBuilder:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.processor = WikiTextProcessor()
        self.stats = {
            "total_pages_processed": 0,
            "non_main_ns_skipped": 0,
            "total_sentences_extracted": 0,
            "valid_experiences_saved": 0,
            "parts_written": 0
        }

    def build(self) -> None:
        logging.info(f"Wiki Veri Hazırlama Hattı Başlatılıyor. Girdi: {self.input_file}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        current_part_idx = 1
        current_sentences = []

        def write_part():
            nonlocal current_part_idx, current_sentences
            if not current_sentences:
                return
            part_name = f"part_{current_part_idx:04d}.txt"
            part_path = self.output_dir / part_name
            with open(part_path, "w", encoding="utf-8") as pf:
                for s in current_sentences:
                    pf.write(s + "\n")
            logging.info(f"[Writer] Yazıldı: {part_name} | Deneyim: {len(current_sentences):,}")
            self.stats["parts_written"] += 1
            current_part_idx += 1
            current_sentences = []

        try:
            with bz2.open(self.input_file, "rt", encoding="utf-8") as xml_file:
                context = ET.iterparse(xml_file, events=("start", "end"))
                event, root = next(context)
                
                pbar = tqdm(desc="Extracted Sentences")
                
                for event, elem in context:
                    if event == "end" and elem.tag.endswith("page"):
                        self.stats["total_pages_processed"] += 1
                        
                        ns_elem = elem.find(".//{*}ns")
                        if ns_elem is not None and ns_elem.text != "0":
                            self.stats["non_main_ns_skipped"] += 1
                            elem.clear()
                            root.clear()
                            continue

                        text_elem = elem.find(".//{*}text")
                        text = text_elem.text if text_elem is not None else ""
                        
                        if text:
                            cleaned_text = self.processor.clean_wiki_text(text)
                            sentences = self.processor.split_sentences_advanced(cleaned_text)
                            
                            for sentence in sentences:
                                self.stats["total_sentences_extracted"] += 1
                                if self.processor.is_valid_experience(sentence):
                                    current_sentences.append(sentence)
                                    self.stats["valid_experiences_saved"] += 1
                                    pbar.update(1)

                                    if len(current_sentences) >= WikiConfig.SENTENCES_PER_PART:
                                        write_part()

                        elem.clear()
                        root.clear()
                
                pbar.close()
                write_part()  # Write any remaining sentences
                
        except FileNotFoundError:
            logging.error(f"Hata: {self.input_file} dosyası bulunamadı. Lütfen dump dosyasını doğru konuma yerleştirin.")
            return
        except Exception as e:
            logging.error(f"Pipeline sırasında hata oluştu: {str(e)}")
            return

        self._print_report()

    def _print_report(self) -> None:
        logging.info("=" * 50)
        logging.info("WIKI VERİ HAZIRLAMA TAMAMLANDI RAPORU")
        logging.info("=" * 50)
        logging.info(f"İşlenen Toplam Wikipedia Sayfası : {self.stats['total_pages_processed']:,}")
        logging.info(f"Elenen Yan Veri (Namespace != 0)  : {self.stats['non_main_ns_skipped']:,}")
        logging.info(f"Analiz Edilen Ham Cümle Sayısı   : {self.stats['total_sentences_extracted']:,}")
        logging.info(f"Üretilen Kaliteli Deneyim (Exp)   : {self.stats['valid_experiences_saved']:,}")
        logging.info(f"Yazılan Toplam Part Sayısı       : {self.stats['parts_written']}")
        logging.info(f"Hedef Dizin                      : {self.output_dir}")
        logging.info("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Wiki BZ2 Dump Hazırlama ve Partlama Scripti")
    parser.add_argument("--input", type=str, default="wikitr.bz2", help="Girdi BZ2 wiki dump dosya yolu")
    parser.add_argument("--output_dir", type=str, default="data/full_circullum_training", help="Çıktı part klasörü")
    args = parser.parse_args()

    builder = WikiDatasetBuilder(input_file=args.input, output_dir=args.output_dir)
    builder.build()

if __name__ == "__main__":
    main()
