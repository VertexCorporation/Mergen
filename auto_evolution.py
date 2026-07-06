"""
Auto Evolution - Mergen'in sürekli gelişimi (VDS'de çalışacak)
API anahtarı sadece bellekte tutulur, dosyaya yazılmaz
"""

import sys
import time
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Optional

# Güvenlik Kilidi
EVOLUTION_ENABLED = False

if not EVOLUTION_ENABLED:
    print("[Evolution] [X] OTOMATIK GELISIM SISTEMI GUVENLIK NEDENIYLE KAPATILMISTIR (EVOLUTION_ENABLED = False).")
    print("[Evolution] Kodu aktif etmek icin auto_evolution.py dosyasindaki EVOLUTION_ENABLED degerini True yapin.")
    sys.exit(0)

# API anahtarını komut satırından al
if len(sys.argv) < 2:
    print("Kullanım: python auto_evolution.py <OPENROUTER_API_KEY>")
    print("Örnek: python auto_evolution.py sk-or-v1-dummy")
    sys.exit(1)

API_KEY = sys.argv[1]  # Bellekte tutulur, dosyaya YAZILMAZ

# Modüller
try:
    from openrouter_client import OpenRouterClient, FREE_MODELS
    from code_evolution import CodeEvolutionEngine
    import curriculum
except ImportError as e:
    print(f"Modül yükleme hatası: {e}")
    sys.exit(1)


class AutoEvolutionController:
    """
    Mergen'in otomatik olarak gelişimini kontrol eder.
    VDS'de sürekli çalışır, insan beyni gibi öğrenir.
    """
    
    def __init__(self, api_key: str, project_root: str = "."):
        self.api_key = api_key
        self.project_root = Path(project_root)
        
        # İstatistikler - ÖNCE BUNU TANIMLA
        self.is_running = False
        self.current_cycle = 0
        self.stats = {
            "start_time": datetime.now().isoformat(),
            "total_improvements": 0,
            "total_errors": 0,
            "models_used": set(),
            "files_modified": set(),
        }
        
        # AI istemcisi (API anahtarı sadece burada)
        self.ai_client = OpenRouterClient(api_key)
        
        # Kod geliştirme motoru
        self.evolver = CodeEvolutionEngine(project_root)
        
        # Müfredat
        self.curriculum = curriculum.CURRICULUM
        
        # Durum - hemen status.json oluştur
        self._write_status_json()
        
        print("=" * 70)
        print("🧠 MERGEN OTOMATİK GELİŞİM SİSTEMİ")
        print("=" * 70)
        print(f"API Anahtarı: {api_key[:10]}...{api_key[-5:]} (bellekte)")
        print(f"Mevcut Seviye: {self.evolver.current_grade}. Sınıf")
        print(f"Hedef: 12. Sınıf (İnsan beyni gibi)")
        print("=" * 70)
    
    def run_forever(self, cycles_per_grade: int = 5, sleep_seconds: int = 60):
        """
        Sürekli gelişim döngüsü.
        
        Args:
            cycles_per_grade: Her sınıf için kaç geliştirme döngüsü yapılacak
            sleep_seconds: Döngüler arası bekleme süresi
        """
        self.is_running = True
        
        try:
            while self.evolver.current_grade <= 12:
                grade = self.evolver.current_grade
                grade_name = self.evolver.learning_progress.get("current_grade_name", f"{grade}. Sınıf")
                
                print(f"\n{'='*70}")
                print(f"📚 {grade_name} - Gelişim Döngüsü")
                print(f"{'='*70}")
                
                # Bu sınıf için konuları al
                grade_data = self.curriculum.get(grade)
                if not grade_data:
                    print(f"⚠ {grade}. sınıf verisi bulunamadı, atlanıyor...")
                    if self.evolver.current_grade < 12:
                        self.evolver.advance_grade()
                    continue
                
                topics = []
                for subject, subject_topics in grade_data["subjects"].items():
                    for topic in subject_topics:
                        topics.append(f"{subject}: {topic}")
                
                print(f"📖 Öğrenilecek konular: {len(topics)}")
                
                grade_improvements = 0  # Bu sınıfta kaç geliştirme yapıldı
                
                # Her konu için geliştirme yap (en az 1 tane başarılı olana kadar)
                attempts = 0
                max_attempts = cycles_per_grade * 3  # Daha fazla deneme hakkı
                
                while grade_improvements < cycles_per_grade and attempts < max_attempts and self.is_running:
                    attempts += 1
                    self.current_cycle += 1
                    
                    # Rastgele bir konu seç
                    topic = random.choice(topics) if topics else f"Genel programlama {grade}. sınıf"
                    
                    print(f"\n🔄 Döngü {self.current_cycle} (Konu: {topic})")
                    
                    # Rastgele bir model seç
                    model = random.choice(FREE_MODELS)
                    print(f"🤖 Öğretmen Model: {model}")
                    
                    # Geliştirilecek dosyaları seç - en uygun dosyayı seç
                    py_files = self.evolver.get_all_python_files()
                    
                    if not py_files:
                        print("⚠ Geliştirilecek Python dosyası bulunamadı!")
                        # Yeni bir dosya oluştur
                        print("📝 Yeni bir dosya oluşturuluyor...")
                        new_file = self.project_root / f"grade_{grade}_lesson.py"
                        with open(new_file, 'w', encoding='utf-8') as f:
                            f.write(f"# {topic}\n# Mergen {grade}. sınıf öğrenme dosyası\n\n")
                        py_files = self.evolver.get_all_python_files()
                    
                    # İlk dosyayı seç (küçükten büyüğe sıralı)
                    target_file = py_files[0] if py_files else None
                    
                    if not target_file:
                        print("⚠ Dosya seçilemedi!")
                        continue
                    
                    print(f"📝 Geliştirilecek dosya: {target_file.name} ({target_file.stat().st_size} byte)")
                    
                    # Modelden iyileştirme iste
                    success, result = self.evolver.request_improvement(
                        ai_client=self.ai_client,
                        model=model,
                        file_path=target_file,
                        grade=grade,
                        topic=topic,
                    )
                    
                    if success:
                        # İyileştirmeyi uygula
                        if self.evolver.apply_improvement(
                            file_path=target_file,
                            new_code=result,
                            model=model,
                            topic=topic,
                        ):
                            self.stats["total_improvements"] += 1
                            self.stats["models_used"].add(model)
                            self.stats["files_modified"].add(str(target_file))
                            grade_improvements += 1
                            print(f"✅ Geliştirme başarılı! ({grade_improvements}/{cycles_per_grade})")
                            print(f"📏 Yeni kod uzunluğu: {len(result)} karakter")
                        else:
                            self.stats["total_errors"] += 1
                            print(f"❌ Kod uygulanamadı!")
                    else:
                        self.stats["total_errors"] += 1
                        print(f"❌ Geliştirme başarısız: {result}")
                    
                    # İstatistikleri göster
                    self._print_stats()
                    
                    # Bir sonraki döngüye kadar bekle
                    if self.is_running and grade_improvements < cycles_per_grade:
                        print(f"⏳ {sleep_seconds} saniye bekleniyor...")
                        time.sleep(sleep_seconds)
                
                # Sınıfı bitir, sonraki sınıfa geç (en az 1 geliştirme yapıldıysa)
                if grade_improvements > 0 and self.is_running and grade < 12:
                    print(f"\n🎓 {grade_name} tamamlandı! ({grade_improvements} geliştirme yapıldı)")
                    print(f"🎓 Bir üst sınıfa geçiliyor...")
                    self.evolver.advance_grade()
                elif grade_improvements == 0:
                    print(f"\n⚠ {grade_name} için hiç geliştirme yapılamadı! 30 saniye bekleniyor...")
                    time.sleep(30)
                elif grade >= 12:
                    print(f"\n🎓 Tüm sınıflar tamamlandı! Mergen artık 12. sınıf seviyesinde.")
            
            print("\n" + "=" * 70)
            print("🏆 GELİŞİM TAMAMLANDI!")
            print("=" * 70)
            
        except KeyboardInterrupt:
            print("\n\n⚠ Gelişim kullanıcı tarafından durduruldu.")
        except Exception as e:
            print(f"\n\n❌ Beklenmeyen hata: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()
    
    def _print_stats(self):
        """İstatistikleri yazdır ve status.json'a yaz."""
        print(f"\n📊 İSTATİSTİKLER:")
        print(f"  • Toplam geliştirme: {self.stats['total_improvements']}")
        print(f"  • Hata sayısı: {self.stats['total_errors']}")
        print(f"  • Kullanılan model sayısı: {len(self.stats['models_used'])}")
        print(f"  • Değiştirilen dosya sayısı: {len(self.stats['files_modified'])}")
        print(f"  • Mevcut sınıf: {self.evolver.current_grade}")
        
        # status.json'a yaz (monitor için)
        self._write_status_json()
    
    def _write_status_json(self):
        """İzleme için status.json dosyasını yazar."""
        status = {
            "current_grade": self.evolver.current_grade,
            "grade_name": f"{self.evolver.current_grade}. Sınıf",
            "stats": {
                "total_improvements": self.stats["total_improvements"],
                "total_errors": self.stats["total_errors"],
                "models_used": list(self.stats["models_used"]),
                "files_modified": list(self.stats["files_modified"]),
            },
            "evolution_log": self.evolver.evolution_log[-50:] if hasattr(self.evolver, 'evolution_log') else [],
            "last_update": datetime.now().isoformat(),
            "is_running": self.is_running,
        }
        
        try:
            with open(self.project_root / "status.json", 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠ status.json yazma hatası: {e}")
    
    def _cleanup(self):
        """Temizlik işlemleri."""
        print("\n🧹 Temizlik yapılıyor...")
        
        # API anahtarını bellekten sil
        self.api_key = None
        if hasattr(self, 'ai_client'):
            self.ai_client.clear_key()
        
        # İstatistikleri kaydet
        stats_path = self.project_root / "evolution_stats.json"
        save_stats = dict(self.stats)
        save_stats["models_used"] = list(save_stats["models_used"])
        save_stats["files_modified"] = list(save_stats["files_modified"])
        
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(save_stats, f, ensure_ascii=False, indent=2)
            print(f"✅ İstatistikler kaydedildi: {stats_path}")
        except Exception as e:
            print(f"⚠ İstatistik kaydetme hatası: {e}")
        
        print("✅ Temizlik tamamlandı. API anahtarı bellekten silindi.")
    
    def stop(self):
        """Gelişimi durdur."""
        self.is_running = False
        print("⚠ Durdurma isteği alındı...")


def main():
    """Ana fonksiyon."""
    import sys
    
    # Parametreleri kontrol et
    api_key = None
    telegram_token = None
    
    if len(sys.argv) < 2:
        print("Kullanım: python auto_evolution.py <OPENROUTER_API_KEY> [TELEGRAM_BOT_TOKEN]")
        print("Örnek: python auto_evolution.py sk-or-v1-xxx 123456789:ABCdefGHIjklMNOpqrsTUVwxyz1234567")
        sys.exit(1)
    
    api_key = sys.argv[1]
    if len(sys.argv) > 2:
        telegram_token = sys.argv[2]
    
    controller = AutoEvolutionController(api_key)
    
    # Telegram botunu başlat (opsiyonel)
    bot = None
    if telegram_token:
        try:
            from telegram_bot import TelegramBot
            bot = TelegramBot(telegram_token)
            bot.start()
            print("[Main] Telegram botu başlatıldı.")
        except Exception as e:
            print(f"[Main] Telegram botu başlatılamadı: {e}")
    
    # VDS'de sürekli çalışması için
    controller.run_forever(cycles_per_grade=5, sleep_seconds=60)
    
    # Temizlik
    if bot:
        bot.stop()


if __name__ == "__main__":
    main()