"""
Code Evolution Engine - Yapay zekalar Mergen'in kodunu geliştirir
API anahtarı sadece bellekte tutulur
"""

import json
import re
import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class CodeEvolutionEngine:
    """
    Mergen'in kodunu analiz eder, diğer AI modellerinden
    iyileştirme önerileri alır ve kodu günceller.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.evolution_log = []
        self.current_grade = 1  # 1. sınıftan başlar
        self.learning_progress = {
            "current_grade": 1,
            "completed_topics": [],
            "code_versions": [],
            "teacher_feedback": [],
        }
        self._load_progress()
    
    def _load_progress(self):
        """Öğrenme ilerlemesini yükle."""
        progress_path = self.project_root / "learning_progress.json"
        if progress_path.exists():
            try:
                with open(progress_path, 'r', encoding='utf-8') as f:
                    self.learning_progress = json.load(f)
                    self.current_grade = self.learning_progress.get("current_grade", 1)
                print(f"[Evolution] İlerleme yüklendi: {self.current_grade}. sınıf")
            except Exception as e:
                print(f"[Evolution] İlerleme yükleme hatası: {e}")
    
    def _save_progress(self):
        """Öğrenme ilerlemesini kaydet."""
        progress_path = self.project_root / "learning_progress.json"
        try:
            with open(progress_path, 'w', encoding='utf-8') as f:
                json.dump(self.learning_progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Evolution] Kaydetme hatası: {e}")
    
    def get_all_python_files(self) -> List[Path]:
        """Tüm Python dosyalarını listeler (sadece anlamlı olanları)."""
        py_files = []
        for py_file in self.project_root.rglob("*.py"):
            # __pycache__ dizinini atla
            if "__pycache__" in str(py_file):
                continue
            # __init__.py dosyalarını atla
            if py_file.name == "__init__.py":
                continue
            # Dosya var mı ve dosya mı kontrol et
            if not py_file.exists() or not py_file.is_file():
                continue
            # Dosya boyutu 0'dan büyük mü?
            try:
                if py_file.stat().st_size == 0:
                    continue
                # En az 100 byte olsun (çok küçük dosyaları atla)
                if py_file.stat().st_size < 100:
                    continue
            except:
                continue
            py_files.append(py_file)
        
        if not py_files:
            return []
        
        # Boyuta göre sırala (küçükten büyüğe)
        return sorted(py_files, key=lambda f: f.stat().st_size)
    
    def read_file_content(self, file_path: Path) -> Optional[str]:
        """Dosya içeriğini okur."""
        try:
            if not file_path.exists():
                print(f"[Evolution] Dosya mevcut değil: {file_path}")
                return None
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content.strip():
                print(f"[Evolution] Dosya boş: {file_path}")
                return None
            return content
        except Exception as e:
            print(f"[Evolution] Dosya okuma hatası {file_path}: {e}")
            return None
    
    def write_file_content(self, file_path: Path, content: str) -> bool:
        """Dosya içeriğini yazar."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"[Evolution] Dosya yazma hatası {file_path}: {e}")
            return False
    
    def analyze_code_quality(self, code: str) -> Dict:
        """Kod kalitesini analiz eder."""
        analysis = {
            "lines": len(code.splitlines()),
            "functions": len(re.findall(r'def\s+\w+', code)),
            "classes": len(re.findall(r'class\s+\w+', code)),
            "imports": len(re.findall(r'^(import|from)\s+', code, re.MULTILINE)),
            "comments": len(re.findall(r'#.*$|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', code, re.MULTILINE)),
            "issues": [],
        }
        
        # Basit kontroller
        if analysis["functions"] == 0 and analysis["classes"] == 0:
            analysis["issues"].append("Kodda fonksiyon veya sınıf yok")
        
        if "TODO" in code or "FIXME" in code:
            analysis["issues"].append("Tamamlanmamış kısımlar var")
        
        return analysis
    
    def prepare_improvement_prompt(
        self,
        file_path: str,
        current_code: str,
        grade: int,
        topic: str,
    ) -> str:
        """
        Kod iyileştirme isteği için çok detaylı prompt hazırlar.
        Öğretmen AI'dan kodu derinlemesine analiz etmesi, hataları çözmesi ve
        Mergen'in o konudaki yeteneğini geliştirmesi istenir.
        """
        # Kod uzunluğuna göre ne kadar detay istediğimizi ayarla
        code_len = len(current_code)
        if code_len < 500:
            detail_level = "ÇOK DETAYLI ve KAPSAMLI"
        elif code_len < 2000:
            detail_level = "DETAYLI"
        else:
            detail_level = "ODAKLANMIŞ"
        
        prompt = f"""Sen kıdemli bir Python geliştiricisi ve yapay zeka eğitmenisin.
Görevin: Mergen projesinin kodunu DERİNLEMESİNE analiz et, hataları bul ve düzelt,
ve ona {topic} konusunu öğret.

PROJE HAKKINDA:
- Mergen: İnsan beyni gibi çalışmayı hedefleyen dijital beyin projesi
- Mevcut Seviye: {grade}. sınıf (1'den 12'ye kadar)
- Hedef: İnsan beyni gibi düşünebilme, anlayabilme, özetleyebilme
- Öğrenilecek Konu: {topic}

MEVCUT KOD DURUMU:
- Dosya: {file_path}
- Kod Uzunluğu: {code_len} karakter
- Beklenen: Mergen {topic} konusunu anlayacak, işleyecek yetenek kazanmalı

GÖREV TANIMI:
1. KODU DERİNLEMESİNE İNCELE:
   - Sentaks hatalarını bul ve düzelt
   - Mantık hatalarını tespit et
   - Eksik işlevselliği belirle
   - {topic} konusunda ne eksiği var analiz et

2. HATALARI ÇÖZ:
   - Tüm hataları düzelt
   - Eksikleri tamamla
   - Kodu optimize et

3. MERGEN'İ ÖĞRET:
   - {topic} konusunu işleyen YENİ FONKSİYONLAR ve SINIFLAR EKLE
   - Mergen bu konuyu "anlamalı", "okumalı", "özetlemeli"
   - Örnek: Eğer konu "Türkçe: Harfler" ise, harf tanıma, sesli okuma fonksiyonları ekle
   - Örnek: Eğer konu "Matematik: Toplama" ise, toplama işlemi yapan fonksiyonlar ekle

4. KODU DEĞİŞTİR:
   - Sadece açıklama yapma, KODU YAZ
   - Mevcut işlevselliği koru, üzerine ekleme yap
   - {detail_level} kod yaz
   - Türkçe yorum satırları ekle
   - Değişiklik yaptığın yerlere "# GELİŞTİRME: {topic}" yorumu ekle

KURALLAR:
1. Çıktın SADECE kod bloğu içermeli (```python ile başla, ``` ile bitir)
2. Açıklama yapma, sadece kodu ver
3. Kod çalışır durumda olmalı, sentaks hatası olmamalı
4. {grade}. sınıf seviyesine uygun olsun
5. Mergen'i bir üst seviyeye taşıyacak kodu yaz

MEVCUT KOD:
```python
{current_code}
```

Şimdi yukarıdaki talimatlara uyarak, {topic} konusunu Mergen'e öğretecek,
hataları çözecek ve kodu DEĞİŞTİRECEK fonksiyonel kodu yaz:"""
        
        return prompt
    
    def extract_code_from_response(self, response: str) -> Optional[str]:
        """AI yanıtından kodu çıkarır."""
        # ```python ... ``` bloğunu ara
        pattern = r'```python\s*(.*?)\s*```'
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Sadece ``` ... ``` bloğunu ara
        pattern = r'```\s*(.*?)\s*```'
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Kod bloğu bulunamadı, tüm yanıtı döndür
        return response.strip() if response else None
    
    def request_improvement(
        self,
        ai_client,
        model: str,
        file_path: Path,
        grade: int,
        topic: str,
    ) -> Tuple[bool, str]:
        """
        Belirli bir modelden kod iyileştirmesi ister.
        Derin öğretim yaparak kodu gerçekten değiştirmesini sağlar.
        
        Returns:
            (başarı, yeni_kod)
        """
        current_code = self.read_file_content(file_path)
        if not current_code:
            # Dosya boşsa, konuya uygun basit kod iste
            current_code = f"# {file_path.name}\n# {topic} için kod\n# Lütfen bu dosyayı geliştirin\n"
        
        prompt = self.prepare_improvement_prompt(
            str(file_path.relative_to(self.project_root)),
            current_code,
            grade,
            topic,
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        print(f"[Evolution] {model} modelinden iyileştirme isteniyor: {file_path.name} (kod uzunluğu: {len(current_code)} karakter)")
        
        improved_code = ai_client.chat(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=4000,
        )
        
        if not improved_code:
            return False, "Modelden yanıt alınamadı"
        
        # Kodu yanıttan çıkar
        new_code = self.extract_code_from_response(improved_code)
        
        if not new_code:
            # Yanıt kod bloğu içermiyorsa, tüm yanıtı kod olarak dene
            if "def " in improved_code or "class " in improved_code:
                new_code = improved_code
            else:
                return False, "Yanıttan kod çıkarılamadı"
        
        # Kodda anlamlı değişiklik var mı kontrol et
        if len(new_code) < 50:
            return False, "Yeni kod çok kısa"
        
        # Kod sentaks kontrolü
        try:
            ast.parse(new_code)
        except SyntaxError as e:
            return False, f"Kod sentaks hatası: {e}"
        
        return True, new_code
    
    def apply_improvement(
        self,
        file_path: Path,
        new_code: str,
        model: str,
        topic: str,
    ) -> bool:
        """İyileştirilmiş kodu uygular."""
        # Yedek al
        backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
        current_code = self.read_file_content(file_path)
        
        if current_code:
            self.write_file_content(backup_path, current_code)
        
        # Yeni kodu yaz
        success = self.write_file_content(file_path, new_code)
        
        if success:
            # Logla
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "file": str(file_path.relative_to(self.project_root)),
                "model": model,
                "topic": topic,
                "backup": str(backup_path.relative_to(self.project_root)),
            }
            self.evolution_log.append(log_entry)
            self._save_progress()
            
            print(f"[Evolution] ✅ {file_path.name} güncellendi ({model})")
            return True
        else:
            print(f"[Evolution] ❌ {file_path.name} güncellenemedi")
            return False
    
    def advance_grade(self):
        """Sonraki sınıfa geç."""
        if self.current_grade < 12:
            self.current_grade += 1
            self.learning_progress["current_grade"] = self.current_grade
            self._save_progress()
            print(f"[Evolution] 🎓 {self.current_grade}. sınıfa geçildi!")
            return True
        else:
            print("[Evolution] ✨ Tüm sınıflar tamamlandı!")
            return False
    
    def get_current_topic(self, curriculum_module) -> str:
        """Müfredattan mevcut konuyu alır."""
        grade_data = curriculum_module.get_grade_topics(self.current_grade)
        if not grade_data:
            return "Genel programlama"
        
        # İlk konuyu al (ileride daha akıllı seçim yapılabilir)
        for subject, topics in grade_data["subjects"].items():
            if topics:
                return f"{subject}: {topics[0]}"
        
        return "Genel programlama"
