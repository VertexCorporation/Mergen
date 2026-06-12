# -*- coding: utf-8 -*-
"""
Mergen Türkçe Veri Yükleyici

5 GitHub reposundan Türkçe veri çeker, işler ve RAGEngine'e indeksler:
  - Türkçe Sözlük       (mertkahyaoglu/turkish-dictionary-json)
  - TDK Sözlük Verisi   (vigo/tdk-sozluk-verisi)
  - Türkçe NLP Verisi   (vngrs/turkish-nlp-data)
  - Türkçe Wikipedia    (mizmirli/turkish-wikipedia-dump)
  - Açık Ders Akademik  (acikders/acikders)
"""

import re
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class TurkishDataLoader:
    """GitHub repolarından Türkçe veri indirir ve RAGEngine'e indeksler."""

    SOURCES: Dict[str, Dict] = {
        "dictionary": {
            "repo":  "mertkahyaoglu/turkish-dictionary-json",
            "label": "Türkçe Sözlük",
            "limit": 8000,
        },
        "tdk": {
            "repo":  "vigo/tdk-sozluk-verisi",
            "label": "TDK Sözlük Verisi",
            "limit": 6000,
        },
        "nlp_data": {
            "repo":  "vngrs/turkish-nlp-data",
            "label": "Türkçe NLP Verisi",
            "limit": 4000,
        },
        "wikipedia": {
            "repo":  "mizmirli/turkish-wikipedia-dump",
            "label": "Türkçe Wikipedia",
            "limit": 3000,
        },
        "academic": {
            "repo":  "acikders/acikders",
            "label": "Açık Ders Akademik",
            "limit": 3000,
        },
    }

    API_BASE       = "https://api.github.com/repos"
    MAX_FILE_BYTES = 12 * 1024 * 1024   # 12 MB
    HEADERS        = {
        "Accept":     "application/vnd.github.v3+json",
        "User-Agent": "Mergen-DataLoader/1.0",
    }

    def __init__(
        self,
        rag_engine,
        hebbian_bridge = None,
        cache_dir: str = "./mergen_data_cache",
        verbose:   bool = True,
    ):
        self.rag            = rag_engine
        self.hebbian_bridge = hebbian_bridge   # HebbianRAGBridge (opsiyonel)
        self.cache_dir      = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.verbose        = verbose

    # ──────────────────────────────────────────────────────────
    #  ANA GİRİŞ NOKTASI
    # ──────────────────────────────────────────────────────────

    def load_all(self, sources: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Tüm kaynakları (veya seçilen alt kümeyi) indir ve indeksle.
        Zaten indekslenmiş kaynaklar atlanır.
        """
        results: Dict[str, int] = {}
        targets = sources or list(self.SOURCES.keys())

        for key in targets:
            if key not in self.SOURCES:
                continue
            info = self.SOURCES[key]

            if self.rag.is_source_indexed(key):
                self._log(f"✓ {info['label']} zaten yüklü — atlandı.")
                continue

            self._log(f"⟳ {info['label']} yükleniyor...")
            try:
                count = self._dispatch(key, info)
                results[key] = count
                self._log(f"✓ {info['label']}: {count} kayıt indekslendi.")
            except Exception as e:
                self._log(f"✗ {info['label']} hatası: {e}")
                results[key] = 0

            time.sleep(1.2)   # GitHub API hız sınırı

        total = sum(results.values())
        if results:
            self._log(f"Toplam {total} yeni kayıt indekslendi.")
        return results

    def status(self) -> Dict[str, Any]:
        """Her kaynak için indeksleme durumunu döner."""
        out: Dict[str, Any] = {}
        for key, info in self.SOURCES.items():
            out[key] = {
                "label":   info["label"],
                "indexed": self.rag.is_source_indexed(key),
            }
        out["toplam_kayit"] = self.rag.count()
        return out

    # ──────────────────────────────────────────────────────────
    #  YÖNLENDİRİCİ
    # ──────────────────────────────────────────────────────────

    def _dispatch(self, key: str, info: Dict) -> int:
        method = getattr(self, f"_load_{key}", self._load_generic)
        count, texts = method(info["repo"], key, info["limit"])

        # Hebbian-RAG köprüsü: indekslenen metinler sinaptik ağırlıkları besler
        if self.hebbian_bridge is not None and texts:
            self.hebbian_bridge.update_from_batch(texts, source=key, reward=0.75)

        return count

    # ──────────────────────────────────────────────────────────
    #  GITHUB API YARDIMCILARI
    # ──────────────────────────────────────────────────────────

    def _list_repo(self, repo: str, path: str = "") -> List[Dict]:
        import requests
        url = f"{self.API_BASE}/{repo}/contents/{path}"
        try:
            r = requests.get(url, headers=self.HEADERS, timeout=25)
            if r.status_code == 200:
                return r.json() if isinstance(r.json(), list) else []
            if r.status_code == 403:
                self._log("GitHub API hız sınırı — 60s bekleniyor...")
                time.sleep(60)
        except Exception:
            pass
        return []

    def _download_text(self, url: str) -> Optional[str]:
        import requests
        try:
            r = requests.get(url, headers=self.HEADERS, timeout=60)
            if r.status_code == 200:
                return r.content.decode("utf-8", errors="replace")
        except Exception:
            pass
        return None

    def _cache_write(self, key: str, filename: str, content: str):
        try:
            (self.cache_dir / f"{key}_{filename}").write_text(
                content, encoding="utf-8"
            )
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────
    #  METİN İŞLEME
    # ──────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, size: int = 450) -> List[str]:
        """Metni cümle sınırlarına göre örtüşen parçalara böl."""
        parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for part in parts:
            part = part.strip()
            if len(part) < 10:
                continue
            if current_len + len(part) > size and current:
                chunk = " ".join(current).strip()
                if len(chunk) > 20:
                    chunks.append(chunk)
                current     = current[-1:]
                current_len = len(current[0]) if current else 0
            current.append(part)
            current_len += len(part)

        if current:
            chunk = " ".join(current).strip()
            if len(chunk) > 20:
                chunks.append(chunk)

        return chunks

    def _strip_markdown(self, text: str) -> str:
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
        text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        return text.strip()

    def _dict_to_sentence(self, d: dict) -> Optional[str]:
        """Sözlük girişinden okunabilir cümle oluştur."""
        word = (
            d.get("madde") or d.get("word") or d.get("title") or
            d.get("kelime") or d.get("name") or ""
        )
        meaning = (
            d.get("anlam") or d.get("meaning") or d.get("tanim") or
            d.get("description") or d.get("aciklama") or ""
        )
        if isinstance(meaning, list):
            meaning = ". ".join(str(m) for m in meaning if m)
        meaning = str(meaning).strip()
        word    = str(word).strip()

        if word and meaning:
            return f"{word}: {meaning[:450]}"
        if meaning and len(meaning) > 10:
            return meaning[:450]
        return None

    def _extract_from_json(
        self,
        data:       Any,
        source_key: str,
        texts:      List[str],
        metas:      List[Dict],
        limit:      int = 5000,
    ):
        entries = (
            data if isinstance(data, list) else
            list(data.values()) if isinstance(data, dict) else
            []
        )
        for entry in entries[:limit]:
            if len(texts) >= limit:
                break
            if isinstance(entry, str) and len(entry) > 10:
                texts.append(entry[:500])
                metas.append({"source": source_key, "type": "json"})
            elif isinstance(entry, dict):
                s = self._dict_to_sentence(entry)
                if s:
                    texts.append(s)
                    metas.append({"source": source_key, "type": "sozluk"})

    # ──────────────────────────────────────────────────────────
    #  KAYNAK YÜKLEYİCİLER
    # ──────────────────────────────────────────────────────────

    def _load_dictionary(self, repo: str, source_key: str, limit: int):
        files  = self._list_repo(repo)
        texts: List[str] = []
        metas: List[Dict] = []

        for f in files:
            if f.get("type") != "file":
                continue
            if not f["name"].lower().endswith(".json"):
                continue
            if f.get("size", 0) > self.MAX_FILE_BYTES:
                continue

            content = self._download_text(f["download_url"])
            if not content:
                continue
            self._cache_write(source_key, f["name"], content)

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                continue

            self._extract_from_json(data, source_key, texts, metas, limit)
            if len(texts) >= limit:
                break

        t = texts[:limit]
        return self.rag.index_texts(t, source_key, metas[:limit]), t

    def _load_tdk(self, repo: str, source_key: str, limit: int):
        all_files = self._list_repo(repo)
        # Bir seviye derine in
        for item in list(all_files):
            if item.get("type") == "dir":
                sub = self._list_repo(repo, item["path"])
                all_files.extend(sub)

        texts: List[str] = []
        metas: List[Dict] = []

        for f in all_files:
            if f.get("type") != "file":
                continue
            if f.get("size", 0) > self.MAX_FILE_BYTES:
                continue
            name = f["name"].lower()
            if not any(name.endswith(ext) for ext in [".json", ".txt", ".csv"]):
                continue

            content = self._download_text(f["download_url"])
            if not content:
                continue
            self._cache_write(source_key, f["name"], content)

            if name.endswith(".json"):
                try:
                    data = json.loads(content)
                    self._extract_from_json(data, source_key, texts, metas, limit)
                except json.JSONDecodeError:
                    for line in content.splitlines():
                        if len(line.strip()) > 15 and len(texts) < limit:
                            texts.append(line.strip()[:500])
                            metas.append({"source": source_key, "type": "tdk"})
            else:
                for chunk in self._chunk_text(content):
                    if len(texts) >= limit:
                        break
                    texts.append(chunk)
                    metas.append({"source": source_key, "type": "tdk"})

            if len(texts) >= limit:
                break

        t = texts[:limit]
        return self.rag.index_texts(t, source_key, metas[:limit]), t

    def _load_nlp_data(self, repo: str, source_key: str, limit: int):
        all_files = self._list_repo(repo)
        for item in list(all_files):
            if item.get("type") == "dir":
                sub = self._list_repo(repo, item["path"])
                all_files.extend(sub[:15])

        texts: List[str] = []
        metas: List[Dict] = []

        for f in all_files:
            if f.get("type") != "file":
                continue
            if f.get("size", 0) > self.MAX_FILE_BYTES:
                continue
            name = f["name"].lower()
            if not any(name.endswith(ext) for ext in [".txt", ".json", ".csv", ".tsv"]):
                continue

            content = self._download_text(f["download_url"])
            if not content:
                continue

            if name.endswith(".json"):
                try:
                    data = json.loads(content)
                    self._extract_from_json(data, source_key, texts, metas, limit)
                except json.JSONDecodeError:
                    pass
            else:
                for chunk in self._chunk_text(content):
                    if len(texts) >= limit:
                        break
                    texts.append(chunk)
                    metas.append({"source": source_key, "type": "nlp"})

            if len(texts) >= limit:
                break

        t = texts[:limit]
        return self.rag.index_texts(t, source_key, metas[:limit]), t

    def _load_wikipedia(self, repo: str, source_key: str, limit: int):
        files = self._list_repo(repo)
        texts: List[str] = []
        metas: List[Dict] = []

        for f in files:
            if f.get("type") != "file":
                continue
            size = f.get("size", 0)
            name = f["name"].lower()

            if size > self.MAX_FILE_BYTES:
                self._log(
                    f"  Wikipedia '{f['name']}' çok büyük "
                    f"({size // 1024} KB) — atlandı."
                )
                continue
            if not any(name.endswith(ext) for ext in [".txt", ".json", ".jsonl"]):
                continue

            content = self._download_text(f["download_url"])
            if not content:
                continue

            if name.endswith(".jsonl") or (name.endswith(".json") and "\n{" in content):
                # JSONL satır satır
                for line in content.splitlines()[:1000]:
                    if len(texts) >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj  = json.loads(line)
                        text = (
                            obj.get("text") or obj.get("content") or
                            obj.get("abstract") or obj.get("metin") or ""
                        )
                        if text and len(text) > 30:
                            for chunk in self._chunk_text(text):
                                if len(texts) >= limit:
                                    break
                                texts.append(chunk)
                                metas.append({"source": source_key, "type": "wikipedia"})
                    except json.JSONDecodeError:
                        if len(line) > 30:
                            texts.append(line[:500])
                            metas.append({"source": source_key, "type": "wikipedia"})
            elif name.endswith(".json"):
                try:
                    data = json.loads(content)
                    self._extract_from_json(data, source_key, texts, metas, limit)
                except json.JSONDecodeError:
                    for chunk in self._chunk_text(content):
                        if len(texts) >= limit:
                            break
                        texts.append(chunk)
                        metas.append({"source": source_key, "type": "wikipedia"})
            else:
                for chunk in self._chunk_text(content):
                    if len(texts) >= limit:
                        break
                    texts.append(chunk)
                    metas.append({"source": source_key, "type": "wikipedia"})

            if len(texts) >= limit:
                break

        t = texts[:limit]
        return self.rag.index_texts(t, source_key, metas[:limit]), t

    def _load_academic(self, repo: str, source_key: str, limit: int):
        all_files = self._list_repo(repo)
        for item in list(all_files):
            if item.get("type") == "dir":
                sub = self._list_repo(repo, item["path"])
                for sf in sub[:8]:
                    if sf.get("type") in ("file", "dir"):
                        all_files.append(sf)

        texts: List[str] = []
        metas: List[Dict] = []

        for f in all_files:
            if f.get("type") != "file":
                continue
            if f.get("size", 0) > self.MAX_FILE_BYTES:
                continue
            name = f["name"].lower()
            if not any(name.endswith(ext) for ext in [".md", ".txt", ".rst", ".json"]):
                continue

            content = self._download_text(f["download_url"])
            if not content:
                continue

            if name.endswith(".json"):
                try:
                    data = json.loads(content)
                    self._extract_from_json(data, source_key, texts, metas, limit)
                except json.JSONDecodeError:
                    pass
            else:
                clean = self._strip_markdown(content)
                for chunk in self._chunk_text(clean):
                    if len(texts) >= limit:
                        break
                    texts.append(chunk)
                    metas.append({"source": source_key, "type": "akademik"})

            if len(texts) >= limit:
                break

        t = texts[:limit]
        return self.rag.index_texts(t, source_key, metas[:limit]), t

    def _load_generic(self, repo: str, source_key: str, limit: int):
        files  = self._list_repo(repo)
        texts: List[str] = []
        metas: List[Dict] = []

        for f in files:
            if f.get("type") != "file":
                continue
            if f.get("size", 0) > self.MAX_FILE_BYTES:
                continue
            name = f["name"].lower()
            if not any(name.endswith(ext) for ext in [".txt", ".json", ".md"]):
                continue

            content = self._download_text(f["download_url"])
            if not content:
                continue

            for chunk in self._chunk_text(content):
                if len(texts) >= limit:
                    break
                texts.append(chunk)
                metas.append({"source": source_key, "type": "genel"})

            if len(texts) >= limit:
                break

        t = texts[:limit]
        return self.rag.index_texts(t, source_key, metas[:limit]), t

    # ──────────────────────────────────────────────────────────
    #  LOG
    # ──────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Mergen RAG] {msg}")
