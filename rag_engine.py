# -*- coding: utf-8 -*-
"""
Mergen RAG Engine — Transformer-Free

ChromaDB + BioVectorizer + HTMRetriever üçlüsü:

  1. BioVectorizer   — karakter n-gram hash + rastgele projeksiyon
                       (Attention / BERT / SentenceTransformer YOK)
  2. ChromaDB        — kalıcı vektör veritabanı
  3. HTMRetriever    — SDR örtüşmesi + yayılımsal aktivasyon + yanal inhibisyon
                       (softmax olasılığı YOK)
"""

import hashlib
import time
from typing import Any, Dict, List, Optional

import numpy as np

# ARCH-04 FIX: SemanticMemory dead code'dan kurtarıldı.
# index_texts() artık her belgeyi SemanticMemory'ye de kaydeder.
try:
    from memory.semantic import SemanticMemory as _SemanticMemory
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False


class RAGEngine:
    """
    Mergen için tamamen Transformer-free semantik bilgi arama motoru.

    Embedding: BioVectorizer (n-gram hash + projeksiyon, 512 boyut)
    Arama:     ChromaDB kosinüs → HTMRetriever yeniden sıralama
    """

    COLLECTION_NAME = "mergen_bilgi_bio"   # "bio" son eki: BioVec ile oluşturuldu
    DREAM_COLLECTION_NAME = "mergen_ruya_bio"
    EMBED_DIM       = 512                   # BioVectorizer çıkış boyutu

    def __init__(self, db_path: str = "./mergen_rag_db"):
        self.db_path     = db_path
        self._client     = None
        self._collection = None
        self._dream_collection = None
        self._vectorizer = None
        self._htm        = None
        self._ready      = False
        # ARCH-04 FIX: SemanticMemory artık aktif — deduplikasyon ve ağırlıklandırma
        self._semantic_mem = _SemanticMemory() if _SEMANTIC_AVAILABLE else None

    # ──────────────────────────────────────────────────────────
    #  BAŞLATMA
    # ──────────────────────────────────────────────────────────

    def initialize(self, verbose: bool = True) -> bool:
        """BioVectorizer, ChromaDB ve HTMRetriever'ı başlat."""

        # 1. BioVectorizer — Transformer gerektirmez
        try:
            from bio_vectorizer import BioVectorizer
            self._vectorizer = BioVectorizer(dim=self.EMBED_DIM)
            if verbose:
                print(f"[Mergen RAG] BioVectorizer aktif (dim={self.EMBED_DIM}, Transformer-free).")
        except ImportError as e:
            if verbose:
                print(f"[Mergen RAG] bio_vectorizer.py bulunamadı: {e}")
            return False

        # 2. ChromaDB — kalıcı vektör deposu
        try:
            import chromadb
        except ImportError:
            if verbose:
                print("[Mergen RAG] chromadb yüklü değil.  →  pip install chromadb")
            return False

        try:
            self._client = chromadb.PersistentClient(path=self.db_path)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._dream_collection = self._client.get_or_create_collection(
                name=self.DREAM_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            existing = self._collection.count()
            existing_dream = self._dream_collection.count()
            if verbose:
                print(f"[Mergen RAG] ChromaDB hazır — {existing} kayıt, {existing_dream} rüya kaydı.")
        except Exception as e:
            if verbose:
                print(f"[Mergen RAG] ChromaDB hatası: {e}")
            return False

        # 3. HTMRetriever — biyolojik yeniden sıralama
        try:
            from htm_retriever import HTMRetriever
            self._htm = HTMRetriever()
            if verbose:
                print("[Mergen RAG] HTMRetriever aktif (SDR örtüşmesi + yayılımsal aktivasyon).")
        except ImportError:
            if verbose:
                print("[Mergen RAG] htm_retriever.py bulunamadı — kosinüs sıralaması kullanılacak.")

        self._ready = True
        return True

    # ──────────────────────────────────────────────────────────
    #  EMBEDDING
    # ──────────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """BioVectorizer ile Transformer-free vektörleme."""
        vecs = self._vectorizer.encode(texts)
        return vecs.tolist()

    def embed_single(self, text: str) -> np.ndarray:
        return self._vectorizer.encode_single(text)

    # ──────────────────────────────────────────────────────────
    #  İNDEKSLEME
    # ──────────────────────────────────────────────────────────

    def index_texts(
        self,
        texts:      List[str],
        source:     str,
        metadatas:  Optional[List[Dict]] = None,
        batch_size: int = 256,
    ) -> int:
        """
        Metinleri BioVectorizer ile vektörleştirip ChromaDB'ye ekle.
        Tekrar yükleme: upsert (deterministik ID → çakışma yok).
        """
        if not self._ready:
            return 0

        texts = [t.strip() for t in texts if t and len(t.strip()) > 15]
        if not texts:
            return 0

        if metadatas is None:
            # Varsayılan olarak semantik bellek tipini atayalım
            metadatas = [{"source": source, "memory_type": "semantic"}] * len(texts)
        else:
            for meta in metadatas:
                if "memory_type" not in meta:
                    meta["memory_type"] = "semantic"
                meta["source"] = meta.get("source", source)

        indexed = 0
        for i in range(0, len(texts), batch_size):
            batch_t = texts[i : i + batch_size]
            batch_m = metadatas[i : i + batch_size]

            ids = [
                hashlib.md5(
                    f"{source}::{i + j}::{t[:60]}".encode("utf-8")
                ).hexdigest()
                for j, t in enumerate(batch_t)
            ]

            try:
                embeddings = self._embed(batch_t)
                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=batch_t,
                    metadatas=batch_m,
                )
                indexed += len(batch_t)

                # ARCH-04 FIX: SemanticMemory'ye de yaz — deduplikasyon ve erişim sayacı
                if self._semantic_mem is not None:
                    for t in batch_t:
                        try:
                            self._semantic_mem.add_fact(
                                text=t,
                                concept_ids=[],   # Embedding gerektirmeden basit kayıt
                                weight=1.0,
                            )
                        except Exception:
                            pass

            except Exception as e:
                print(f"[Mergen RAG] İndeksleme hatası (batch {i}): {e}")

        return indexed

    # ──────────────────────────────────────────────────────────
    #  ARAMA — HTM Biyolojik Dikkat
    # ──────────────────────────────────────────────────────────

    def search(
        self,
        query:         str,
        top_k:         int = 5,
        source_filter: Optional[str] = None,
        memory_type:   Optional[str] = None,
        min_relevance: float = 0.25,
    ) -> List[Dict]:
        """
        Semantik arama + HTM yeniden sıralama.

        Adım 1: BioVectorizer ile sorgu vektörü
        Adım 2: ChromaDB kosinüs benzerliği (top_k × 2 aday)
        Adım 3: HTMRetriever SDR örtüşme + yayılım + inhibisyon ile yeniden sırala
        Adım 4: min_relevance eşiği filtrelemesi
        """
        if not self._ready or not query.strip():
            return []

        total = self._collection.count()
        if total == 0:
            return []

        try:
            q_vec      = self._vectorizer.encode([query.strip()])[0]  # (dim,)
            q_emb_list = [q_vec.tolist()]

            # Daha fazla aday al — HTM sonra en iyilerini seçecek
            n_candidates = min(top_k * 3, total)

            kwargs: Dict[str, Any] = {
                "query_embeddings": q_emb_list,
                "n_results":        n_candidates,
                "include":          ["documents", "distances", "metadatas", "embeddings"],
            }
            if source_filter or memory_type:
                where_clause = {}
                if source_filter:
                    where_clause["source"] = source_filter
                if memory_type:
                    where_clause["memory_type"] = memory_type
                
                # Eğer birden fazla koşul varsa $and kullan
                if len(where_clause) > 1:
                    kwargs["where"] = {"$and": [{k: v} for k, v in where_clause.items()]}
                else:
                    kwargs["where"] = where_clause

            results = self._collection.query(**kwargs)

            docs      = results["documents"][0]
            distances = results["distances"][0]
            metas     = results["metadatas"][0]
            raw_embeddings = results.get("embeddings")
            embeddings = raw_embeddings[0] if raw_embeddings is not None else None

            if not docs:
                return []

            # Kosinüs mesafesi → benzerlik skoru
            cos_scores = np.array([max(0.0, 1.0 - d) for d in distances], dtype=np.float32)

            # HTM yeniden sıralama (embedding'ler mevcutsa)
            if self._htm is not None and embeddings is not None and len(embeddings) == len(docs):
                cand_vecs = np.array(embeddings, dtype=np.float32)
                ranked_idx = self._htm.rerank(
                    query_vec=q_vec,
                    candidate_vecs=cand_vecs,
                    cosine_scores=cos_scores,
                    top_k=top_k,
                )
            else:
                # HTM yoksa saf kosinüs sırası
                ranked_idx = np.argsort(cos_scores)[::-1][:top_k].tolist()

            hits = []
            for idx in ranked_idx:
                relevance = float(cos_scores[idx])
                if relevance < min_relevance:
                    continue
                hits.append({
                    "text":      docs[idx],
                    "relevance": round(relevance, 4),
                    "source":    metas[idx].get("source", "rag"),
                    "type":      metas[idx].get("type",   "fact"),
                })

            return hits

        except Exception as e:
            print(f"[Mergen RAG] Arama hatası: {e}")
            return []

    # ──────────────────────────────────────────────────────────
    #  YARDIMCI
    # ──────────────────────────────────────────────────────────

    def count(self) -> int:
        if not self._ready or not self._collection:
            return 0
        try:
            return self._collection.count()
        except Exception as e:
            print(f"[Mergen RAG] Count error: {e}")
            return 0

    def is_source_indexed(self, source: str, min_count: int = 10) -> bool:
        if not self._ready:
            return False
        try:
            r = self._collection.get(where={"source": source}, limit=min_count)
            return len(r["ids"]) >= min_count
        except Exception as e:
            print(f"[Mergen RAG] Source status error ({source}): {e}")
            return False

    @property
    def ready(self) -> bool:
        return self._ready

    def add_dream_fact(self, text: str, confidence: float = 0.2) -> bool:
        """
        Sentetik rüya çıkarımını (insight) mergen_ruya_bio koleksiyonuna kaydeder.
        Katı metadata izolasyonu uygulanır.
        """
        if not self._ready or not self._dream_collection:
            return False

        text = text.strip()
        if len(text) <= 15:
            return False

        doc_id = hashlib.md5(f"DREAM::{time.time()}::{text[:60]}".encode("utf-8")).hexdigest()
        metadata = {
            "source": "DREAM",
            "reliability": "synthetic",
            "confidence": confidence
        }

        try:
            q_vec = self._vectorizer.encode([text])[0].tolist()
            self._dream_collection.upsert(
                ids=[doc_id],
                embeddings=[q_vec],
                documents=[text],
                metadatas=[metadata]
            )
            return True
        except Exception as e:
            print(f"[Mergen RAG] Rüya kaydetme hatası: {e}")
            return False

    def query_dream(self, query: str, top_k: int = 5, min_relevance: float = 0.25) -> List[Dict]:
        """
        Sentetik rüya koleksiyonu mergen_ruya_bio üzerinde semantik arama yapar.
        """
        if not self._ready or not self._dream_collection or not query.strip():
            return []

        total = self._dream_collection.count()
        if total == 0:
            return []

        try:
            q_vec      = self._vectorizer.encode([query.strip()])[0]  # (dim,)
            q_emb_list = [q_vec.tolist()]

            n_candidates = min(top_k * 3, total)

            results = self._dream_collection.query(
                query_embeddings=q_emb_list,
                n_results=n_candidates,
                include=["documents", "distances", "metadatas", "embeddings"]
            )

            docs      = results["documents"][0]
            distances = results["distances"][0]
            metas     = results["metadatas"][0]
            raw_embeddings = results.get("embeddings")
            embeddings = raw_embeddings[0] if raw_embeddings is not None else None

            if not docs:
                return []

            # Kosinüs mesafesi → benzerlik skoru
            cos_scores = np.array([max(0.0, 1.0 - d) for d in distances], dtype=np.float32)

            # HTM yeniden sıralama
            if self._htm is not None and embeddings is not None and len(embeddings) == len(docs):
                cand_vecs = np.array(embeddings, dtype=np.float32)
                ranked_idx = self._htm.rerank(
                    query_vec=q_vec,
                    candidate_vecs=cand_vecs,
                    cosine_scores=cos_scores,
                    top_k=top_k,
                )
            else:
                ranked_idx = np.argsort(cos_scores)[::-1][:top_k].tolist()

            hits = []
            for idx in ranked_idx:
                relevance = float(cos_scores[idx])
                if relevance < min_relevance:
                    continue
                hits.append({
                    "text":      docs[idx],
                    "relevance": round(relevance, 4),
                    "source":    metas[idx].get("source", "DREAM"),
                    "reliability": metas[idx].get("reliability", "synthetic"),
                    "confidence": float(metas[idx].get("confidence", 0.2)),
                })

            return hits

        except Exception as e:
            print(f"[Mergen RAG] Rüya arama hatası: {e}")
            return []
