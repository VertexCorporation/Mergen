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
from typing import Any, Dict, List, Optional

import numpy as np


class RAGEngine:
    """
    Mergen için tamamen Transformer-free semantik bilgi arama motoru.

    Embedding: BioVectorizer (n-gram hash + projeksiyon, 512 boyut)
    Arama:     ChromaDB kosinüs → HTMRetriever yeniden sıralama
    """

    COLLECTION_NAME = "mergen_bilgi_bio"   # "bio" son eki: BioVec ile oluşturuldu
    EMBED_DIM       = 512                   # BioVectorizer çıkış boyutu

    def __init__(self, db_path: str = "./mergen_rag_db"):
        self.db_path     = db_path
        self._client     = None
        self._collection = None
        self._vectorizer = None
        self._htm        = None
        self._ready      = False

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
            existing = self._collection.count()
            if verbose:
                print(f"[Mergen RAG] ChromaDB hazır — {existing} kayıt.")
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
            metadatas = [{"source": source}] * len(texts)

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
            if source_filter:
                kwargs["where"] = {"source": source_filter}

            results = self._collection.query(**kwargs)

            docs      = results["documents"][0]
            distances = results["distances"][0]
            metas     = results["metadatas"][0]
            embeddings = results.get("embeddings", [[]])[0]  # Aday vektörleri

            if not docs:
                return []

            # Kosinüs mesafesi → benzerlik skoru
            cos_scores = np.array([max(0.0, 1.0 - d) for d in distances], dtype=np.float32)

            # HTM yeniden sıralama (embedding'ler mevcutsa)
            if self._htm is not None and embeddings and len(embeddings) == len(docs):
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
            return []

    # ──────────────────────────────────────────────────────────
    #  YARDIMCI
    # ──────────────────────────────────────────────────────────

    def count(self) -> int:
        if not self._ready or not self._collection:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def is_source_indexed(self, source: str, min_count: int = 10) -> bool:
        if not self._ready:
            return False
        try:
            r = self._collection.get(where={"source": source}, limit=min_count)
            return len(r["ids"]) >= min_count
        except Exception:
            return False

    @property
    def ready(self) -> bool:
        return self._ready
