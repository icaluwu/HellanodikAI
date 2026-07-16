"""
retriever.py — Modul retriever hybrid (BM25 + FAISS + CrossEncoder Reranker)

Fungsi utama:
1. Load indeks BM25 dan FAISS yang telah dibuat oleh indexer.
2. Melakukan hybrid search: mengambil top kandidat dari BM25 (sparse) dan FAISS (dense).
3. Melakukan penggabungan (deduplikasi) kandidat pencarian.
4. Melakukan perankingan ulang (reranking) menggunakan model CrossEncoder.
5. Mengembalikan daftar pasal terurut berdasarkan tingkat relevansi tertinggi.
"""

import sys
import pickle
import json
import re
from pathlib import Path
from loguru import logger
import numpy as np

# Hubungkan path ke parent dir agar bisa import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg

from src.rag.indexer import clean_and_tokenize, get_enriched_text

class HybridRetriever:
    def __init__(self):
        logger.info("Menginisialisasi HybridRetriever...")
        
        # 1. Load database corpus utama
        self.corpus_path = cfg.DATA_PROCESSED_DIR / "pasal_corpus.json"
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus tidak ditemukan di {self.corpus_path}. Jalankan parse.py terlebih dahulu.")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)
        logger.info(f"Loaded {len(self.corpus)} entri pasal ke memory.")
        
        # 2. Load BM25 Index
        if not cfg.BM25_INDEX_PATH.exists():
            raise FileNotFoundError(f"Indeks BM25 tidak ditemukan di {cfg.BM25_INDEX_PATH}. Jalankan indexer.py terlebih dahulu.")
        with open(cfg.BM25_INDEX_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        logger.info("Indeks BM25 berhasil dimuat.")
        
        # 3. Load FAISS Index & Metadata (with fallback)
        self.use_dense = True
        try:
            import faiss
            if not cfg.FAISS_INDEX_PATH.exists():
                raise FileNotFoundError(f"Indeks FAISS tidak ditemukan di {cfg.FAISS_INDEX_PATH}. Jalankan indexer.py terlebih dahulu.")
            self.faiss_index = faiss.read_index(str(cfg.FAISS_INDEX_PATH))
            
            if not cfg.FAISS_META_PATH.exists():
                raise FileNotFoundError(f"Metadata FAISS tidak ditemukan di {cfg.FAISS_META_PATH}.")
            with open(cfg.FAISS_META_PATH, "r", encoding="utf-8") as f:
                self.faiss_metadata = json.load(f)
            logger.info("Indeks FAISS dan metadata berhasil dimuat.")
            
            # 4. Load Models (SentenceTransformer & CrossEncoder)
            from sentence_transformers import SentenceTransformer, CrossEncoder
            logger.info(f"Loading embedding model: {cfg.EMBEDDING_MODEL} ...")
            self.embed_model = SentenceTransformer(cfg.EMBEDDING_MODEL, device="cpu")
            
            logger.info(f"Loading reranker model: {cfg.RERANKER_MODEL} ...")
            self.rerank_model = CrossEncoder(cfg.RERANKER_MODEL, device="cpu")
        except Exception as e:
            logger.warning(f"Gagal memuat sistem dense FAISS/Transformers (error: {e}). Beralih ke mode pencarian BM25 saja secara penuh.")
            self.use_dense = False
            self.faiss_index = None
            self.faiss_metadata = None
            self.embed_model = None
            self.rerank_model = None
            
        logger.info("HybridRetriever siap digunakan.")

    def retrieve(self, query: str, top_k_bm25: int = None, top_k_faiss: int = None, rerank_k: int = None) -> list[dict]:
        """
        Mengambil pasal paling relevan menggunakan pencarian hybrid + reranking.
        """
        import faiss
        
        k_bm25 = top_k_bm25 or cfg.BM25_TOP_K
        k_faiss = top_k_faiss or cfg.FAISS_TOP_K
        k_rerank = rerank_k or cfg.RERANKER_TOP_K
        
        logger.info(f"Retrieving for query: '{query}' (BM25={k_bm25}, FAISS={k_faiss}, Rerank={k_rerank})")
        
        # Fallback jika model dense / PyTorch tidak dapat dimuat
        if not self.use_dense:
            tokenized_query = clean_and_tokenize(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:k_rerank]
            
            results = []
            for idx in top_bm25_indices:
                score = bm25_scores[idx]
                if score > 0.0:
                    entry_copy = self.corpus[idx].copy()
                    entry_copy["retrieval_score"] = float(score)
                    entry_copy["retrieval_source"] = "BM25"
                    results.append(entry_copy)
            logger.info(f"Berhasil mengambil top {len(results)} pasal via BM25 (Dense disabled).")
            return results

        # -------------------------------------------------------------------
        # 1. Sparse Search (BM25)
        # -------------------------------------------------------------------
        tokenized_query = clean_and_tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Ambil indeks dengan skor tertinggi
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:k_bm25]
        
        candidates = {}  # key: index in self.corpus, value: (entry, source)
        for idx in top_bm25_indices:
            score = bm25_scores[idx]
            if score > 0.0:  # Hanya ambil dokumen dengan kemiripan kata positif
                candidates[idx] = (self.corpus[idx], "BM25")
                
        # -------------------------------------------------------------------
        # 2. Dense Search (FAISS)
        # -------------------------------------------------------------------
        query_vector = self.embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.faiss_index.search(query_vector, k_faiss)
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # -1 menunjukkan tidak ada kecocokan
                # Gunakan metadata untuk mencocokkan index ke entri
                # Karena metadatanya sama dengan self.corpus
                if idx < len(self.corpus):
                    if idx in candidates:
                        # Jika sudah terdeteksi BM25, ubah source menjadi Hybrid
                        candidates[idx] = (self.corpus[idx], "Hybrid")
                    else:
                        candidates[idx] = (self.corpus[idx], "FAISS")
                        
        if not candidates:
            logger.warning("Tidak ditemukan kandidat pasal untuk query ini.")
            return []
            
        # -------------------------------------------------------------------
        # 3. Reranking (CrossEncoder)
        # -------------------------------------------------------------------
        candidate_indices = list(candidates.keys())
        pairs = []
        for idx in candidate_indices:
            entry = self.corpus[idx]
            enriched = get_enriched_text(entry)
            pairs.append((query, enriched))
            
        # Hitung skor reranking
        logger.info(f"Reranking {len(pairs)} kandidat...")
        rerank_scores = self.rerank_model.predict(pairs)
        
        # Masukkan skor dan source ke tiap entri
        results = []
        for i, idx in enumerate(candidate_indices):
            entry, source = candidates[idx]
            # Copy entri agar tidak merusak corpus di memory
            entry_copy = entry.copy()
            entry_copy["retrieval_score"] = float(rerank_scores[i])
            entry_copy["retrieval_source"] = source
            results.append(entry_copy)
            
        # Urutkan berdasarkan skor reranking tertinggi
        results.sort(key=lambda x: x["retrieval_score"], reverse=True)
        
        # Ambil top K hasil reranking
        top_results = results[:k_rerank]
        logger.info(f"Berhasil merangking top {len(top_results)} pasal.")
        return top_results

if __name__ == "__main__":
    # Uji inisialisasi singkat jika dijalankan langsung
    try:
        retriever = HybridRetriever()
        res = retriever.retrieve("zina delik aduan")
        for r in res:
            p_num = f"Pasal {r.get('pasal') or r.get('pasal_romawi')}"
            print(f"[{r['retrieval_source']}][Score: {r['retrieval_score']:.4f}] {r['uu']} - {p_num}")
    except Exception as e:
        print(f"Inisialisasi test dilewati / belum di-index: {e}")
