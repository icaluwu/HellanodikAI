"""
indexer.py — Script untuk membuat indeks pencarian sparse (BM25) dan dense (FAISS)

Fungsi utama:
1. Membaca data/processed/pasal_corpus.json.
2. Memperkaya representasi teks pasal (UU + Bab + Nomor Pasal + Isi) untuk embedding.
3. Mem-build indeks sparse BM25 menggunakan rank_bm25 dan menyimpannya ke pickle.
4. Mem-build indeks dense FAISS menggunakan faiss-cpu dan sentence-transformers,
   kemudian menyimpannya ke disk beserta file metadatanya.
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

# FAISS dan SentenceTransformer di-import di dalam fungsi agar script tidak error
# jika library belum selesai di-install saat file ini ditulis.

def clean_and_tokenize(text: str) -> list[str]:
    """Tokenisasi kata sederhana untuk BM25."""
    s = text.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return s.split()

def get_enriched_text(entry: dict) -> str:
    """Menggabungkan metadata UU, Bab, dan nomor pasal untuk memperkaya konteks pencarian."""
    uu = entry.get("uu") or ""
    bab = entry.get("bab") or ""
    pasal = entry.get("pasal")
    pasal_rom = entry.get("pasal_romawi")
    
    p_num = ""
    if pasal:
        p_num = f"Pasal {pasal}"
    elif pasal_rom:
        p_num = f"Pasal {pasal_rom}"
        
    teks = entry.get("teks") or ""
    return f"{uu} - {bab} - {p_num} - {teks}"

def build_bm25_index(corpus: list[dict], output_path: Path):
    """Membangun dan menyimpan indeks BM25."""
    logger.info("Membangun indeks BM25...")
    from rank_bm25 import BM25Okapi
    
    # Tokenisasi setiap pasal yang diperkaya
    tokenized_corpus = []
    for entry in corpus:
        enriched = get_enriched_text(entry)
        tokenized_corpus.append(clean_and_tokenize(enriched))
        
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Simpan ke file pickle
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(bm25, f)
    logger.info(f"Indeks BM25 berhasil disimpan ke {output_path}")

def build_faiss_index(corpus: list[dict], index_path: Path, meta_path: Path):
    """Membangun dan menyimpan indeks FAISS."""
    logger.info("Membangun indeks FAISS...")
    import faiss
    from sentence_transformers import SentenceTransformer
    
    # Inisialisasi model embedding
    logger.info(f"Loading embedding model: {cfg.EMBEDDING_MODEL} ...")
    model = SentenceTransformer(cfg.EMBEDDING_MODEL, device="cpu")
    
    # Siapkan teks yang diperkaya
    texts = [get_enriched_text(entry) for entry in corpus]
    
    # Hitung embeddings
    logger.info("Menghitung embeddings (proses ini berjalan di CPU)...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Buat indeks FAISS (Cosine Similarity via Inner Product pada normalisasi L2)
    dimension = embeddings.shape[1]
    logger.info(f"Dimensi embedding: {dimension}")
    
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Simpan indeks FAISS
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info(f"Indeks FAISS berhasil disimpan ke {index_path}")
    
    # Simpan metadata FAISS (mapping index ke entri pasal asli)
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(corpus, f_meta, ensure_ascii=False, indent=2)
    logger.info(f"Metadata FAISS berhasil disimpan ke {meta_path}")

def main():
    corpus_path = cfg.DATA_PROCESSED_DIR / "pasal_corpus.json"
    if not corpus_path.exists():
        logger.error(f"File corpus tidak ditemukan di {corpus_path}. Jalankan parse.py terlebih dahulu.")
        sys.exit(1)
        
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    logger.info(f"Loaded {len(corpus)} entri pasal dari corpus.")
    
    # 1. Build BM25
    build_bm25_index(corpus, cfg.BM25_INDEX_PATH)
    
    # 2. Build FAISS
    build_faiss_index(corpus, cfg.FAISS_INDEX_PATH, cfg.FAISS_META_PATH)
    
    logger.info("=== PROSES INDEXING SELESAI DENGAN SUKSES ===")

if __name__ == "__main__":
    main()
