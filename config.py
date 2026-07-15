"""
config.py — Konfigurasi terpusat HellanodikAI
Semua parameter yang mungkin berubah didefinisikan di sini.
Ubah nilai-nilai ini untuk beralih antara model, path, atau hyperparameter.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR          = BASE_DIR / "data"
RAW_DIR           = DATA_DIR / "raw"
PROCESSED_DIR     = DATA_DIR / "processed"
CORPUS_DIR        = PROCESSED_DIR / "corpus"
SFT_DATASET_DIR   = PROCESSED_DIR / "sft_dataset"
INDEX_DIR         = DATA_DIR / "index"

# Konstanta tambahan yang diminta user
DATASET_PRIMER_DIR = BASE_DIR / "dataset_primer"
DATA_RAW_DIR       = RAW_DIR
DATA_PROCESSED_DIR = PROCESSED_DIR

APP_DIR           = BASE_DIR / "app"
ASSETS_DIR        = APP_DIR / "assets"

# Buat semua direktori jika belum ada
for _dir in [RAW_DIR, CORPUS_DIR, SFT_DATASET_DIR, INDEX_DIR, ASSETS_DIR, DATASET_PRIMER_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Source Documents (PDF)
# ---------------------------------------------------------------------------
# Tambahkan file PDF baru di sini untuk memperluas knowledge base RAG
SOURCE_PDFS: dict[str, Path] = {
    "KUHP_2023":          BASE_DIR / "dataset_primer" / "UU Nomor 1 Tahun 2023.pdf",
    "PENYESUAIAN_2026":   BASE_DIR / "dataset_primer" / "UU Nomor 1 Tahun 2026.pdf",
    "UU_20_2025":         BASE_DIR / "dataset_primer" / "UU Nomor 20 Tahun 2025.pdf",
}

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
# Ganti MODEL_ID di .env atau di sini untuk beralih model tanpa ubah kode lain.
# Opsi:
#   "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct"  ← Sahabat-AI (default)
#   "Qwen/Qwen2.5-7B-Instruct"                          ← fallback lisensi bebas
MODEL_ID: str = os.getenv(
    "MODEL_ID",
    "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct",
)

# Nama repo HF untuk model fine-tuned yang akan di-push
# Harus diawali "Llama-" sesuai Llama Community License
HF_FINETUNED_REPO: str = os.getenv(
    "HF_FINETUNED_REPO",
    "Llama-HellanodikAI-8B",
)

# HuggingFace token (dari .env)
HF_TOKEN: str | None = os.getenv("HF_TOKEN", None)

# ---------------------------------------------------------------------------
# Local Inference (llama-cpp-python / GGUF)
# ---------------------------------------------------------------------------
# Path ke file GGUF lokal setelah download & konversi
GGUF_MODEL_PATH: Path = Path(
    os.getenv("GGUF_MODEL_PATH", str(BASE_DIR / "models" / "hellanodikai.gguf"))
)

# Quantization level yang digunakan untuk GGUF (Q4_K_M direkomendasikan untuk CPU)
GGUF_QUANT: str = os.getenv("GGUF_QUANT", "Q4_K_M")

# Jumlah CPU threads untuk llama-cpp inference
N_THREADS: int = int(os.getenv("N_THREADS", "4"))

# Konteks window (token) untuk inference
N_CTX: int = int(os.getenv("N_CTX", "4096"))

# Jumlah token maksimum yang di-generate per respons
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "1024"))

# Temperature — 0.1 untuk jawaban faktual/hukum (deterministik)
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))

# ---------------------------------------------------------------------------
# RAG Configuration
# ---------------------------------------------------------------------------
# Jumlah dokumen yang diambil oleh masing-masing retriever sebelum reranking
BM25_TOP_K: int      = int(os.getenv("BM25_TOP_K", "10"))
FAISS_TOP_K: int     = int(os.getenv("FAISS_TOP_K", "10"))

# Jumlah dokumen setelah reranking CrossEncoder yang masuk ke prompt LLM
RERANKER_TOP_K: int  = int(os.getenv("RERANKER_TOP_K", "5"))

# Model embedding untuk FAISS (multilingual, CPU-friendly)
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)

# Model CrossEncoder reranker (multilingual/Indonesian-friendly)
RERANKER_MODEL: str = os.getenv(
    "RERANKER_MODEL",
    "madebyaris/rerank-indonesia",
)

# Path file index
BM25_INDEX_PATH:   Path = INDEX_DIR / "bm25_index.pkl"
FAISS_INDEX_PATH:  Path = INDEX_DIR / "faiss_index.bin"
FAISS_META_PATH:   Path = INDEX_DIR / "faiss_metadata.json"
CORPUS_JSON_PATH:  Path = CORPUS_DIR / "corpus.json"

# ---------------------------------------------------------------------------
# Fine-tuning Configuration (dipakai di Kaggle)
# ---------------------------------------------------------------------------
# LoRA rank — trade-off antara ukuran adapter dan kualitas
LORA_R: int         = int(os.getenv("LORA_R", "16"))
LORA_ALPHA: int     = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT: float = float(os.getenv("LORA_DROPOUT", "0.05"))

# Training hyperparameters
TRAIN_EPOCHS: int         = int(os.getenv("TRAIN_EPOCHS", "3"))
TRAIN_BATCH_SIZE: int     = int(os.getenv("TRAIN_BATCH_SIZE", "2"))
GRAD_ACCUM_STEPS: int     = int(os.getenv("GRAD_ACCUM_STEPS", "8"))
LEARNING_RATE: float      = float(os.getenv("LEARNING_RATE", "2e-4"))
MAX_SEQ_LENGTH: int       = int(os.getenv("MAX_SEQ_LENGTH", "2048"))

# ---------------------------------------------------------------------------
# App Configuration
# ---------------------------------------------------------------------------
APP_TITLE:       str = "HellanodikAI"
APP_SUBTITLE:    str = "Asisten Literasi Hukum Pidana Indonesia"
APP_DESCRIPTION: str = (
    "HellanodikAI membantu Anda memahami KUHP baru (UU No. 1/2023, "
    "berlaku 2 Januari 2026) dan perubahannya dalam bahasa yang mudah dipahami. "
    "Setiap penjelasan dilengkapi rujukan pasal yang akurat."
)

# Port Gradio
GRADIO_PORT:  int  = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_SHARE: bool = os.getenv("GRADIO_SHARE", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Safety & Disclaimer
# ---------------------------------------------------------------------------
DISCLAIMER_ID: str = (
    "\n\n---\n"
    "⚠️ **Penting:** Informasi ini bersifat edukatif dan bukan nasihat hukum profesional. "
    "Untuk kasus hukum konkret, konsultasikan dengan advokat atau lembaga bantuan hukum."
)

# Topik di luar scope yang akan ditolak
OUT_OF_SCOPE_KEYWORDS: list[str] = [
    "memutuskan vonis", "putusan pengadilan", "terdakwa bersalah",
    "hukuman yang tepat", "apakah saya bisa menang", "strategi persidangan",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    print("=" * 60)
    print(f"  {APP_TITLE} — Config Check")
    print("=" * 60)

    checks = {
        "MODEL_ID":           MODEL_ID,
        "HF_FINETUNED_REPO":  HF_FINETUNED_REPO,
        "EMBEDDING_MODEL":    EMBEDDING_MODEL,
        "RERANKER_MODEL":     RERANKER_MODEL,
        "GGUF_MODEL_PATH":    str(GGUF_MODEL_PATH),
        "BM25_TOP_K":         BM25_TOP_K,
        "FAISS_TOP_K":        FAISS_TOP_K,
        "RERANKER_TOP_K":     RERANKER_TOP_K,
        "N_CTX":              N_CTX,
        "MAX_NEW_TOKENS":     MAX_NEW_TOKENS,
        "TEMPERATURE":        TEMPERATURE,
        "GRADIO_PORT":        GRADIO_PORT,
    }
    for k, v in checks.items():
        print(f"  {k:<22} = {v}")

    print()
    print("Source PDFs:")
    for name, path in SOURCE_PDFS.items():
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {name}: {path.name}")

    print()
    print("Directories created:")
    for d in [RAW_DIR, CORPUS_DIR, SFT_DATASET_DIR, INDEX_DIR, ASSETS_DIR]:
        print(f"  [OK] {d.relative_to(BASE_DIR)}")

    print()
    print("Config OK — siap lanjut ke Phase 1.")
