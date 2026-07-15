# HellanodikAI

**Asisten Literasi Hukum Pidana Indonesia**  
*Built with Llama · Responsible AI Licenses (RAIL) · CC-BY-4.0 (dataset) · Llama Community License (model weights)*

> HellanodikAI membantu masyarakat awam memahami **KUHP baru (UU No. 1 Tahun 2023, berlaku 2 Januari 2026)** dan perubahannya dalam bahasa yang sederhana, dilengkapi rujukan pasal yang akurat.
>
> ⚠️ **Ini adalah asisten INFORMASI/EDUKASI, bukan pengganti nasihat hukum profesional.**

---

## Arsitektur

```
Query → [Safety Filter] → [RAG: BM25 + FAISS + CrossEncoder] → [LLM Inference] → [Post-processor + Disclaimer] → Respons
```

- **Base model:** Sahabat-AI (`GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct`) — fine-tuned dengan QLoRA
- **Fine-tuned model:** `Llama-HellanodikAI-8B` (HuggingFace Hub)
- **RAG:** BM25 + FAISS dense retrieval + CrossEncoder reranker
- **Local inference:** `llama-cpp-python` (GGUF, CPU-only)
- **UI:** Gradio

---

## Struktur Proyek

```
hellanodikai/
├── config.py                   # Konfigurasi terpusat (MODEL_ID, path, dll.)
├── .env.example                # Template env vars
├── requirements/
│   ├── base.txt                # Shared deps
│   ├── local.txt               # CPU-only inference
│   ├── kaggle.txt              # Fine-tuning (GPU, Kaggle)
│   └── gradio.txt              # Deployment
├── data/
│   ├── raw/                    # PDF sumber hukum
│   ├── processed/corpus/       # Corpus pasal (JSON)
│   ├── processed/sft_dataset/  # Dataset fine-tuning
│   └── index/                  # BM25 + FAISS index
├── src/
│   ├── data_pipeline/          # PDF extraction, pasal parsing, dataset builder
│   ├── rag/                    # Indexer, retriever, reranker
│   ├── inference/              # Model loader, RAG+LLM pipeline
│   ├── training/               # Fine-tuning (Kaggle only)
│   └── utils/                  # Prompt templates, safety layer
├── app/
│   └── main.py                 # Gradio entry point
├── notebooks/
│   └── kaggle_finetune.ipynb   # Notebook fine-tuning Kaggle
├── NOTICE                      # Llama Community License notice
└── LICENSE                     # Apache-2.0
```

---

## Quick Start

### 1. Setup environment

```bash
# Clone repo
git clone https://github.com/yourusername/hellanodikai.git
cd hellanodikai

# Buat virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# Install dependencies (CPU-only lokal)
pip install -r requirements/local.txt

# Setup env vars
copy .env.example .env
# Edit .env — isi HF_TOKEN dan sesuaikan konfigurasi
```

### 2. Verifikasi config

```bash
python config.py
```

### 3. Build data pipeline (Phase 1)

```bash
python -m src.data_pipeline.pdf_extractor
```

### 4. Build RAG index (Phase 2)

```bash
python -m src.rag.indexer
```

### 5. Jalankan app

```bash
python app/main.py
# Buka http://localhost:7860
```

---

## Fine-tuning (Kaggle)

Fine-tuning dilakukan di Kaggle dengan NVIDIA T4 x2, bukan di lokal.  
Lihat [`notebooks/kaggle_finetune.ipynb`](notebooks/kaggle_finetune.ipynb) untuk instruksi lengkap.

Stack: **Unsloth + QLoRA (4-bit) + TRL SFTTrainer**

---

## Dokumen Hukum yang Dicakup

| Dokumen | Keterangan |
|---------|-----------|
| UU No. 1 Tahun 2023 | KUHP baru (berlaku 2 Januari 2026) |
| UU No. 1 Tahun 2026 | Penyesuaian Tindak Pidana |
| UU No. 20 Tahun 2025 | (lihat dataset_primer/) |

---

## Lisensi

| Komponen | Lisensi |
|----------|---------|
| Kode sumber | [Apache-2.0](LICENSE) |
| Dataset kurasi | CC-BY-4.0 |
| Model weights (Llama-HellanodikAI-8B) | [Llama Community License](https://llama.meta.com/llama-downloads) |

**Built with Llama.** Lihat [NOTICE](NOTICE) untuk detail atribusi lengkap.

---

## Disclaimer

HellanodikAI adalah alat informasi dan edukasi hukum. Informasi yang diberikan **bukan nasihat hukum profesional** dan tidak dapat dijadikan dasar tindakan hukum. Untuk kasus hukum konkret, konsultasikan dengan advokat berlisensi atau hubungi Lembaga Bantuan Hukum (LBH) terdekat.
