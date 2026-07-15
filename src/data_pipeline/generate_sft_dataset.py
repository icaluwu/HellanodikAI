"""
generate_sft_dataset.py — Script untuk membuat dataset Q&A sintetis (SFT) secara otomatis.

Fungsi utama:
1. Membaca data/processed/pasal_corpus.json.
2. Menggunakan LLM (via transformers) untuk membuat pasangan pertanyaan awam
   dan jawaban edukatif yang bersahabat untuk tiap pasal.
3. Menyimpan hasil generator ke data/processed/sft_dataset.jsonl.
"""

import sys
import json
import re
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import torch

# Hubungkan path ke parent dir agar bisa import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg

from src.rag.indexer import get_enriched_text

PROMPT_TEMPLATE = """Anda adalah asisten hukum pidana Indonesia yang cerdas dan edukatif.
Tugas Anda adalah membuat 1 pasang pertanyaan dan jawaban berdasarkan pasal undang-undang berikut.

Kriteria:
1. PERTANYAAN: Harus menggunakan bahasa sehari-hari yang santai, awam, dan natural (seolah-olah ditanyakan oleh masyarakat biasa yang tidak mengerti hukum). Jangan memakai bahasa formal hukum di pertanyaan.
2. JAWABAN: Harus menjelaskan maksud pasal tersebut secara sederhana, mudah dipahami, bersahabat, bersikap netral (edukatif, bukan nasihat hukum formal, tidak menjatuhkan vonis), dan WAJIB menyertakan rujukan pasal secara akurat (misal: "Berdasarkan Pasal X UU Y...").

Berikut adalah pasal rujukan:
{enriched_text}

Format output harus berupa JSON objek persis seperti ini (tanpa markdown, tanpa teks tambahan lain):
{{
  "instruction": "Tulis pertanyaan awam di sini...",
  "response": "Tulis jawaban edukatif dengan rujukan pasal di sini..."
}}"""

def extract_json(text: str) -> dict | None:
    """Mengekstrak JSON objek dari teks output model."""
    try:
        # Cari pola curly braces {}
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return None

def main():
    corpus_path = cfg.DATA_PROCESSED_DIR / "pasal_corpus.json"
    output_path = cfg.DATA_PROCESSED_DIR / "sft_dataset.jsonl"
    
    if not corpus_path.exists():
        logger.error(f"Corpus tidak ditemukan di {corpus_path}. Jalankan parse.py terlebih dahulu.")
        sys.exit(1)
        
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    logger.info(f"Loaded {len(corpus)} entri pasal dari corpus.")
    
    # Inisialisasi model generator (menggunakan model terkecil/efisien untuk running di Kaggle T4 GPU)
    # Kami menggunakan Qwen2.5-7B-Instruct sebagai default generator di Kaggle
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Menggunakan device: {device} untuk generasi dataset.")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    
    generator_model_id = "Qwen/Qwen2.5-7B-Instruct"
    logger.info(f"Loading generator model: {generator_model_id} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(generator_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        generator_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )
    
    sft_data = []
    
    logger.info("Memulai pembuatan pasangan Q&A sintetis...")
    # Batasi jumlah sampel untuk mempercepat (misal 300 sampel representatif untuk demo,
    # atau proses seluruh corpus jika berjalan di Kaggle)
    # Di Kaggle, Anda bisa menghapus batas slice [:150] ini untuk memproses seluruh corpus.
    sample_corpus = corpus[:150] # Mengambil 150 sampel representatif untuk efisiensi
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for idx, entry in enumerate(tqdm(sample_corpus, desc="Generating QA Pairs")):
            enriched = get_enriched_text(entry)
            prompt = PROMPT_TEMPLATE.format(enriched_text=enriched)
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            # Ganti dengan format chat template Qwen
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            try:
                outputs = pipe(formatted_prompt)
                generated_text = outputs[0]["generated_text"]
                
                # Cari bagian asisten setelah prompt
                assistant_text = generated_text[len(formatted_prompt):].strip()
                
                qa_data = extract_json(assistant_text)
                if qa_data and "instruction" in qa_data and "response" in qa_data:
                    # Tambahkan metadata
                    qa_data["context"] = enriched
                    qa_data["pasal"] = entry.get("pasal") or entry.get("pasal_romawi")
                    qa_data["uu"] = entry.get("uu")
                    
                    # Simpan langsung ke JSONL
                    f_out.write(json.dumps(qa_data, ensure_ascii=False) + "\n")
                    sft_data.append(qa_data)
                else:
                    # Fallback jika model gagal mengeluarkan JSON valid
                    fallback_data = {
                        "instruction": f"Bagaimana isi dari {entry.get('uu')} tentang pasal {entry.get('pasal') or entry.get('pasal_romawi')}?",
                        "response": f"Berdasarkan {entry.get('uu')} - {entry.get('bab')}, {entry.get('pasal') or entry.get('pasal_romawi')} mengatur: {entry.get('teks')}",
                        "context": enriched,
                        "pasal": entry.get("pasal") or entry.get("pasal_romawi"),
                        "uu": entry.get("uu")
                    }
                    f_out.write(json.dumps(fallback_data, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Gagal generate untuk indeks {idx}: {e}")
                
    logger.info(f"Pembuatan SFT Dataset selesai. {len(sft_data)} pasangan Q&A berhasil disimpan ke {output_path}")

if __name__ == "__main__":
    main()
