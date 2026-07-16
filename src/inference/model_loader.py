"""
model_loader.py — Modul pengambil dan pemuat model LLM HellanodikAI

Mendukung:
1. GGUF via llama-cpp-python (CPU optimal).
2. Fallback ke HuggingFace Transformers (CPU/GPU).
3. Mock mode (tanpa LLM, hemat memori untuk pengembangan lokal).
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Hubungkan path ke parent dir agar bisa import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg

class HellanodikAIModel:
    def __init__(self, force_mock: bool = False):
        self.model_type = None
        self.llm = None
        self.tokenizer = None
        self.pipeline = None
        
        # Check environment or config for mock mode
        self.is_mock = force_mock or os.getenv("MOCK_LLM", "false").lower() == "true"
        
        if self.is_mock:
            logger.info("Menginisialisasi HellanodikAIModel dalam mode MOCK (hemat memori).")
            self.model_type = "mock"
            return

        # 1. Try to load GGUF model via llama-cpp-python
        if cfg.GGUF_MODEL_PATH.exists():
            try:
                logger.info(f"Memuat model GGUF dari {cfg.GGUF_MODEL_PATH} via llama-cpp-python...")
                from llama_cpp import Llama
                self.llm = Llama(
                    model_path=str(cfg.GGUF_MODEL_PATH),
                    n_ctx=cfg.N_CTX,
                    n_threads=cfg.N_THREADS,
                    verbose=False
                )
                self.model_type = "gguf"
                logger.info("Model GGUF berhasil dimuat.")
                return
            except Exception as e:
                logger.warning(f"Gagal memuat model GGUF via llama-cpp-python: {e}")
        else:
            logger.warning(f"Model GGUF tidak ditemukan di {cfg.GGUF_MODEL_PATH}.")

        # 2. Fallback to Hugging Face Transformers
        logger.info(f"Fallback memuat model HuggingFace Transformers: {cfg.MODEL_ID} ...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            device = 0 if torch.cuda.is_available() else -1
            device_map = "auto" if torch.cuda.is_available() else None
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            logger.info("Memuat AutoTokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID, token=cfg.HF_TOKEN)
            logger.info("Memuat AutoModelForCausalLM (proses ini memakan waktu dan RAM)...")
            self.llm = AutoModelForCausalLM.from_pretrained(
                cfg.MODEL_ID,
                torch_dtype=torch_dtype,
                device_map=device_map,
                token=cfg.HF_TOKEN
            )
            self.pipeline = pipeline(
                "text-generation",
                model=self.llm,
                tokenizer=self.tokenizer,
                max_new_tokens=cfg.MAX_NEW_TOKENS,
                temperature=cfg.TEMPERATURE,
                do_sample=cfg.TEMPERATURE > 0.0,
            )
            self.model_type = "transformers"
            logger.info("Model Transformers berhasil dimuat.")
        except Exception as e:
            logger.error(f"Gagal memuat model Transformers fallback: {e}")
            logger.warning("Mengaktifkan mode MOCK secara otomatis sebagai fallback darurat.")
            self.model_type = "mock"
            self.is_mock = True

    def generate(self, prompt: str, max_new_tokens: int = None, temperature: float = None) -> str:
        """Menghasilkan teks berdasarkan prompt input."""
        tokens_limit = max_new_tokens or cfg.MAX_NEW_TOKENS
        temp = temperature if temperature is not None else cfg.TEMPERATURE
        
        if self.model_type == "mock":
            return self._generate_mock_response(prompt)
            
        elif self.model_type == "gguf":
            res = self.llm(
                prompt,
                max_tokens=tokens_limit,
                temperature=temp,
                stop=["<|im_end|>", "<|im_start|>", "<|end_of_text|>", "<|eot_id|>"],
                echo=False
            )
            return res["choices"][0]["text"]
            
        elif self.model_type == "transformers":
            out = self.pipeline(
                prompt,
                max_new_tokens=tokens_limit,
                temperature=temp if temp > 0.0 else 0.01,
                do_sample=temp > 0.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_text = out[0]["generated_text"]
            if generated_text.startswith(prompt):
                reply = generated_text[len(prompt):]
            else:
                reply = generated_text
            reply = reply.split("<|im_end|>")[0].split("<|im_start|>")[0].strip()
            return reply
            
        else:
            raise RuntimeError("Model belum diinisialisasi.")

    def _generate_mock_response(self, prompt: str) -> str:
        """Menghasilkan respons tiruan berkualitas tinggi menggunakan konteks pasal yang ada dalam prompt."""
        # Cari pertanyaan
        question = "pertanyaan Anda"
        if "PERTANYAAN PENGGUNA:" in prompt:
            parts = prompt.split("PERTANYAAN PENGGUNA:")
            question = parts[-1].split("<|im_end|>")[0].strip()

        # Ekstrak pasal rujukan
        matched_pasals = []
        for line in prompt.split("\n"):
            if " - Pasal " in line:
                parts = line.split(" - ")
                uu = parts[0] if len(parts) > 0 else ""
                pasal = ""
                teks = ""
                for p in parts:
                    if p.startswith("Pasal "):
                        pasal = p
                    elif p != uu and not p.startswith("BAB") and not p.startswith("BUKU") and len(p) > 20:
                        teks = p
                if pasal:
                    matched_pasals.append((uu, pasal, teks))

        if matched_pasals:
            uu, pasal, teks = matched_pasals[0]
            teks_trimmed = teks[:250] + "..." if len(teks) > 250 else teks
            response = (
                f"Halo! Terkait pertanyaan Anda mengenai \"{question}\", berikut adalah penjelasan ringkas berdasarkan aturan hukum pidana Indonesia.\n\n"
                f"Menurut **{pasal} {uu}**, ketentuannya menyatakan:\n"
                f"*\"{teks_trimmed}\"*\n\n"
                f"Secara ringkas, pasal ini mengatur sanksi dan larangan mengenai perbuatan yang Anda tanyakan. "
                f"Untuk rincian selengkapnya, Anda dapat meninjau rujukan teks penuh pasal di panel sebelah kanan."
            )
            if len(matched_pasals) > 1:
                uu2, pasal2, teks2 = matched_pasals[1]
                teks_trimmed2 = teks2[:150] + "..." if len(teks2) > 150 else teks2
                response += f"\n\nSelain itu, **{pasal2} {uu2}** juga relevan dengan menyatakan:\n*\"{teks_trimmed2}\"*"
        else:
            response = (
                f"Halo! Mengenai pertanyaan Anda tentang \"{question}\", "
                f"berdasarkan penelusuran data hukum, topik ini diatur dalam undang-undang pidana terkait. "
                f"Silakan periksa daftar pasal rujukan di panel samping kanan untuk mempelajari detail rumusan hukumnya."
            )
            
        return response
