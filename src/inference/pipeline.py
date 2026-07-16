"""
pipeline.py — Modul pipeline RAG end-to-end untuk HellanodikAI

Mengintegrasikan:
1. Filter keamanan out-of-scope.
2. Hybrid retrieval (BM25 + FAISS + Reranking).
3. Pembuatan prompt ChatML dengan konteks RAG.
4. Model LLM Inference (GGUF, Transformers, atau Mock).
5. Post-processing keluaran (disclaimer & normalisasi pasal).
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Hubungkan path ke parent dir agar bisa import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg

from src.rag.retriever import HybridRetriever
from src.inference.model_loader import HellanodikAIModel
from src.utils.safety import is_out_of_scope, process_response
from src.utils.prompt_templates import build_chatml_prompt, OUT_OF_SCOPE_RESPONSE

class HellanodikAIPipeline:
    def __init__(self, force_mock: bool = False):
        logger.info("Menginisialisasi HellanodikAIPipeline...")
        self.retriever = HybridRetriever()
        self.model = HellanodikAIModel(force_mock=force_mock)
        logger.info("HellanodikAIPipeline siap digunakan.")

    def query(self, user_query: str, history: list[tuple[str, str]] | None = None) -> dict:
        """
        Menangani pertanyaan pengguna secara end-to-end.
        
        Args:
            user_query: Pertanyaan hukum dari pengguna.
            history: Riwayat percakapan sebelumnya.
            
        Returns:
            dict berisi respons, pasal rujukan, dan status.
        """
        logger.info(f"Menerima query: '{user_query}'")
        
        # 1. Deteksi Out-of-Scope
        if is_out_of_scope(user_query):
            logger.info("Query terdeteksi di luar ruang lingkup (OUT-OF-SCOPE).")
            return {
                "text": OUT_OF_SCOPE_RESPONSE,
                "sources": [],
                "out_of_scope": True,
                "model_type": self.model.model_type,
                "has_pasal_ref": False,
                "pasal_refs": []
            }

        # 2. Ambil artikel pasal relevan lewat Hybrid RAG
        try:
            sources = self.retriever.retrieve(user_query)
        except Exception as e:
            logger.error(f"Gagal melakukan retrieval: {e}")
            sources = []

        # 3. Format RAG Context
        context_parts = []
        for r in sources:
            p_num = f"Pasal {r['pasal']}" if r.get('pasal') else f"Pasal {r.get('pasal_romawi')}"
            context_parts.append(f"{r['uu']} - {r['bab']} - {p_num} - {r['teks']}")
            
        context_str = "\n\n".join(context_parts)

        # 4. Bangun prompt ChatML
        prompt = build_chatml_prompt(user_message=user_query, context=context_str, history=history)

        # 5. Model Inference
        try:
            raw_response = self.model.generate(prompt)
        except Exception as e:
            logger.error(f"Gagal generate respons dari model: {e}")
            raw_response = "Maaf, terjadi kesalahan teknis saat memproses respons Anda. Silakan coba kembali."

        # 6. Post-processing (Disclaimer & Normalisasi Referensi)
        processed = process_response(raw_response)
        
        return {
            "text": processed["text"],
            "sources": sources,
            "out_of_scope": False,
            "model_type": self.model.model_type,
            "has_pasal_ref": processed["has_pasal_ref"],
            "pasal_refs": processed["pasal_refs"]
        }
