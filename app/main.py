"""
main.py — Titik masuk utama aplikasi Gradio HellanodikAI

Menyediakan antarmuka web interaktif premium untuk berinteraksi dengan RAG pipeline.
Jalankan dengan: python app/main.py
"""

import sys
import os
from pathlib import Path
import gradio as gr
from loguru import logger

# Hubungkan path ke parent dir agar bisa import src dan config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

from src.inference.pipeline import HellanodikAIPipeline
from app.ui_components import HEADER_HTML, CUSTOM_CSS, format_references_html

# Inisialisasi pipeline global
pipeline = HellanodikAIPipeline()

def respond(message: str, chat_history: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]], str]:
    """Menangani pengiriman pertanyaan dari pengguna dan memperbarui status UI."""
    if not message.strip():
        return "", chat_history, format_references_html([])
    
    logger.info(f"Pertanyaan pengguna di UI: '{message}'")
    
    # Kirim query ke pipeline
    res = pipeline.query(message, history=chat_history)
    bot_message = res["text"]
    sources = res["sources"]
    
    # Update chat history
    chat_history.append((message, bot_message))
    
    # Render ulang referensi menjadi HTML
    ref_html = format_references_html(sources)
    
    return "", chat_history, ref_html

def main():
    """Method utama untuk membangun dan menjalankan aplikasi Gradio."""
    logger.info("Membangun antarmuka Gradio...")
    
    # Membuat blok antarmuka dengan judul
    with gr.Blocks(title="HellanodikAI") as demo:
        # Header HTML Kustom
        gr.HTML(HEADER_HTML)
        
        with gr.Row():
            # Kolom Kiri: Chatbot & Pertanyaan
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Percakapan HellanodikAI", 
                    elem_classes=["chatbot-container"],
                    height=450,
                    show_label=False
                )
                
                with gr.Row():
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="Tanyakan tentang pasal KUHP baru di sini (misal: perzinaan, pencurian, denda korporasi)...",
                        scale=9
                    )
                    submit_btn = gr.Button("Kirim", elem_classes=["accent-btn"], scale=1)
                
                # Contoh pertanyaan cepat
                examples = [
                    "Apakah berzina bisa langsung dilaporkan oleh tetangga?",
                    "Bagaimana hukum pidana untuk orang yang mengaku punya kekuatan gaib/santet?",
                    "Berapa lama batas maksimal penahanan tersangka oleh penyidik kepolisian?",
                    "Apakah korporasi atau perusahaan bisa dijatuhi hukuman denda jika melakukan kejahatan?"
                ]
                
                gr.Examples(
                    examples=examples,
                    inputs=txt,
                    label="Contoh Pertanyaan Cepat"
                )

            # Kolom Kanan: Rujukan Hukum RAG
            with gr.Column(scale=2):
                gr.Markdown("### 📖 Rujukan Hukum / Konteks RAG")
                references_html = gr.HTML(format_references_html([]))
                
        # Integrasi Event Listener
        txt.submit(respond, [txt, chatbot], [txt, chatbot, references_html])
        submit_btn.click(respond, [txt, chatbot], [txt, chatbot, references_html])
        
        # Disclaimer & Footer Info
        gr.Markdown(
            f"<div style='text-align: center; margin-top: 24px; font-size: 0.82rem; color: #6B7280;'>"
            f"HellanodikAI dibangun dengan teknologi Llama. Built with Llama. "
            f"Status Mesin LLM: <span style='color: #10B981; font-weight: 600;'>{pipeline.model.model_type.upper()}</span>"
            f"</div>"
        )
        
    logger.info(f"Menjalankan server Gradio pada port {cfg.GRADIO_PORT}...")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=cfg.GRADIO_PORT, 
        share=cfg.GRADIO_SHARE,
        theme=gr.themes.Default(primary_hue="emerald", neutral_hue="slate"),
        css=CUSTOM_CSS
    )

if __name__ == "__main__":
    main()
