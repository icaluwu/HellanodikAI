"""
ui_components.py — Desain antarmuka premium dan komponen Gradio untuk HellanodikAI
"""

def format_references_html(sources: list[dict]) -> str:
    """Memformat daftar pasal hukum hasil retrieval RAG menjadi kartu HTML yang premium."""
    if not sources:
        return """
        <div style='text-align: center; padding: 60px 20px; color: #6B7280; border: 1px dashed rgba(255,255,255,0.08); border-radius: 12px; background: rgba(17, 24, 39, 0.2);'>
            <svg style="width: 48px; height: 48px; margin: 0 auto 16px auto; opacity: 0.4;" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
            <p style='font-size: 1.1rem; font-weight: 500; margin-bottom: 6px; color: #E5E7EB;'>Belum Ada Rujukan Hukum</p>
            <p style='font-size: 0.85rem; max-width: 280px; margin: 0 auto;'>Ajukan pertanyaan Anda untuk mencari dan menampilkan pasal KUHP yang relevan secara otomatis.</p>
        </div>
        """
    
    html = '<div class="scrollable-references">'
    for idx, r in enumerate(sources):
        p_num = f"Pasal {r['pasal']}" if r.get('pasal') else f"Pasal {r.get('pasal_romawi')}"
        score = r.get("retrieval_score", 0.0)
        source = r.get("retrieval_source", "RAG")
        
        # Penentuan style badge berdasarkan metode retrieval
        if source == "Hybrid":
            badge_style = "background: rgba(16, 185, 129, 0.12); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.2);"
        elif source == "FAISS":
            badge_style = "background: rgba(59, 130, 246, 0.12); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.2);"
        else:  # BM25
            badge_style = "background: rgba(245, 158, 11, 0.12); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.2);"

        # Label status jika pasal telah diubah oleh UU 1/2026
        status_badge = ""
        if r.get("status") == "diubah":
            status_badge = '<span style="background: rgba(239, 68, 68, 0.12); color: #FCA5A5; border: 1px solid rgba(239, 68, 68, 0.2); padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; margin-right: 8px;">DIUBAH</span>'

        html += f"""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 8px;">
                <div>
                    {status_badge}
                    <span style="font-weight: 700; font-size: 1.1rem; color: #F3F4F6;">{p_num}</span>
                </div>
                <span style="padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; {badge_style}">
                    {source} | Rerank: {score:.3f}
                </span>
            </div>
            <div style="font-size: 0.8rem; color: #10B981; margin-bottom: 10px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">
                {r['uu']} • {r['bab'][:60]}...
            </div>
            <div style="font-size: 0.92rem; color: #D1D5DB; line-height: 1.6; white-space: pre-line; background: rgba(0,0,0,0.15); padding: 10px; border-radius: 6px;">
                {r['teks']}
            </div>
        </div>
        """
    html += '</div>'
    return html

HEADER_HTML = """
<div class="header-container">
    <h1 class="header-title">🏛️ HellanodikAI</h1>
    <p class="header-subtitle">Asisten Cerdas Literasi Hukum Pidana Indonesia (KUHP 2023 & UU Penyesuaian)</p>
</div>
"""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Terapkan font premium dan background global */
body, .gradio-container {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #0A0E17 !important;
    color: #E5E7EB !important;
}

/* Glassmorphic Panel Cards untuk Pasal Rujukan */
.glass-card {
    background: rgba(17, 24, 39, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(12px) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35) !important;
    margin-bottom: 14px !important;
    padding: 16px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.glass-card:hover {
    border-color: rgba(16, 185, 129, 0.3) !important;
    box-shadow: 0 8px 32px 0 rgba(16, 185, 129, 0.08) !important;
    transform: translateY(-2px) !important;
}

/* Kustomisasi kontainer chatbot */
.chatbot-container {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background: rgba(17, 24, 39, 0.4) !important;
    border-radius: 12px !important;
}

/* Efek tombol modern */
.accent-btn {
    background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.accent-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 0 16px rgba(16, 185, 129, 0.45) !important;
}

/* Styling sidebar panel rujukan */
.scrollable-references {
    max-height: 580px !important;
    overflow-y: auto !important;
    padding-right: 6px !important;
}

/* Desain Header Utama */
.header-container {
    text-align: center;
    padding: 24px 0;
    margin-bottom: 24px;
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.12) 0%, rgba(10, 14, 23, 0) 100%);
    border-bottom: 1px solid rgba(16, 185, 129, 0.12);
    border-radius: 0 0 24px 24px;
}

.header-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #34D399 0%, #10B981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}

.header-subtitle {
    color: #9CA3AF;
    font-size: 1.15rem;
    font-weight: 400;
}

/* Styling custom scrollbar */
.scrollable-references::-webkit-scrollbar {
    width: 6px;
}
.scrollable-references::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 3px;
}
.scrollable-references::-webkit-scrollbar-thumb {
    background: rgba(16, 185, 129, 0.3);
    border-radius: 3px;
}
.scrollable-references::-webkit-scrollbar-thumb:hover {
    background: rgba(16, 185, 129, 0.5);
}
"""
