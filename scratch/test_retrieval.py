"""
test_retrieval.py — Script untuk menguji Hybrid Retrieval HellanodikAI secara lokal.
"""

import sys
from pathlib import Path

# Hubungkan path ke parent dir agar bisa import src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.rag.retriever import HybridRetriever

def main():
    print("=" * 80)
    print("           UJI RETRIEVAL HYBRID HELLANODIKAI (BM25 + FAISS + RERANKER)")
    print("=" * 80)
    
    try:
        retriever = HybridRetriever()
    except Exception as e:
        print(f"\nError: Gagal menginisialisasi retriever: {e}")
        print("Pastikan Anda sudah menjalankan: python -m src.rag.indexer")
        sys.exit(1)
        
    queries = [
        "Apakah berzina bisa langsung dilaporkan oleh tetangga?",
        "Bagaimana hukum pidana untuk orang yang mengaku punya kekuatan gaib/santet untuk mencelakakan orang?",
        "Berapa lama batas maksimal penahanan tersangka oleh penyidik kepolisian?",
        "Apakah korporasi atau perusahaan bisa dijatuhi hukuman denda jika melakukan kejahatan?",
    ]
    
    for q_idx, query in enumerate(queries):
        print(f"\n[Test {q_idx+1}] Query: '{query}'")
        print("-" * 70)
        
        try:
            results = retriever.retrieve(query, rerank_k=3)
            if not results:
                print("   (Tidak ditemukan pasal yang cocok)")
                continue
                
            for idx, r in enumerate(results):
                p_num = f"Pasal {r['pasal']}" if r.get('pasal') else f"Pasal {r.get('pasal_romawi')}"
                print(f"   {idx+1}. [{r['retrieval_source']}][Score: {r['retrieval_score']:.4f}] {r['uu']} - {p_num}")
                print(f"      Bab  : {r['bab']}")
                # print first 150 chars of text
                snippet = r['teks'].replace('\n', ' ').strip()
                if len(snippet) > 150:
                    snippet = snippet[:150] + "..."
                print(f"      Teks : {snippet}")
                print()
        except Exception as e:
            print(f"   Error saat retrieve: {e}")
            
    print("=" * 80)
    print("                   PENGUJIAN SELESAI")
    print("=" * 80)

if __name__ == "__main__":
    main()
