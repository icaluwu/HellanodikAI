"""
test_pipeline.py — Script pengujian RAG end-to-end pipeline HellanodikAI secara lokal.
"""

import sys
from pathlib import Path
import os

# Konfigurasi encoding stdout ke UTF-8 untuk mendukung karakter emoji di terminal Windows
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Hubungkan path ke parent dir agar bisa import src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.pipeline import HellanodikAIPipeline

def main():
    print("=" * 80)
    print("              UJI END-TO-END PIPELINE HELLANODIKAI")
    print("=" * 80)
    
    # Paksa mode MOCK untuk pengujian lokal cepat dan ringan
    os.environ["MOCK_LLM"] = "true"
    
    try:
        pipeline = HellanodikAIPipeline(force_mock=True)
    except Exception as e:
        print(f"\nError: Gagal menginisialisasi pipeline: {e}")
        sys.exit(1)
        
    test_queries = [
        "Apakah berzina bisa langsung dilaporkan oleh tetangga?",
        "Bagaimana cara memenangkan persidangan di pengadilan negeri?", # out of scope
        "Apakah korporasi atau perusahaan bisa dijatuhi hukuman denda jika melakukan kejahatan?"
    ]
    
    for q_idx, q in enumerate(test_queries):
        print(f"\n[Test {q_idx+1}] Query: '{q}'")
        print("-" * 70)
        
        try:
            res = pipeline.query(q)
            print(f"   Model Type     : {res.get('model_type')}")
            print(f"   Out of Scope   : {res.get('out_of_scope')}")
            print(f"   Has Pasal Ref  : {res.get('has_pasal_ref')}")
            print(f"   Pasal Refs     : {res.get('pasal_refs')}")
            print(f"   Respons Teks   :\n{res.get('text')}")
            print(f"   Retrieved Pasals:")
            for s_idx, s in enumerate(res.get('sources', [])):
                p_num = f"Pasal {s['pasal']}" if s.get('pasal') else f"Pasal {s.get('pasal_romawi')}"
                print(f"     * {s_idx+1}. {s['uu']} - {p_num} [Score: {s['retrieval_score']:.4f}]")
        except Exception as e:
            print(f"   Error saat memproses query: {e}")
            
    print("=" * 80)
    print("                   PENGUJIAN PIPELINE SELESAI")
    print("=" * 80)

if __name__ == "__main__":
    main()
