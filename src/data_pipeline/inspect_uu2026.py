"""
inspect_uu2026.py — Script inspeksi untuk menganalisis hasil parsing UU No. 1 Tahun 2026

Fungsi utama:
1. Load data/processed/pasal_corpus.json, filter entri uu == "UU No. 1 Tahun 2026".
2. Print urut semua nilai pasal_romawi yang terdeteksi.
3. Print panjang teks (jumlah karakter) untuk entri dengan pasal_romawi == "VII"
   (atau yang mereferensikan sub-pasal di bawahnya).
4. Print daftar lengkap nomor pasal arab ("mengubah_pasal") urut numerik.
"""

import sys
import json
from pathlib import Path
import re

# Hubungkan path ke parent dir agar bisa import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg

def roman_to_int(roman: str) -> int:
    roman = roman.upper()
    val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    ans = 0
    for i in range(len(roman)):
        if i > 0 and val[roman[i]] > val[roman[i - 1]]:
            ans += val[roman[i]] - 2 * val[roman[i - 1]]
        else:
            ans += val[roman[i]]
    return ans

def get_sortable_key(pasal_str: str) -> int:
    match = re.match(r"^(\d+)", pasal_str)
    if match:
        return int(match.group(1))
    return 9999

def main():
    corpus_path = cfg.DATA_PROCESSED_DIR / "pasal_corpus.json"
    if not corpus_path.exists():
        print(f"Error: File corpus tidak ditemukan di {corpus_path}.")
        sys.exit(1)
        
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    # Filter UU No. 1 Tahun 2026
    entries = [e for e in corpus if e["uu"] == "UU No. 1 Tahun 2026"]
    
    print("=" * 80)
    print("        INSPEKSI DATA PARSING UU NO. 1 TAHUN 2026 (PENYESUAIAN PIDANA)")
    print("=" * 80)
    print(f"Total entri terdeteksi: {len(entries)}")
    print()
    
    # 1. Tampilkan semua pasal_romawi yang terdeteksi (dan diurutkan)
    romawi_entries = [e for e in entries if e["pasal_romawi"] is not None]
    romawi_list = sorted(list(set(e["pasal_romawi"] for e in romawi_entries)), key=roman_to_int)
    print("1. Daftar Nilai 'pasal_romawi' yang Terdeteksi (Urut):")
    print(f"   {romawi_list}")
    print()
    
    # 2. Detail entri pasal Romawi (termasuk panjang teks)
    print("2. Detail Panjang Teks Entri Romawi:")
    for r_val in romawi_list:
        matching = [e for e in romawi_entries if e["pasal_romawi"] == r_val]
        for idx, m in enumerate(matching):
            print(f"   - Pasal Romawi {r_val} (Entri {idx+1}): {len(m['teks'])} karakter")
            # print first 60 chars
            snippet = m['teks'][:60].replace('\n', ' ')
            print(f"     Snippet: '{snippet}...'")
    print()
    
    # 3. Tampilkan detail untuk sub-pasal di bawah Pasal VII
    arab_entries = [e for e in entries if e["pasal"] is not None]
    print(f"3. Detail Panjang Teks Sub-Pasal Arab (di bawah Pasal VII) - Total {len(arab_entries)} entri:")
    # hitung total karakter sub-pasal arab
    total_len_arab = sum(len(e["teks"]) for e in arab_entries)
    print(f"   - Total panjang seluruh teks sub-pasal Arab: {total_len_arab} karakter")
    
    # Tampilkan 5 sub-pasal arab terpanjang sebagai sample
    sorted_by_len = sorted(arab_entries, key=lambda x: len(x["teks"]), reverse=True)
    print("   - 5 sub-pasal arab terpanjang:")
    for idx, e in enumerate(sorted_by_len[:5]):
        print(f"     * Pasal {e['pasal']}: {len(e['teks'])} karakter")
        
    print()
    
    # 4. Tampilkan daftar lengkap 49 nomor pasal arab (mengubah_pasal) urut numerik
    arab_pasal_nums = sorted([e["mengubah_pasal"] for e in arab_entries if e["mengubah_pasal"] is not None], key=get_sortable_key)
    print("4. Daftar Lengkap Nomor Pasal Arab ('mengubah_pasal') Terdeteksi (Urut Numerik):")
    print(f"   Total terdeteksi: {len(arab_pasal_nums)}")
    # Print per baris kelompok agar rapi
    chunk_size = 10
    for i in range(0, len(arab_pasal_nums), chunk_size):
        chunk = arab_pasal_nums[i:i+chunk_size]
        print(f"   {', '.join(chunk)}")
        
if __name__ == "__main__":
    main()
