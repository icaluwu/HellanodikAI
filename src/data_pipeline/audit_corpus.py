"""
audit_corpus.py — Script audit untuk menganalisis isi pasal_corpus.json

Fungsi utama:
1. Load data/processed/pasal_corpus.json.
2. Grouping entri berdasarkan dokumen undang-undang ("uu").
3. Untuk UU No. 1 Tahun 2026 secara khusus: membagi audit menjadi dua pool:
   - Pool Struktur Utama (penomoran Romawi).
   - Pool Amandemen KUHP (penomoran Arab).
4. Untuk tiap dokumen dan pool, menganalisis:
   - Jumlah total entri pasal.
   - Range nomor pasal (min-max).
   - Daftar nomor pasal yang terduplikasi.
   - Daftar nomor pasal anomali (> 624 untuk UU No. 1 Tahun 2023).
5. Menampilkan snippet 100 karakter pertama dari pasal terduplikasi/anomali.
6. Menyimpan laporan audit lengkap ke data/processed/audit_report.txt.
7. Jika terdapat duplikat sisa, menyimpan full raw text dari maksimal 3 contoh duplikat
   ke data/processed/duplicates_diagnosis.txt untuk analisis mendalam.
"""

import sys
import json
from pathlib import Path
import re
from collections import defaultdict

# Hubungkan path ke parent dir agar bisa import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg

def roman_to_int(roman: str) -> int:
    """Konversi angka romawi ke integer untuk sorting range."""
    if not roman:
        return 0
    roman = roman.upper()
    val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    ans = 0
    for i in range(len(roman)):
        if i > 0 and val[roman[i]] > val[roman[i - 1]]:
            ans += val[roman[i]] - 2 * val[roman[i - 1]]
        else:
            ans += val[roman[i]]
    return ans

def get_sortable_key(pasal_str: str) -> tuple[bool, int, str]:
    """
    Mengubah nomor pasal (arab atau romawi) menjadi tuple sortable.
    Returns: (is_roman, numeric_value, suffix)
    """
    if not pasal_str:
        return (False, 0, "")
        
    # Cek apakah Romawi
    if re.match(r"^[IVXLCDM]+$", pasal_str, re.IGNORECASE):
        try:
            return (True, roman_to_int(pasal_str), "")
        except Exception:
            return (True, 0, pasal_str)
            
    # Cek apakah Arab alfanumerik (misal '597A' atau '12')
    match = re.match(r"^(\d+)([A-Za-z]?)$", pasal_str)
    if match:
        num = int(match.group(1))
        suffix = match.group(2)
        return (False, num, suffix)
        
    return (False, 9999, pasal_str)

def audit_pool(entries: list[dict], pool_name: str, doc_name: str, check_anomaly: bool = False) -> tuple[list[str], dict, list[dict]]:
    """
    Melakukan audit terhadap suatu pool entri pasal.
    Returns: (lines_output, duplicates_map, duplicate_items_list)
    """
    lines = []
    lines.append(f"  Pool: {pool_name}")
    lines.append(f"  Jumlah entri total: {len(entries)}")
    
    # Map nomor pasal/romawi ke entri
    pasal_map = defaultdict(list)
    for entry in entries:
        key = entry["pasal"] if entry["pasal"] is not None else entry["pasal_romawi"]
        pasal_map[key].append(entry)
        
    # Range
    keys = [k for k in pasal_map.keys() if k is not None]
    sorted_keys = sorted(keys, key=get_sortable_key)
    if sorted_keys:
        lines.append(f"  Range pasal terdeteksi: {sorted_keys[0]} s.d. {sorted_keys[-1]}")
    else:
        lines.append("  Range pasal terdeteksi: N/A")
        
    # Duplikat
    duplicates = {pasal: items for pasal, items in pasal_map.items() if len(items) > 1}
    lines.append(f"  Jumlah nomor pasal terduplikasi: {len(duplicates)}")
    if duplicates:
        lines.append("  Daftar pasal terduplikasi:")
        for pasal in sorted(duplicates.keys(), key=get_sortable_key):
            lines.append(f"    - Pasal {pasal} (muncul {len(duplicates[pasal])} kali)")
            
    # Anomali (> 624) untuk UU 1/2023
    anomalies = []
    if check_anomaly:
        for pasal_str in pasal_map.keys():
            if pasal_str is not None:
                is_rom, val, suffix = get_sortable_key(pasal_str)
                if not is_rom and val > 624:
                    anomalies.append(pasal_str)
        lines.append(f"  Jumlah pasal anomali (> 624): {len(anomalies)}")
        if anomalies:
            lines.append("  Daftar pasal anomali:")
            for pasal in sorted(anomalies, key=get_sortable_key):
                lines.append(f"    - Pasal {pasal}")
                
    # Detail Snippet
    if duplicates or anomalies:
        lines.append("  Detail Snippet untuk Duplikat/Anomali:")
        if duplicates:
            lines.append("    [DUPLIKAT]")
            for pasal in sorted(duplicates.keys(), key=get_sortable_key):
                lines.append(f"      * Pasal {pasal}:")
                for idx, item in enumerate(duplicates[pasal]):
                    text_snippet = item["teks"][:100].replace("\n", " ")
                    lines.append(f"        (Entri {idx+1}) [{item['bab'][:30]}...] : {text_snippet}...")
        if anomalies:
            lines.append("    [ANOMALI (> 624)]")
            for pasal in sorted(anomalies, key=get_sortable_key):
                lines.append(f"      * Pasal {pasal}:")
                item = pasal_map[pasal][0]
                text_snippet = item["teks"][:100].replace("\n", " ")
                lines.append(f"        [{item['bab'][:30]}...] : {text_snippet}...")
                
    lines.append("\n")
    return lines, duplicates, entries

def main():
    corpus_path = cfg.DATA_PROCESSED_DIR / "pasal_corpus.json"
    report_path = cfg.DATA_PROCESSED_DIR / "audit_report.txt"
    diagnosis_path = cfg.DATA_PROCESSED_DIR / "duplicates_diagnosis.txt"
    
    if not corpus_path.exists():
        print(f"Error: File corpus tidak ditemukan di {corpus_path}. Jalankan parse.py terlebih dahulu.")
        sys.exit(1)
        
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    # Grouping by UU
    uu_groups = defaultdict(list)
    for entry in corpus:
        uu_groups[entry["uu"]].append(entry)
        
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("               LAPORAN AUDIT CORPUS PASAL HELLANODIKAI")
    report_lines.append("=" * 80)
    report_lines.append(f"Total Entri di Seluruh Corpus: {len(corpus)}")
    report_lines.append("\n")
    
    all_duplicates_found = []
    
    for uu_name, entries in uu_groups.items():
        report_lines.append("-" * 80)
        report_lines.append(f"Dokumen: {uu_name}")
        report_lines.append("-" * 80)
        
        if uu_name == "UU No. 1 Tahun 2026":
            # Bagi menjadi dua pool untuk UU 1/2026
            romawi_entries = [e for e in entries if e["pasal_romawi"] is not None and e["pasal"] is None]
            arab_entries = [e for e in entries if e["pasal"] is not None]
            
            # Audit pool romawi
            r_lines, r_dups, _ = audit_pool(romawi_entries, "Struktur Utama UU (Pasal Romawi I-IX)", uu_name, check_anomaly=False)
            report_lines.extend(r_lines)
            for k, items in r_dups.items():
                all_duplicates_found.append((uu_name, f"Romawi {k}", items))
                
            # Audit pool arab (sub-pasal KUHP)
            a_lines, a_dups, _ = audit_pool(arab_entries, "Amandemen KUHP (Sub-pasal Arab)", uu_name, check_anomaly=False)
            report_lines.extend(a_lines)
            for k, items in a_dups.items():
                all_duplicates_found.append((uu_name, f"Arab {k}", items))
        else:
            # Audit standar
            is_anomaly_check = (uu_name == "UU No. 1 Tahun 2023")
            s_lines, s_dups, _ = audit_pool(entries, "Pasal Arab Standar", uu_name, check_anomaly=is_anomaly_check)
            report_lines.extend(s_lines)
            for k, items in s_dups.items():
                all_duplicates_found.append((uu_name, str(k), items))
                
    # Gabungkan semua baris
    report_text = "\n".join(report_text_line for report_text_line in report_lines)
    
    # Cetak ke terminal
    print(report_text)
    
    # Simpan laporan
    with open(report_path, "w", encoding="utf-8") as f_out:
        f_out.write(report_text)
    print(f"Laporan audit berhasil disimpan ke: {report_path}")
    
    # 7. Simpan full raw text duplikat jika ada (max 3 contoh)
    if all_duplicates_found:
        print(f"\nWarning: Terdeteksi {len(all_duplicates_found)} pasal terduplikasi!")
        diag_lines = []
        diag_lines.append("=" * 80)
        diag_lines.append("             DIAGNOSIS DETAIL FULL TEXT PASAL TERDUPLIKASI")
        diag_lines.append("=" * 80)
        diag_lines.append(f"Menampilkan detail teks lengkap dari maksimal 3 contoh duplikat.")
        diag_lines.append("\n")
        
        for idx, (doc, key, items) in enumerate(all_duplicates_found[:3]):
            diag_lines.append(f"Contoh {idx+1}: Dokumen '{doc}', Pasal/Key: '{key}'")
            diag_lines.append("-" * 60)
            for i_idx, item in enumerate(items):
                diag_lines.append(f"[Entri {i_idx+1}] Bab: {item['bab']}")
                diag_lines.append(f"[Teks Lengkap]:\n{item['teks']}")
                diag_lines.append("-" * 40)
            diag_lines.append("\n")
            
        diag_text = "\n".join(diag_lines)
        with open(diagnosis_path, "w", encoding="utf-8") as f_diag:
            f_diag.write(diag_text)
        print(f"Diagnosis teks lengkap duplikat ditulis ke: {diagnosis_path}")
    else:
        # Bersihkan file diagnosis lama jika tidak ada duplikat lagi
        if diagnosis_path.exists():
            diagnosis_path.unlink()
        print("\nSempurna! Tidak ada duplikat terdeteksi di seluruh dokumen.")

if __name__ == "__main__":
    main()
