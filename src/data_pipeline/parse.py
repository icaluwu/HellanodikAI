"""
parse.py — Pipeline ekstraksi dan parsing dokumen hukum primer

Fungsi utama:
1. Membaca PDF hukum dari dataset_primer/ menggunakan PyMuPDF (fitz).
2. Menyimpan teks mentah per halaman ke data/raw/{nama_dokumen}.txt untuk debugging.
3. Menghilangkan boilerplate (header/footer) per halaman menggunakan frequency analysis
   dan SequenceMatcher (similarity > 0.8) pada baris yang muncul di > 70% halaman.
4. Menggabungkan semua halaman batang tubuh menjadi satu teks kontinu (full_text)
   untuk meniadakan masalah batas halaman (page boundary).
5. Menjalankan deteksi regex "Pasal N" sekali pada teks kontinu tersebut.
6. Menerapkan post-processing safety pass: menggabungkan chunk pasal berurutan yang
   memiliki nomor pasal sama jika chunk pertama berukuran < 30 karakter.
7. Menangani UU No. 1 Tahun 2026 secara khusus dengan memisahkan penomoran Romawi (I-IX)
   ke field 'pasal_romawi', sedangkan nomor pasal KUHP arab ke field 'pasal'.
8. Post-processing untuk meng-update status pasal KUHP 2023 lama yang diubah oleh UU 1/2026
   menjadi 'diubah'.
9. Menyimpan corpus akhir ke data/processed/pasal_corpus.json.
"""

import sys
from pathlib import Path
import re
import json
import difflib
from collections import defaultdict
from loguru import logger
import fitz  # PyMuPDF

# Hubungkan path ke parent dir agar bisa import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg

# ---------------------------------------------------------------------------
# Metadata Halaman untuk Batang Tubuh
# ---------------------------------------------------------------------------
DOC_METADATA = {
    "UU Nomor 1 Tahun 2023.pdf": {
        "short_name": "UU No. 1 Tahun 2023",
        "batang_tubuh_end_page": 229,  # halaman fisik 1 s.d. 229
    },
    "UU Nomor 1 Tahun 2026.pdf": {
        "short_name": "UU No. 1 Tahun 2026",
        "batang_tubuh_end_page": 51,   # halaman fisik 1 s.d. 51
    },
    "UU Nomor 20 Tahun 2025.pdf": {
        "short_name": "UU No. 20 Tahun 2025",
        "batang_tubuh_end_page": 184,  # halaman fisik 1 s.d. 184
    }
}


# ---------------------------------------------------------------------------
# Utility & Normalization Functions
# ---------------------------------------------------------------------------

def roman_to_int(roman: str) -> int:
    """Konversi angka romawi ke integer."""
    roman = roman.upper()
    val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    ans = 0
    for i in range(len(roman)):
        if i > 0 and val[roman[i]] > val[roman[i - 1]]:
            ans += val[roman[i]] - 2 * val[roman[i - 1]]
        else:
            ans += val[roman[i]]
    return ans

def is_valid_pasal_header(line: str) -> tuple[bool, str]:
    """
    Memeriksa dan menormalisasi baris judul Pasal.
    Menyingkirkan rujukan tengah kalimat dan running footer halaman secara presisi.
    Returns: (is_valid, normalized_pasal_num)
    """
    s = line.strip()
    
    # 1. Abaikan baris kelanjutan halaman yang diakhiri titik-titik
    if re.search(r"\.\s*\.\s*\.$", s) or s.endswith("...") or s.endswith(". ."):
        return False, ""
        
    # 2. Cocokkan kata pembuka Pasal dengan varian OCR-nya
    match = re.match(r"^Pasa[l7iI1\]\|T]\s*(.*)$", s, re.IGNORECASE)
    if not match:
        return False, ""
        
    num_part_raw = match.group(1).strip()
    if not num_part_raw:
        return False, ""
        
    # 3. Jika terlalu panjang dan mengandung spasi, kemungkinan rujukan dalam kalimat
    if len(num_part_raw) > 7 and " " in num_part_raw:
        return False, ""
        
    # 4. Hapus spasi internal
    num_part = re.sub(r"\s+", "", num_part_raw)
    
    # 5. Deteksi angka Romawi terlebih dahulu sebelum normalisasi Arab
    # (Mencegah huruf 'I' pada romawi diubah menjadi '1')
    if re.match(r"^[IVXLCDM]+$", num_part, re.IGNORECASE):
        return True, num_part.upper()
        
    # 6. Jalankan normalisasi OCR Arab jika bukan Romawi
    num_part_arab = num_part.replace('O', '0').replace('o', '0')
    num_part_arab = num_part_arab.replace('I', '1').replace('l', '1').replace('|', '1')
    num_part_arab = num_part_arab.replace('T', '7')
    
    # Validasi format Arab alfanumerik (misal 12, 12A)
    is_arab = re.match(r"^(\d+[A-Za-z]?)$", num_part_arab)
    if is_arab:
        return True, is_arab.group(1)
        
    return False, ""


# ---------------------------------------------------------------------------
# Frequency Analysis Boilerplate Detection
# ---------------------------------------------------------------------------

def identify_boilerplate(pages_lines: list[list[str]]) -> set[str]:
    """
    Mendeteksi baris boilerplate (letterhead/running header/footer)
    berdasarkan frekuensi kemunculan fuzzy/identik di >70% halaman.
    """
    line_occurrences = defaultdict(set)
    for p_idx, page in enumerate(pages_lines):
        for line in page:
            line_str = line.strip()
            if line_str:
                line_occurrences[line_str].add(p_idx)
                
    num_pages = len(pages_lines)
    threshold = 0.70 * num_pages
    boilerplate = set()
    
    # 1. Cari exact match candidates
    exact_candidates = []
    for line_str, pages in line_occurrences.items():
        if len(pages) >= threshold:
            boilerplate.add(line_str)
        elif len(pages) >= 0.20 * num_pages:
            exact_candidates.append(line_str)
            
    # 2. Cari fuzzy match candidates (SequenceMatcher > 0.8)
    for cand in exact_candidates:
        if cand in boilerplate:
            continue
        matching_pages = 0
        for p_idx, page in enumerate(pages_lines):
            matched_page = False
            for line in page:
                line_str = line.strip()
                if not line_str:
                    continue
                if abs(len(cand) - len(line_str)) > 5:
                    continue
                if cand == line_str:
                    matched_page = True
                    break
                ratio = difflib.SequenceMatcher(None, cand, line_str).ratio()
                if ratio >= 0.8:
                    matched_page = True
                    break
            if matched_page:
                matching_pages += 1
        if matching_pages >= threshold:
            boilerplate.add(cand)
            
    return boilerplate


def clean_pages(pages_lines: list[list[str]], boilerplate: set[str]) -> list[str]:
    """
    Membersihkan baris-baris dokumen dari boilerplate terdeteksi
    serta penomoran halaman dan security codes dengan regex.
    """
    cleaned_all_lines = []
    for page in pages_lines:
        for line in page:
            line_str = line.strip()
            if not line_str:
                continue
                
            # Filter regex untuk nomor halaman dan kode sekuritas cetakan
            if re.match(r"^-\s*\d+\s*-$", line_str):
                continue
            if re.match(r"^SK\s+No.*$", line_str, re.IGNORECASE):
                continue
                
            # Cek terhadap boilerplate
            is_boilerplate = False
            for bp in boilerplate:
                if abs(len(bp) - len(line_str)) > 5:
                    continue
                if bp == line_str:
                    is_boilerplate = True
                    break
                ratio = difflib.SequenceMatcher(None, bp, line_str).ratio()
                if ratio >= 0.8:
                    is_boilerplate = True
                    break
                    
            if not is_boilerplate:
                cleaned_all_lines.append(line)
    return cleaned_all_lines


# ---------------------------------------------------------------------------
# Structural Parsing & Body Processing
# ---------------------------------------------------------------------------

def process_body_lines(body_text: str, state: dict) -> str:
    """
    Mem-parse baris isi pasal, mengupdate state struktur Buku/BAB/Bagian/Paragraf,
    dan mengembalikan teks pasal yang bersih (tanpa baris struktur).
    """
    lines = body_text.split("\n")
    article_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        # Buku
        if re.match(r"^BUKU\s+([A-Z]+)$", line, re.IGNORECASE):
            title_parts = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not any(re.match(p, lines[i].strip(), re.IGNORECASE) for p in [r"^BAB\b", r"^Bag[ia1l]an\b", r"^Paragraf\b", r"^Pasa[l7iI1\]\|T]\b"]):
                title_parts.append(lines[i].strip())
                i += 1
            state["buku"] = " ".join(title_parts)
            state["bab"] = ""
            state["bagian"] = ""
            state["paragraf"] = ""
            continue
            
        # BAB
        if re.match(r"^BAB\s+([IVXLCDM\d]+)$", line, re.IGNORECASE):
            title_parts = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not any(re.match(p, lines[i].strip(), re.IGNORECASE) for p in [r"^BUKU\b", r"^Bag[ia1l]an\b", r"^Paragraf\b", r"^Pasa[l7iI1\]\|T]\b"]):
                title_parts.append(lines[i].strip())
                i += 1
            state["bab"] = " ".join(title_parts)
            state["bagian"] = ""
            state["paragraf"] = ""
            continue
            
        # Bagian
        if re.match(r"^Bag[ia1l]an\s+([A-Za-z]+)$", line, re.IGNORECASE):
            title_parts = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not any(re.match(p, lines[i].strip(), re.IGNORECASE) for p in [r"^BUKU\b", r"^BAB\b", r"^Paragraf\b", r"^Pasa[l7iI1\]\|T]\b"]):
                title_parts.append(lines[i].strip())
                i += 1
            state["bagian"] = " ".join(title_parts)
            state["paragraf"] = ""
            continue
            
        # Paragraf
        if re.match(r"^Paragraf\s+(\d+)$", line, re.IGNORECASE):
            title_parts = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not any(re.match(p, lines[i].strip(), re.IGNORECASE) for p in [r"^BUKU\b", r"^BAB\b", r"^Bag[ia1l]an\b", r"^Pasa[l7iI1\]\|T]\b"]):
                title_parts.append(lines[i].strip())
                i += 1
            state["paragraf"] = " ".join(title_parts)
            continue
            
        article_lines.append(lines[i])
        i += 1
        
    return "\n".join(article_lines).strip()


# ---------------------------------------------------------------------------
# Safety Merger Pass
# ---------------------------------------------------------------------------

def merge_split_pasals(pasals: list[dict]) -> list[dict]:
    """
    Menggabungkan entri berurutan yang memiliki nomor pasal yang sama
    jika entri pertama berukuran sangat pendek (<30 karakter tanpa spasi).
    Mencegah pasal terpotong akibat page boundary.
    """
    if not pasals:
        return []
    merged = [pasals[0]]
    for current in pasals[1:]:
        last = merged[-1]
        
        # Cek kesamaan dokumen, nomor pasal arab, dan nomor pasal romawi
        same_article = (
            last["uu"] == current["uu"] and
            last["pasal"] == current["pasal"] and
            last["pasal_romawi"] == current["pasal_romawi"]
        )
        if same_article:
            last_clean_len = len(re.sub(r"\s+", "", last["teks"]))
            if last_clean_len < 30:
                logger.info(
                    f"Menggabungkan pasal terpotong di {last['uu']}: "
                    f"pasal={last['pasal']}, romawi={last['pasal_romawi']} (panjang sblmnya: {last_clean_len} char)"
                )
                last["teks"] = (last["teks"] + "\n" + current["teks"]).strip()
                continue
        merged.append(current)
    return merged


# ---------------------------------------------------------------------------
# Parsing Standard Law (UU 1/2023 & UU 20/2025)
# ---------------------------------------------------------------------------

def parse_standard_law(pdf_path: Path, short_name: str, end_page: int) -> list[dict]:
    """
    Mengekstrak dan mem-parse undang-undang dengan penomoran pasal arab standar.
    """
    logger.info(f"Memproses {short_name}...")
    doc = fitz.open(pdf_path)
    
    # 1. Ekstrak baris per halaman
    pages_lines = []
    for p_idx in range(min(end_page, len(doc))):
        pages_lines.append(doc[p_idx].get_text().split("\n"))
        
    # 2. Identifikasi & bersihkan boilerplate (frequency analysis)
    boilerplate = identify_boilerplate(pages_lines)
    logger.info(f"Berhasil mengidentifikasi {len(boilerplate)} baris boilerplate.")
    
    cleaned_lines = clean_pages(pages_lines, boilerplate)
    full_text = "\n".join(cleaned_lines)
    
    # 3. Cari semua header pasal valid
    headers = []
    for match in re.finditer(r"(?:^|\n)([^\n]*Pasa[l7iI1\]\|T][^\n]*)", full_text, re.IGNORECASE):
        line_text = match.group(1)
        is_val, pasal_num = is_valid_pasal_header(line_text)
        if is_val:
            # Jika berupa romawi, konversikan ke arab untuk konsistensi dokumen standar
            if re.match(r"^[IVXLCDM]+$", pasal_num, re.IGNORECASE):
                pasal_num = str(roman_to_int(pasal_num))
            headers.append({
                "pasal": pasal_num,
                "start": match.start(1),
                "end": match.end(1),
                "raw": line_text
            })
            
    logger.info(f"Ditemukan {len(headers)} judul pasal valid.")
    
    # 4. Segmentasi teks kontinu berdasarkan posisi header
    pasals = []
    state = {"buku": "", "bab": "", "bagian": "", "paragraf": ""}
    
    if headers:
        # Preamble sebelum pasal pertama
        preamble = full_text[:headers[0]["start"]]
        process_body_lines(preamble, state)
        
        for idx, h in enumerate(headers):
            start_body = h["end"]
            end_body = headers[idx + 1]["start"] if idx + 1 < len(headers) else len(full_text)
            body_text = full_text[start_body:end_body]
            
            clean_text = process_body_lines(body_text, state)
            
            context_bab = " - ".join(filter(None, [state["buku"], state["bab"], state["bagian"], state["paragraf"]]))
            pasals.append({
                "uu": short_name,
                "bab": context_bab or "Ketentuan Umum",
                "pasal": h["pasal"],
                "pasal_romawi": None,
                "teks": clean_text,
                "mengubah_pasal": None,
                "status": "berlaku"
            })
            
    return pasals


# ---------------------------------------------------------------------------
# Parsing UU No. 1 Tahun 2026 (Romawi + Arab Sub-articles)
# ---------------------------------------------------------------------------

def parse_uu1_2026(pdf_path: Path, short_name: str, end_page: int) -> list[dict]:
    """
    Parser khusus untuk UU No. 1 Tahun 2026:
    - Pasal utama menggunakan angka Romawi (Pasal I s.d. Pasal IX) -> disimpan dengan pasal=None, pasal_romawi="I-IX".
    - Sub-pasal amandemen KUHP (di dalam Pasal VII) menggunakan angka Arab -> disimpan dengan pasal="3", pasal_romawi=None.
    """
    logger.info(f"Memproses {short_name}...")
    doc = fitz.open(pdf_path)
    
    pages_lines = []
    for p_idx in range(min(end_page, len(doc))):
        pages_lines.append(doc[p_idx].get_text().split("\n"))
        
    boilerplate = identify_boilerplate(pages_lines)
    logger.info(f"Berhasil mengidentifikasi {len(boilerplate)} baris boilerplate.")
    
    cleaned_lines = clean_pages(pages_lines, boilerplate)
    full_text = "\n".join(cleaned_lines)
    
    # 1. Cari judul pasal Romawi utama (I s.d. IX)
    romawi_headers = []
    for match in re.finditer(r"(?:^|\n)([^\n]*Pasa[l7iI1\]\|T][^\n]*)", full_text, re.IGNORECASE):
        line_text = match.group(1)
        is_val, pasal_num = is_valid_pasal_header(line_text)
        if is_val and re.match(r"^[IVXLCDM]+$", pasal_num):  # Hanya romawi
            romawi_headers.append({
                "pasal_romawi": pasal_num,
                "start": match.start(1),
                "end": match.end(1),
                "raw": line_text
            })
            
    logger.info(f"Ditemukan {len(romawi_headers)} pasal utama Romawi (I-IX) valid.")
    
    pasals = []
    state = {"buku": "", "bab": "", "bagian": "", "paragraf": ""}
    
    if romawi_headers:
        # Preamble sebelum Pasal I
        preamble = full_text[:romawi_headers[0]["start"]]
        process_body_lines(preamble, state)
        
        for idx, rh in enumerate(romawi_headers):
            rom_num = rh["pasal_romawi"]
            start_body = rh["end"]
            end_body = romawi_headers[idx + 1]["start"] if idx + 1 < len(romawi_headers) else len(full_text)
            body_text = full_text[start_body:end_body]
            
            clean_text = process_body_lines(body_text, state)
            context_bab = " - ".join(filter(None, [state["buku"], state["bab"], state["bagian"], state["paragraf"]]))
            
            # Khusus Pasal VII (Bab III): Pecah menjadi sub-pasal angka arab
            if rom_num == "VII":
                logger.info("Memproses sub-pasal perubahan KUHP di dalam Pasal VII...")
                
                # Cari header sub-pasal arab di dalam clean_text Pasal VII
                sub_headers = []
                for sub_match in re.finditer(r"(?:^|\n)([^\n]*Pasa[l7iI1\]\|T][^\n]*)", clean_text, re.IGNORECASE):
                    sub_line = sub_match.group(1)
                    is_sub_val, sub_pasal_num = is_valid_pasal_header(sub_line)
                    if is_sub_val and not re.match(r"^[IVXLCDM]+$", sub_pasal_num):  # Hanya arab
                        sub_headers.append({
                            "pasal": sub_pasal_num,
                            "start": sub_match.start(1),
                            "end": sub_match.end(1),
                            "raw": sub_line
                        })
                
                logger.info(f"Mengekstrak {len(sub_headers)} sub-pasal dari Pasal VII.")
                
                sub_state = {"buku": "", "bab": "", "bagian": "", "paragraf": ""}
                # Preamble sub-pasal (biasanya teks pengantar Pasal VII)
                if sub_headers:
                    sub_preamble = clean_text[:sub_headers[0]["start"]]
                    process_body_lines(sub_preamble, sub_state)
                    
                    for s_idx, sh in enumerate(sub_headers):
                        s_start = sh["end"]
                        s_end = sub_headers[s_idx + 1]["start"] if s_idx + 1 < len(sub_headers) else len(clean_text)
                        sub_body = clean_text[s_start:s_end]
                        
                        clean_sub_text = process_body_lines(sub_body, sub_state)
                        
                        pasals.append({
                            "uu": short_name,
                            "bab": "BAB III - PERUBAHAN DALAM UNDANG-UNDANG NOMOR 1 TAHUN 2023 TENTANG KITAB UNDANG-UNDANG HUKUM PIDANA",
                            "pasal": sh["pasal"],
                            "pasal_romawi": None,
                            "teks": clean_sub_text,
                            "mengubah_pasal": sh["pasal"],  # referensi ke pasal KUHP asal yang diubah
                            "status": "berlaku"
                        })
            else:
                # Pasal utama non-VII disimpan langsung
                pasals.append({
                    "uu": short_name,
                    "bab": context_bab or "Ketentuan Umum",
                    "pasal": None,
                    "pasal_romawi": rom_num,
                    "teks": clean_text,
                    "mengubah_pasal": None,
                    "status": "berlaku"
                })
                
    return pasals


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------

def main():
    logger.info("=== HELLANODIKAI DATA PIPELINE - START ===")
    
    all_parsed_pasals = []
    
    # Loop untuk masing-masing PDF primer
    for pdf_name, meta in DOC_METADATA.items():
        pdf_path = cfg.DATASET_PRIMER_DIR / pdf_name
        if not pdf_path.exists():
            logger.error(f"File PDF tidak ditemukan: {pdf_path}")
            sys.exit(1)
            
        short_name = meta["short_name"]
        end_page = meta["batang_tubuh_end_page"]
        
        # 1. Simpan teks mentah per halaman untuk debugging
        raw_txt_path = cfg.DATA_RAW_DIR / f"{pdf_path.stem}.txt"
        logger.info(f"Mengekstrak teks mentah dari {pdf_path.name} ke {raw_txt_path.name}...")
        
        try:
            doc = fitz.open(pdf_path)
            with open(raw_txt_path, "w", encoding="utf-8") as f_raw:
                for page in doc:
                    f_raw.write(f"--- Page {page.number + 1} ---\n")
                    f_raw.write(page.get_text())
                    f_raw.write("\n\n")
        except Exception as e:
            logger.error(f"Gagal mengekstrak teks mentah: {e}")
            sys.exit(1)
            
        # 2. Parsing pasal-pasal
        if "UU Nomor 1 Tahun 2026" in pdf_path.name:
            pasals = parse_uu1_2026(pdf_path, short_name, end_page)
        else:
            pasals = parse_standard_law(pdf_path, short_name, end_page)
            
        # 3. Jalankan safety merger pass untuk dokumen ini
        pasals = merge_split_pasals(pasals)
        
        all_parsed_pasals.extend(pasals)
        
    # ---------------------------------------------------------------------------
    # Post-processing: Link versi lama vs versi penyesuaian
    # ---------------------------------------------------------------------------
    logger.info("Melakukan post-processing status pasal...")
    
    # Cari set nomor pasal KUHP 2023 yang diubah oleh UU 1/2026
    modified_kuhp_pasals = set(
        p["mengubah_pasal"] for p in all_parsed_pasals 
        if p["uu"] == "UU No. 1 Tahun 2026" and p["mengubah_pasal"] is not None
    )
    
    # Ubah status pasal KUHP 2023 yang lama menjadi 'diubah'
    modified_count = 0
    for p in all_parsed_pasals:
        if p["uu"] == "UU No. 1 Tahun 2023" and p["pasal"] in modified_kuhp_pasals:
            p["status"] = "diubah"
            modified_count += 1
            
    logger.info(f"Berhasil memperbarui status {modified_count} pasal KUHP 2023 lama menjadi 'diubah'.")
    
    # ---------------------------------------------------------------------------
    # Save structured corpus
    # ---------------------------------------------------------------------------
    output_path = cfg.DATA_PROCESSED_DIR / "pasal_corpus.json"
    logger.info(f"Menyimpan corpus terstruktur ke {output_path}...")
    
    try:
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(all_parsed_pasals, f_out, ensure_ascii=False, indent=2)
        logger.info("=== DATA PIPELINE BERHASIL DISAJIKAN ===")
    except Exception as e:
        logger.error(f"Gagal menyimpan corpus: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
