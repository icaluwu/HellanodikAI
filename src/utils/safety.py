"""
safety.py — Safety layer untuk HellanodikAI

Tanggung jawab:
1. Deteksi pertanyaan out-of-scope (strategi persidangan, vonis, dll.)
2. Injeksi disclaimer wajib ke setiap output
3. Normalisasi referensi nomor pasal dalam output

Disclaimer diinjeksi secara rule-based (tidak bergantung LLM)
agar jaminan keamanannya bersifat deterministik.
"""

from __future__ import annotations

import re
import sys
import os

# Tambahkan root ke path agar config bisa diimport
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config as cfg
from loguru import logger


# ---------------------------------------------------------------------------
# Out-of-scope Detection
# ---------------------------------------------------------------------------

# Pola pertanyaan yang meminta vonis/keputusan hukum konkret
_OUT_OF_SCOPE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(memutuskan|putusan)\s+(vonis|hukuman)\b", re.I),
    re.compile(r"\b(apakah|apa)\s+(saya|dia|terdakwa)\s+(bersalah|bisa\s+dihukum)\b", re.I),
    re.compile(r"\bstrategi\s+(persidangan|membela\s+diri|sidang)\b", re.I),
    re.compile(r"\b(berapa|apa)\s+hukuman\s+(yang\s+tepat|yang\s+pantas|yang\s+seharusnya)\b", re.I),
    re.compile(r"\b(bisakah|dapatkah)\s+(saya|kami)\s+menang\b", re.I),
    re.compile(r"\bvonis\s+apa\s+yang\s+akan\b", re.I),
]

# Kata kunci tambahan dari config (string-based, case-insensitive)
_OUT_OF_SCOPE_KEYWORDS: list[str] = [kw.lower() for kw in cfg.OUT_OF_SCOPE_KEYWORDS]


def is_out_of_scope(query: str) -> bool:
    """
    Cek apakah query meminta informasi di luar scope HellanodikAI.

    Args:
        query: Teks pertanyaan dari pengguna.

    Returns:
        True jika query out-of-scope, False jika dalam scope.
    """
    query_lower = query.lower()

    # Cek keyword config
    for kw in _OUT_OF_SCOPE_KEYWORDS:
        if kw in query_lower:
            logger.debug(f"Out-of-scope keyword terdeteksi: '{kw}'")
            return True

    # Cek regex patterns
    for pattern in _OUT_OF_SCOPE_PATTERNS:
        if pattern.search(query):
            logger.debug(f"Out-of-scope pattern terdeteksi: {pattern.pattern}")
            return True

    return False


# ---------------------------------------------------------------------------
# Disclaimer Injection
# ---------------------------------------------------------------------------

def inject_disclaimer(text: str, lang: str = "id") -> str:
    """
    Tambahkan disclaimer wajib ke akhir output jika belum ada.

    Disclaimer diinjeksi secara deterministik — tidak bergantung pada
    apakah LLM memilih untuk menyertakannya atau tidak.

    Args:
        text: Output dari LLM.
        lang: Bahasa disclaimer ('id' = Indonesia).

    Returns:
        Teks dengan disclaimer di akhir.
    """
    disclaimer = cfg.DISCLAIMER_ID

    # Jika disclaimer sudah ada (dari LLM), jangan duplikat
    if "bukan nasihat hukum profesional" in text.lower():
        logger.debug("Disclaimer sudah ada dalam output LLM, tidak diduplikat.")
        return text.rstrip()

    return text.rstrip() + disclaimer


# ---------------------------------------------------------------------------
# Pasal Reference Normalization
# ---------------------------------------------------------------------------

# Pattern untuk mendeteksi referensi pasal dalam teks
_PASAL_PATTERN = re.compile(
    r"[Pp]asal\s+(\d+(?:\s*[A-Za-z])?)"   # "Pasal 362" atau "Pasal 5A"
    r"(?:\s+[Aa]yat\s+\((\d+)\))?"         # opsional: "ayat (1)"
    r"(?:\s+huruf\s+([a-zA-Z]))?"           # opsional: "huruf a"
    r"(?:\s+(KUHP|KUHAP|KUHPidana))?"      # opsional: "KUHP"
)


def extract_pasal_references(text: str) -> list[dict]:
    """
    Ekstrak semua referensi pasal dari teks.

    Args:
        text: Teks yang akan dianalisis (bisa pertanyaan atau jawaban).

    Returns:
        List of dict dengan keys: pasal, ayat, huruf, undang_undang.

    Example:
        >>> refs = extract_pasal_references("Pasal 362 ayat (1) KUHP")
        >>> refs[0]
        {'pasal': '362', 'ayat': '1', 'huruf': None, 'undang_undang': 'KUHP'}
    """
    refs = []
    for match in _PASAL_PATTERN.finditer(text):
        refs.append({
            "pasal":         match.group(1).strip(),
            "ayat":          match.group(2),
            "huruf":         match.group(3),
            "undang_undang": match.group(4) or "KUHP",  # default ke KUHP
        })
    return refs


def validate_pasal_in_response(response: str) -> bool:
    """
    Validasi bahwa respons LLM menyertakan minimal satu referensi pasal.

    Args:
        response: Output dari LLM.

    Returns:
        True jika ada referensi pasal, False jika tidak ada.
    """
    refs = extract_pasal_references(response)
    if not refs:
        logger.warning("Respons LLM tidak menyertakan referensi pasal!")
        return False
    logger.debug(f"Referensi pasal ditemukan: {len(refs)} pasal")
    return True


# ---------------------------------------------------------------------------
# Response Post-processor
# ---------------------------------------------------------------------------

def process_response(
    raw_response: str,
    require_pasal: bool = True,
) -> dict[str, str | bool | list]:
    """
    Post-process output LLM: inject disclaimer, validasi pasal, cleanup.

    Args:
        raw_response: Output mentah dari LLM.
        require_pasal: Jika True, warning jika tidak ada referensi pasal.

    Returns:
        Dict dengan keys:
            - 'text': Output final yang sudah diproses.
            - 'has_pasal_ref': Boolean apakah ada referensi pasal.
            - 'pasal_refs': List referensi pasal yang ditemukan.
            - 'disclaimer_injected': Boolean apakah disclaimer diinjeksi.
    """
    # Bersihkan token ChatML yang mungkin muncul di output
    cleaned = raw_response.strip()
    cleaned = re.sub(r"<\|im_end\|>.*$", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"<\|im_start\|>.*$", "", cleaned, flags=re.DOTALL).strip()

    # Cek apakah disclaimer sudah ada sebelum injeksi
    had_disclaimer = "bukan nasihat hukum profesional" in cleaned.lower()

    # Injeksi disclaimer
    final_text = inject_disclaimer(cleaned)

    # Validasi referensi pasal
    pasal_refs = extract_pasal_references(final_text)
    has_pasal = len(pasal_refs) > 0
    if require_pasal and not has_pasal:
        logger.warning("Output tidak menyertakan referensi pasal.")

    return {
        "text":                final_text,
        "has_pasal_ref":       has_pasal,
        "pasal_refs":          pasal_refs,
        "disclaimer_injected": not had_disclaimer,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Test: is_out_of_scope ===")
    test_queries = [
        ("Apa itu pencurian menurut KUHP baru?", False),
        ("Apakah saya bisa menang di pengadilan?", True),
        ("Jelaskan pasal tentang penganiayaan.", False),
        ("Strategi persidangan untuk kasus narkoba?", True),
        ("Berapa ancaman pidana untuk penipuan?", False),
    ]
    for q, expected in test_queries:
        result = is_out_of_scope(q)
        status = "✓" if result == expected else "✗ FAIL"
        print(f"  [{status}] '{q[:50]}...' → out_of_scope={result}")

    print()
    print("=== Test: extract_pasal_references ===")
    sample = "Berdasarkan Pasal 362 KUHP dan Pasal 5 ayat (1) huruf a KUHAP..."
    refs = extract_pasal_references(sample)
    for r in refs:
        print(f"  → {r}")

    print()
    print("=== Test: process_response ===")
    sample_resp = "Pencurian diatur dalam Pasal 362 KUHP baru..."
    result = process_response(sample_resp)
    print(f"  has_pasal_ref:       {result['has_pasal_ref']}")
    print(f"  disclaimer_injected: {result['disclaimer_injected']}")
    print(f"  pasal_refs:          {result['pasal_refs']}")
    print(f"  text[-100:]:         ...{result['text'][-100:]}")
