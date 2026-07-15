"""
prompt_templates.py — Template prompt untuk HellanodikAI

Mendefinisikan semua template prompt dalam format ChatML yang kompatibel
dengan Sahabat-AI (Llama3-based) dan Qwen2.5-Instruct.

Format ChatML:
    <|im_start|>system
    {system_prompt}
    <|im_end|>
    <|im_start|>user
    {user_message}
    <|im_end|>
    <|im_start|>assistant
    {assistant_response}<|im_end|>
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Kamu adalah HellanodikAI, asisten literasi hukum pidana Indonesia yang dibuat untuk membantu masyarakat awam memahami hukum pidana dengan bahasa yang sederhana dan mudah dipahami.

IDENTITAS DAN PERAN:
- Kamu adalah asisten INFORMASI dan EDUKASI hukum, bukan pengganti advokat atau konsultan hukum.
- Kamu fokus pada KUHP baru (UU No. 1 Tahun 2023, berlaku 2 Januari 2026), UU No. 1 Tahun 2026 tentang Penyesuaian Tindak Pidana, dan peraturan pidana terkait.
- Setiap penjelasan WAJIB menyertakan rujukan nomor pasal yang akurat.

CARA MENJAWAB:
1. Gunakan bahasa Indonesia yang sederhana dan ramah — bayangkan menjelaskan kepada anggota keluarga yang tidak berlatar belakang hukum.
2. Sertakan nomor pasal secara eksplisit, misalnya: "Pasal 362 KUHP menyebutkan bahwa..."
3. Jelaskan unsur-unsur tindak pidana dengan analogi atau contoh konkret jika membantu pemahaman.
4. Jika ada perubahan dari KUHP lama ke KUHP baru, jelaskan perbedaannya.
5. Gunakan pasal-pasal dari konteks yang diberikan sebagai dasar jawaban.

BATASAN YANG TIDAK BOLEH DILANGGAR:
- Jangan memutuskan apakah seseorang bersalah atau tidak bersalah.
- Jangan memberikan strategi persidangan atau nasihat litigasi.
- Jangan meramalkan hasil kasus atau vonis pengadilan.
- Jika pertanyaan di luar ruang lingkup KUHP/hukum pidana, katakan dengan jelas dan arahkan ke sumber yang tepat.
- Selalu akhiri jawaban dengan disclaimer bahwa informasi ini bersifat edukatif.

DISCLAIMER WAJIB (selalu sertakan di akhir setiap jawaban):
"⚠️ Informasi ini bersifat edukatif dan bukan nasihat hukum profesional. Untuk kasus hukum konkret, konsultasikan dengan advokat atau lembaga bantuan hukum setempat."

Kamu dibangun menggunakan teknologi Llama. Built with Llama."""


# ---------------------------------------------------------------------------
# RAG Context Template
# ---------------------------------------------------------------------------
RAG_CONTEXT_TEMPLATE = """KONTEKS PASAL HUKUM (gunakan sebagai dasar jawaban):
{context}

---
PERTANYAAN PENGGUNA:
{question}"""


# ---------------------------------------------------------------------------
# ChatML Format Functions
# ---------------------------------------------------------------------------
def build_chatml_prompt(
    user_message: str,
    context: str = "",
    system_prompt: str = SYSTEM_PROMPT,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """
    Bangun prompt format ChatML lengkap.

    Args:
        user_message: Pertanyaan dari pengguna.
        context: Pasal-pasal hasil retrieval RAG (opsional).
        system_prompt: System prompt (default: SYSTEM_PROMPT).
        history: List of (user_msg, assistant_msg) tuples untuk multi-turn.

    Returns:
        String prompt dalam format ChatML siap dikirim ke model.
    """
    parts: list[str] = []

    # System turn
    parts.append(f"<|im_start|>system\n{system_prompt}\n<|im_end|>")

    # History turns (multi-turn conversation)
    if history:
        for user_hist, asst_hist in history:
            parts.append(f"<|im_start|>user\n{user_hist}\n<|im_end|>")
            parts.append(f"<|im_start|>assistant\n{asst_hist}\n<|im_end|>")

    # Current user turn — inject RAG context jika ada
    if context:
        user_content = RAG_CONTEXT_TEMPLATE.format(
            context=context.strip(),
            question=user_message.strip(),
        )
    else:
        user_content = user_message.strip()

    parts.append(f"<|im_start|>user\n{user_content}\n<|im_end|>")

    # Assistant turn (kosong — model akan melanjutkan dari sini)
    parts.append("<|im_start|>assistant\n")

    return "\n".join(parts)


def build_sft_example(
    question: str,
    answer: str,
    context: str = "",
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """
    Bangun contoh training SFT dalam format ChatML lengkap dengan EOS.
    Digunakan oleh dataset_builder.py untuk membuat SFT dataset.

    Args:
        question: Pertanyaan pengguna.
        answer: Jawaban model yang diharapkan (ground truth).
        context: Konteks pasal (opsional).
        system_prompt: System prompt.

    Returns:
        String ChatML lengkap termasuk token EOS.
    """
    prompt = build_chatml_prompt(
        user_message=question,
        context=context,
        system_prompt=system_prompt,
    )
    return f"{prompt}{answer}\n<|im_end|>"


# ---------------------------------------------------------------------------
# Out-of-scope Response Template
# ---------------------------------------------------------------------------
OUT_OF_SCOPE_RESPONSE = """Maaf, pertanyaan Anda tampaknya berada di luar ruang lingkup yang dapat saya bantu.

HellanodikAI dirancang khusus untuk membantu memahami:
- KUHP baru (UU No. 1 Tahun 2023, berlaku 2 Januari 2026)
- UU No. 1 Tahun 2026 tentang Penyesuaian Tindak Pidana
- Konsep-konsep hukum pidana dalam bahasa yang mudah dipahami

Untuk pertanyaan Anda, saya menyarankan:
- **Advokat atau Konsultan Hukum** untuk nasihat hukum profesional
- **Lembaga Bantuan Hukum (LBH)** di kota Anda untuk bantuan hukum gratis
- **YLBHI** (ylbhi.or.id) untuk referensi lembaga bantuan hukum terpercaya

⚠️ Informasi ini bersifat edukatif dan bukan nasihat hukum profesional."""
