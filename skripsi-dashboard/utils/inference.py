# Logikanya sengaja dibuat identik dengan skrip evaluasi asli (K1inszero.py dkk.)
# supaya hasil halaman "Coba Inferensi" konsisten dengan hasil skripsi.

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Callable

import requests

from . import prompts as P

MAX_TOKENS = 20
TIMEOUT_SECONDS = 60
RETRY_DELAYS = [2, 4, 8]
CONNECTION_ERROR_MARKER = "Tidak dapat terhubung ke server"


@dataclass(frozen=True)
class InferenceEndpoint:
    url: str
    model_name: str
    scheme: str  # "one-pass" atau "two-pass"


# Sesuaikan bila server VLM/DGX Spark berjalan di alamat atau port yang berbeda.
INFERENCE_ENDPOINTS: dict[str, InferenceEndpoint] = {
    "K1": InferenceEndpoint("http://localhost:8003/v1/chat/completions", "qwen2.5-vl-7b-instruct", "one-pass"),
    "K2": InferenceEndpoint("http://localhost:8003/v1/chat/completions", "qwen2.5-vl-7b-instruct", "one-pass"),
    "K3": InferenceEndpoint("http://localhost:8003/v1/chat/completions", "qwen2.5-vl-7b-instruct", "one-pass"),
    "K4": InferenceEndpoint("http://localhost:8003/v1/chat/completions", "qwen2.5-vl-7b-instruct", "two-pass"),
    "K5": InferenceEndpoint("http://localhost:8002/v1/chat/completions", "qwen2.5-vl-7b-awq", "two-pass"),
    "K6": InferenceEndpoint("http://localhost:8003/v1/chat/completions", "qwen2.5-vl-7b-instruct", "two-pass"),
    "K7": InferenceEndpoint("http://localhost:8001/v1/chat/completions", "qwen3-vl-8b", "two-pass"),
}


def pdf_support_available() -> bool:
    try:
        import fitz  # noqa: F401
        return True
    except ImportError:
        return False


def pdf_bytes_to_jpeg_pages(pdf_bytes: bytes, zoom: float = 2.0) -> list[bytes]:
    """Render tiap halaman PDF menjadi JPEG bytes, setara dengan
    pdf_to_images_base64 pada skrip evaluasi asli (tanpa file sementara)."""
    import fitz

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(zoom, zoom)
    pages = []
    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat)
            pages.append(pix.tobytes("jpeg"))
    finally:
        doc.close()
    return pages


def _call_api(url: str, model_name: str, prompt_text: str, image_b64: str):
    """Satu kali panggilan chat completion, dengan retry 3x seperti skrip asli.
    Mengembalikan tuple (raw_text, elapsed_detik, pesan_error)."""
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }],
        "max_tokens": MAX_TOKENS,
    }

    last_error = None
    for attempt, delay in enumerate(RETRY_DELAYS):
        try:
            t0 = time.time()
            resp = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS, headers={"Connection": "keep-alive"})
            elapsed = time.time() - t0
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"], elapsed, None
            last_error = f"Status {resp.status_code}: {resp.text[:200]}"
        except requests.exceptions.ConnectionError:
            last_error = f"{CONNECTION_ERROR_MARKER} di {url}. Pastikan server VLM sedang berjalan."
        except requests.exceptions.Timeout:
            last_error = f"Server tidak merespons dalam {TIMEOUT_SECONDS} detik."
        except Exception as e:  # noqa: BLE001
            last_error = str(e)[:200]

        if attempt < len(RETRY_DELAYS) - 1:
            time.sleep(delay)

    return "", None, last_error


def parse_answer(raw_text: str) -> dict:
    """Parsing identik dengan parsing_jawaban pada skrip evaluasi asli."""
    hasil = {"meterai_prediksi": 0, "ttd_prediksi": 0}
    if not raw_text:
        return hasil
    for line in raw_text.lower().strip().split("\n"):
        line = line.strip()
        if "meterai:" in line or "materai:" in line:
            part = line.split(":", 1)[1].strip() if ":" in line else ""
            hasil["meterai_prediksi"] = 1 if ("ada" in part and not part.startswith("tidak")) else 0
        elif "tanda tangan:" in line:
            part = line.split(":", 1)[1].strip() if ":" in line else ""
            hasil["ttd_prediksi"] = 1 if ("ada" in part and not part.startswith("tidak")) else 0
    return hasil


def _run_page_inference(config_id: str, image_jpeg_bytes: bytes, endpoint: InferenceEndpoint) -> dict:
    image_b64 = base64.b64encode(image_jpeg_bytes).decode("utf-8")

    if endpoint.scheme == "one-pass":
        prompt = P.PROMPT_K1_ZEROSHOT_ONEPASS if config_id == "K1" else P.PROMPT_FEWSHOT_ONEPASS
        raw, elapsed, error = _call_api(endpoint.url, endpoint.model_name, prompt, image_b64)
        return {
            "raw": raw, "raw_meterai": None, "raw_ttd": None,
            "parsed": parse_answer(raw), "elapsed": elapsed, "error": error,
        }

    # two-pass: dua panggilan terpisah, meterai lalu tanda tangan
    raw_m, elapsed_m, error_m = _call_api(endpoint.url, endpoint.model_name, P.PROMPT_METERAI_TWOPASS, image_b64)
    raw_t, elapsed_t, error_t = _call_api(endpoint.url, endpoint.model_name, P.PROMPT_TTD_TWOPASS, image_b64)
    raw_combined = f"{raw_m}\n{raw_t}"
    total_elapsed = None
    if elapsed_m is not None and elapsed_t is not None:
        total_elapsed = elapsed_m + elapsed_t
    return {
        "raw": raw_combined, "raw_meterai": raw_m, "raw_ttd": raw_t,
        "parsed": parse_answer(raw_combined), "elapsed": total_elapsed,
        "error": error_m or error_t,
    }


def run_full_inference(
    config_id: str,
    pdf_bytes: bytes,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    # Returns dict: n_pages, pages (list of per-page results), aggregated (OR across halaman),
    # total_elapsed, fatal_error (None bila sukses).
    endpoint = INFERENCE_ENDPOINTS[config_id]

    if not pdf_support_available():
        return {
            "fatal_error": "Paket PyMuPDF belum terpasang. Jalankan `pip install pymupdf` lalu coba lagi.",
            "pages": [], "n_pages": 0, "aggregated": None, "total_elapsed": 0.0,
        }

    try:
        page_images = pdf_bytes_to_jpeg_pages(pdf_bytes)
    except Exception as e:  # noqa: BLE001
        return {
            "fatal_error": f"Gagal membaca berkas PDF: {e}",
            "pages": [], "n_pages": 0, "aggregated": None, "total_elapsed": 0.0,
        }

    if not page_images:
        return {
            "fatal_error": "Berkas PDF tidak memiliki halaman yang dapat dibaca.",
            "pages": [], "n_pages": 0, "aggregated": None, "total_elapsed": 0.0,
        }

    pages = []
    total_elapsed = 0.0
    for i, img_bytes in enumerate(page_images):
        if progress_callback:
            progress_callback(i, len(page_images))

        result = _run_page_inference(config_id, img_bytes, endpoint)
        result["page"] = i
        result["image"] = img_bytes
        pages.append(result)
        if result.get("elapsed"):
            total_elapsed += result["elapsed"]

        if result.get("error") and CONNECTION_ERROR_MARKER in result["error"]:
            return {
                "fatal_error": (
                    f"Tidak dapat terhubung ke server inferensi di {endpoint.url}. "
                    "Pastikan server VLM/DGX Spark untuk konfigurasi ini sedang berjalan, "
                    "lalu coba lagi."
                ),
                "pages": pages, "n_pages": len(page_images),
                "aggregated": None, "total_elapsed": total_elapsed,
            }

    aggregated = {"meterai_prediksi": 0, "ttd_prediksi": 0}
    for p in pages:
        if p["parsed"]["meterai_prediksi"] == 1:
            aggregated["meterai_prediksi"] = 1
        if p["parsed"]["ttd_prediksi"] == 1:
            aggregated["ttd_prediksi"] = 1

    return {
        "n_pages": len(pages), "pages": pages, "aggregated": aggregated,
        "total_elapsed": total_elapsed, "fatal_error": None,
    }
