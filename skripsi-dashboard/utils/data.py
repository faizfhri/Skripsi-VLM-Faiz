# Metrik dihitung dari CSV mentah; statistik waktu inferensi diambil dari report .txt
# karena tidak tersedia di CSV.

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Relatif terhadap direktori kerja saat `streamlit run app.py`; sesuaikan bila struktur foldermu berbeda.
DATASET_ROOT = Path("../dataset")

CLASS_INFO = {
    "kelas_1": {"label": "Kelas 1", "meterai": 1, "ttd": 1, "deskripsi": "Meterai ada, tanda tangan ada"},
    "kelas_2": {"label": "Kelas 2", "meterai": 0, "ttd": 1, "deskripsi": "Meterai tidak ada, tanda tangan ada"},
    "kelas_3": {"label": "Kelas 3", "meterai": 1, "ttd": 0, "deskripsi": "Meterai ada, tanda tangan tidak ada"},
    "kelas_4": {"label": "Kelas 4", "meterai": 0, "ttd": 0, "deskripsi": "Meterai tidak ada, tanda tangan tidak ada"},
}


@dataclass(frozen=True)
class ConfigMeta:
    id: str
    prompting: str
    inference: str
    model: str
    model_short: str
    csv_file: str
    report_file: str
    json_file: str
    duplicate_of: str | None = None  # id konfigurasi lain yang berbagi data yang sama
    catatan: str = ""


CONFIGS: dict[str, ConfigMeta] = {
    "K1": ConfigMeta(
        id="K1", prompting="Zero-Shot", inference="One-Pass",
        model="Qwen2.5-VL-7B-Instruct", model_short="Instruct",
        csv_file="hasil_K1_zeroshot_onepass_20260608_205437.csv",
        report_file="hasil_K1_zeroshot_onepass_20260608_205437_report.txt",
        json_file="hasil_K1_zeroshot_onepass_20260608_205437_detail.json",
    ),
    "K2": ConfigMeta(
        id="K2", prompting="Few-Shot", inference="One-Pass",
        model="Qwen2.5-VL-7B-Instruct", model_short="Instruct",
        csv_file="hasil_K2K3_fewshot_onepass_20260608_222148.csv",
        report_file="hasil_K2K3_fewshot_onepass_20260608_222148_report.txt",
        json_file="hasil_K2K3_fewshot_onepass_20260608_222148_detail.json",
    ),
    "K3": ConfigMeta(
        id="K3", prompting="Few-Shot", inference="One-Pass",
        model="Qwen2.5-VL-7B-Instruct", model_short="Instruct",
        csv_file="hasil_K2K3_fewshot_onepass_20260608_222148.csv",
        report_file="hasil_K2K3_fewshot_onepass_20260608_222148_report.txt",
        json_file="hasil_K2K3_fewshot_onepass_20260608_222148_detail.json",
        duplicate_of="K2",
        catatan="K2 dan K3 identik pada desain ablation study ini; satu proses evaluasi melayani dua peran perbandingan.",
    ),
    "K4": ConfigMeta(
        id="K4", prompting="Few-Shot", inference="Two-Pass",
        model="Qwen2.5-VL-7B-Instruct", model_short="Instruct",
        csv_file="hasil_K4K6_20260608_201305.csv",
        report_file="hasil_K4K6_20260608_201305_report.txt",
        json_file="hasil_K4K6_20260608_201305_detail.json",
    ),
    "K5": ConfigMeta(
        id="K5", prompting="Few-Shot", inference="Two-Pass",
        model="Qwen2.5-VL-7B-AWQ", model_short="AWQ",
        csv_file="hasil_K5_20260608_225153.csv",
        report_file="hasil_K5_20260608_225153_report.txt",
        json_file="hasil_K5_20260608_225153_detail.json",
    ),
    "K6": ConfigMeta(
        id="K6", prompting="Few-Shot", inference="Two-Pass",
        model="Qwen2.5-VL-7B-Instruct", model_short="Instruct",
        csv_file="hasil_K4K6_20260608_201305.csv",
        report_file="hasil_K4K6_20260608_201305_report.txt",
        json_file="hasil_K4K6_20260608_201305_detail.json",
        duplicate_of="K4",
        catatan="K4 dan K6 identik pada desain ablation study ini; satu proses evaluasi melayani dua peran perbandingan.",
    ),
    "K7": ConfigMeta(
        id="K7", prompting="Few-Shot", inference="Two-Pass",
        model="Qwen3-VL-8B", model_short="Qwen3-VL",
        csv_file="hasil_K7_20260608_231724.csv",
        report_file="hasil_K7_20260608_231724_report.txt",
        json_file="hasil_K7_20260608_231724_detail.json",
    ),
}

CONFIG_ORDER = ["K1", "K2", "K3", "K4", "K5", "K6", "K7"]

ABLATION_STAGES = [
    {
        "judul": "Tahap 1 - Pengaruh Strategi Prompting",
        "sub_bab": "4.2",
        "deskripsi": (
            "Membandingkan prompting zero-shot melawan few-shot pada skema inferensi "
            "one-pass yang sama, untuk melihat apakah penambahan contoh deskriptif "
            "membantu atau justru mengganggu keputusan model."
        ),
        "anggota": ["K1", "K2"],
    },
    {
        "judul": "Tahap 2 - Pengaruh Skema Inferensi",
        "sub_bab": "4.3",
        "deskripsi": (
            "Membandingkan skema one-pass melawan two-pass pada prompting few-shot yang "
            "sama, untuk menguji apakah pemisahan keputusan meterai dan tanda tangan "
            "menjadi dua tahap terpisah memperbaiki presisi."
        ),
        "anggota": ["K2", "K4"],
    },
    {
        "judul": "Tahap 3 - Pengaruh Pilihan Model",
        "sub_bab": "4.4",
        "deskripsi": (
            "Membandingkan tiga model/varian pada kombinasi prompting dan skema inferensi "
            "terbaik (few-shot, two-pass): model dasar, varian terkuantisasi AWQ, dan "
            "generasi model yang lebih baru."
        ),
        "anggota": ["K4", "K5", "K7"],
    },
]


def resolve_config(config_id: str) -> ConfigMeta:
    return CONFIGS[config_id]


@st.cache_data(show_spinner=False)
def load_csv(config_id: str) -> pd.DataFrame:
    meta = CONFIGS[config_id]
    path = DATA_DIR / meta.csv_file
    df = pd.read_csv(path)
    df["kelas_label"] = df["kelas"].map(lambda k: CLASS_INFO.get(k, {}).get("label", k))
    return df


@st.cache_data(show_spinner=False)
def load_report_text(config_id: str) -> str:
    meta = CONFIGS[config_id]
    path = DATA_DIR / meta.report_file
    return path.read_text(encoding="utf-8", errors="replace")


@st.cache_data(show_spinner=False)
def load_detail_json(config_id: str) -> list[dict]:
    meta = CONFIGS[config_id]
    path = DATA_DIR / meta.json_file
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _binary_metrics(df: pd.DataFrame, col_true: str, col_pred: str) -> dict:
    tp = int(((df[col_true] == 1) & (df[col_pred] == 1)).sum())
    tn = int(((df[col_true] == 0) & (df[col_pred] == 0)).sum())
    fp = int(((df[col_true] == 0) & (df[col_pred] == 1)).sum())
    fn = int(((df[col_true] == 1) & (df[col_pred] == 0)).sum())
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
    }


def _per_class_breakdown(df: pd.DataFrame, col_true: str, col_pred: str) -> pd.DataFrame:
    rows = []
    for kelas, info in CLASS_INFO.items():
        sub = df[df["kelas"] == kelas]
        if sub.empty:
            continue
        benar = int((sub[col_true] == sub[col_pred]).sum())
        total = len(sub)
        rows.append({
            "kelas": kelas,
            "label": info["label"],
            "deskripsi": info["deskripsi"],
            "n": total,
            "benar": benar,
            "salah": total - benar,
            "akurasi": benar / total if total else 0.0,
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compute_metrics(config_id: str) -> dict:
    df = load_csv(config_id)
    meterai = _binary_metrics(df, "meterai_asli", "meterai_prediksi")
    ttd = _binary_metrics(df, "ttd_asli", "ttd_prediksi")

    combined_benar = int(
        ((df["meterai_asli"] == df["meterai_prediksi"]) & (df["ttd_asli"] == df["ttd_prediksi"])).sum()
    )
    n = len(df)

    misclassified_meterai = df[df["meterai_asli"] != df["meterai_prediksi"]][
        ["nama_file", "kelas", "kelas_label", "meterai_asli", "meterai_prediksi"]
    ].reset_index(drop=True)
    misclassified_ttd = df[df["ttd_asli"] != df["ttd_prediksi"]][
        ["nama_file", "kelas", "kelas_label", "ttd_asli", "ttd_prediksi"]
    ].reset_index(drop=True)

    return {
        "n_total": n,
        "meterai": meterai,
        "ttd": ttd,
        "meterai_per_kelas": _per_class_breakdown(df, "meterai_asli", "meterai_prediksi"),
        "ttd_per_kelas": _per_class_breakdown(df, "ttd_asli", "ttd_prediksi"),
        "combined_benar": combined_benar,
        "combined_accuracy": combined_benar / n if n else 0.0,
        "misclassified_meterai": misclassified_meterai,
        "misclassified_ttd": misclassified_ttd,
    }


_TIMING_PATTERNS = {
    "total_api_calls": r"Total API calls\s*:\s*([\d.,]+)",
    "total_waktu_detik": r"Total waktu\s*:\s*([\d.,]+)\s*detik",
    "rata_rata_per_call": r"Rata-rata/call\s*:\s*([\d.,]+)\s*detik",
    "min_detik": r"Min\s*:\s*([\d.,]+)\s*detik",
    "max_detik": r"Max\s*:\s*([\d.,]+)\s*detik",
    "p50_detik": r"P50 \(median\)\s*:\s*([\d.,]+)\s*detik",
    "p90_detik": r"P90\s*:\s*([\d.,]+)\s*detik",
    "p95_detik": r"P95\s*:\s*([\d.,]+)\s*detik",
    "rata_rata_per_file": r"Rata-rata/file\s*:\s*([\d.,]+)\s*detik",
    "durasi_menit": r"Durasi\s*:\s*([\d.,]+)\s*menit",
}


@st.cache_data(show_spinner=False)
def parse_timing_stats(config_id: str) -> dict:
    text = load_report_text(config_id)
    result = {}
    for key, pattern in _TIMING_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1).replace(",", "")
            result[key] = float(value)
        else:
            result[key] = None
    return result


@st.cache_data(show_spinner=False)
def all_configs_summary() -> pd.DataFrame:
    """Ringkasan metrik seluruh konfigurasi untuk halaman perbandingan."""
    rows = []
    for cid in CONFIG_ORDER:
        meta = CONFIGS[cid]
        m = compute_metrics(cid)
        t = parse_timing_stats(cid)
        rows.append({
            "id": cid,
            "prompting": meta.prompting,
            "inference": meta.inference,
            "model": meta.model_short,
            "model_full": meta.model,
            "f1_meterai": m["meterai"]["f1"],
            "f1_ttd": m["ttd"]["f1"],
            "acc_meterai": m["meterai"]["accuracy"],
            "acc_ttd": m["ttd"]["accuracy"],
            "prec_meterai": m["meterai"]["precision"],
            "prec_ttd": m["ttd"]["precision"],
            "rec_meterai": m["meterai"]["recall"],
            "rec_ttd": m["ttd"]["recall"],
            "akurasi_gabungan": m["combined_accuracy"],
            "rata_rata_per_file_detik": t["rata_rata_per_file"],
            "duplicate_of": meta.duplicate_of,
        })
    return pd.DataFrame(rows)


def find_file_detail(config_id: str, nama_file: str) -> dict | None:
    detail = load_detail_json(config_id)
    for item in detail:
        if item.get("file") == nama_file:
            return item
    return None


def pdf_support_available() -> bool:
    try:
        import fitz  # noqa: F401
        return True
    except ImportError:
        return False


@st.cache_data(show_spinner=False)
def render_pdf_page(pdf_path_str: str, page_index: int, zoom: float = 2.0):
    # Returns (status, payload) where status is "ok" / "no_module" / "not_found" / "error".
    try:
        import fitz
    except ImportError:
        return "no_module", None

    path = Path(pdf_path_str)
    if not path.exists():
        return "not_found", None

    try:
        doc = fitz.open(str(path))
        n_pages = doc.page_count
        idx = min(max(page_index, 0), max(n_pages - 1, 0))
        page = doc[idx]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img_bytes = pix.tobytes("png")
        doc.close()
        return "ok", {"image": img_bytes, "n_pages": n_pages, "page_index": idx}
    except Exception as e:  # noqa: BLE001
        return "error", str(e)
