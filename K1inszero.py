import os
import fitz
import base64
import requests
import pandas as pd
import time
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

DGX_URL        = "http://localhost:8003/v1/chat/completions"
MODEL_NAME     = "qwen2.5-vl-7b-instruct"
KONFIGURASI    = "K1 | Zero-Shot | One-Pass | Qwen2.5-VL-7B-Instruct"
DATASET_FOLDER = "../datasetlengkap"

TIMESTAMP      = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV     = f"hasil_K1_zeroshot_onepass_{TIMESTAMP}.csv"
OUTPUT_JSON    = f"hasil_K1_zeroshot_onepass_{TIMESTAMP}_detail.json"
OUTPUT_REPORT  = f"hasil_K1_zeroshot_onepass_{TIMESTAMP}_report.txt"

# Zero-shot one-pass: satu instruksi mendeteksi kedua atribut sekaligus, tanpa contoh.
PROMPT_ONEPASS = """Kamu adalah analis dokumen keuangan Indonesia. Tugasmu mendeteksi dua atribut pada dokumen invoice: METERAI dan TANDA TANGAN.

---
DEFINISI METERAI — Ada HANYA jika ditemukan salah satu secara EKSPLISIT dan JELAS:
- Meterai Tempel Fisik: stiker bergambar Garuda Pancasila, bertuliskan "METERAI TEMPEL", nilai "10.000", nomor seri 16 karakter.
- E-Meterai: kotak persegi berisi QR code DENGAN tulisan "METERAI ELEKTRONIK" dan nomor seri digital. Tulisan "METERAI ELEKTRONIK" HARUS terlihat eksplisit.
- Meterai Teraan: cap mesin bertuliskan nilai "10.000" atau "Rp10.000" secara eksplisit sebagai meterai.

BUKAN Meterai:
- QR code TANPA tulisan "METERAI ELEKTRONIK"
- Stempel perusahaan (nama PT/CV, logo)
- Barcode atau QR code untuk verifikasi faktur pajak
- Tanda tangan atau coretan tangan
- Logo atau watermark perusahaan

---
DEFINISI TANDA TANGAN — Ada HANYA jika terlihat JELAS:
- Coretan/guratan tangan yang membentuk pola personal (bukan teks tercetak)
- Harus JELAS TERLIHAT sebagai guratan tangan manusia, bukan artefak cetak
- Biasanya berwarna biru tua atau hitam dengan variasi ketebalan garis

BUKAN Tanda Tangan:
- Garis-garis pada meterai tempel atau e-meterai
- Border atau garis tepi stempel perusahaan
- Garis dekoratif atau garis pemisah
- Artefak cetak, noise, atau bayangan dari scan
- Nama cetak/ketik saja tanpa coretan tangan
- Stempel perusahaan saja

---
ATURAN KETAT:
- Jika ragu pada salah satu atribut, jawab Tidak Ada untuk atribut tersebut.
- Jawab HANYA dalam format dua baris berikut, tanpa penjelasan tambahan.

FORMAT OUTPUT WAJIB (dua baris, tanpa teks lain):
METERAI: [Ada/Tidak Ada]
TANDA TANGAN: [Ada/Tidak Ada]

---
Perhatikan dokumen berikut. Jawab dua baris:
"""

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=5,
    pool_maxsize=10,
    max_retries=requests.adapters.Retry(total=0)
)
session.mount('http://', adapter)

def pdf_to_images_base64(pdf_path):
    doc = fitz.open(pdf_path)
    images_b64 = []
    mat = fitz.Matrix(2.0, 2.0)
    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            pix  = page.get_pixmap(matrix=mat)
            temp = f"temp_page_{page_num}.jpg"
            pix.save(temp)
            with open(temp, "rb") as f:
                images_b64.append(base64.b64encode(f.read()).decode("utf-8"))
            os.remove(temp)
        except Exception as e:
            print(f"[ERROR] Gagal konversi halaman {page_num}: {e}")
    return images_b64

inference_times = []

def _call_api(prompt_text, image_base64):
    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }],
        "max_tokens": 20
    }
    for attempt, delay in enumerate([2, 4, 8]):
        try:
            t0   = time.time()
            resp = session.post(DGX_URL, json=payload, timeout=60,
                                headers={"Connection": "keep-alive"})
            elapsed = time.time() - t0
            if resp.status_code == 200:
                inference_times.append(elapsed)
                return resp.json()["choices"][0]["message"]["content"]
            else:
                print(f"[WARNING] Status {resp.status_code}: {resp.text[:200]}")
                if attempt < 2:
                    time.sleep(delay)
                else:
                    return f"Error: {resp.text}"
        except Exception as e:
            print(f"[ERROR] Attempt {attempt+1} gagal: {str(e)[:100]}")
            if attempt < 2:
                time.sleep(delay)
            else:
                return f"Error: gagal connect setelah 3 percobaan"
    return "Error: unknown"

def panggil_model_vlm(image_base64):
    return _call_api(PROMPT_ONEPASS, image_base64)

def parsing_jawaban(raw_text):
    hasil = {"meterai_prediksi": 0, "ttd_prediksi": 0}
    if not raw_text or raw_text.startswith("Error"):
        return hasil
    for line in raw_text.lower().strip().split('\n'):
        line = line.strip()
        if 'meterai:' in line or 'materai:' in line:
            part = line.split(':', 1)[1].strip() if ':' in line else ""
            hasil["meterai_prediksi"] = 1 if ('ada' in part and not part.startswith('tidak')) else 0
        elif 'tanda tangan:' in line:
            part = line.split(':', 1)[1].strip() if ':' in line else ""
            hasil["ttd_prediksi"] = 1 if ('ada' in part and not part.startswith('tidak')) else 0
    return hasil

def analyze_multipage(pdf_path):
    images_b64 = pdf_to_images_base64(pdf_path)
    if not images_b64:
        return {"meterai_prediksi": 0, "ttd_prediksi": 0}, []

    all_results = []
    for page_num, img_b64 in enumerate(images_b64):
        try:
            raw    = panggil_model_vlm(img_b64)
            parsed = parsing_jawaban(raw)
            all_results.append({"page": page_num, "raw": raw, "parsed": parsed})
            print(f"    [p{page_num}] '{raw[:80].strip()}' "
                  f"-> M={parsed['meterai_prediksi']} T={parsed['ttd_prediksi']}")
        except Exception as e:
            print(f"[ERROR] Halaman {page_num}: {e}")

    # OR aggregation — Ada di halaman manapun = Ada
    aggregated = {"meterai_prediksi": 0, "ttd_prediksi": 0}
    for r in all_results:
        if r["parsed"]["meterai_prediksi"] == 1:
            aggregated["meterai_prediksi"] = 1
        if r["parsed"]["ttd_prediksi"] == 1:
            aggregated["ttd_prediksi"] = 1

    return aggregated, all_results

def hitung_metrik(df, gt_col, pred_col):
    y_true = df[gt_col]
    y_pred = df[pred_col]
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    per_kelas = {}
    for kelas in ['kelas_1', 'kelas_2', 'kelas_3', 'kelas_4']:
        sub = df[df['kelas'] == kelas]
        if len(sub) == 0:
            continue
        tp = int(((sub[gt_col]==1) & (sub[pred_col]==1)).sum())
        fn = int(((sub[gt_col]==1) & (sub[pred_col]==0)).sum())
        fp = int(((sub[gt_col]==0) & (sub[pred_col]==1)).sum())
        tn = int(((sub[gt_col]==0) & (sub[pred_col]==0)).sum())
        n  = len(sub)
        benar = int((sub[gt_col] == sub[pred_col]).sum())
        salah_df = sub[sub[gt_col] != sub[pred_col]]
        per_kelas[kelas] = {
            "n": n, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "benar": benar, "salah": n - benar,
            "accuracy_kelas": benar / n if n > 0 else 0,
            "file_salah": [
                {"file": row["nama_file"],
                 "gt": int(row[gt_col]),
                 "pred": int(row[pred_col])}
                for _, row in salah_df.iterrows()
            ]
        }

    return {
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1,
        "cm": {"tn": int(cm[0][0]), "fp": int(cm[0][1]),
               "fn": int(cm[1][0]), "tp": int(cm[1][1])},
        "per_kelas": per_kelas
    }

def cetak_metrik_terminal(label, m):
    print(f"\n{'='*55}")
    print(f"HASIL EVALUASI {label}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1-Score  : {m['f1']:.4f}")
    cm = m['cm']
    print(f"\n  Confusion Matrix:")
    print(f"                Pred=0   Pred=1")
    print(f"    Actual=0    TN={cm['tn']:4d}   FP={cm['fp']:4d}")
    print(f"    Actual=1    FN={cm['fn']:4d}   TP={cm['tp']:4d}")
    print(f"\n  Per kelas:")
    for kelas, k in m['per_kelas'].items():
        print(f"    {kelas} (n={k['n']}): "
              f"Benar={k['benar']} Salah={k['salah']} "
              f"Acc={k['accuracy_kelas']:.2%} | "
              f"TP={k['tp']} FN={k['fn']} FP={k['fp']} TN={k['tn']}")
        for sf in k['file_salah']:
            print(f"      [SALAH] {sf['file']} "
                  f"(GT={sf['gt']}, Pred={sf['pred']})")

if __name__ == "__main__":
    waktu_mulai = datetime.now()
    print("=" * 60)
    print(f"KONFIGURASI : {KONFIGURASI}")
    print(f"MODEL       : {MODEL_NAME}")
    print(f"SKEMA       : One-Pass (1 API call per halaman, 2 atribut sekaligus)")
    print(f"PROMPTING   : Zero-Shot (tanpa contoh)")
    print(f"MULAI       : {waktu_mulai.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    kunci_jawaban = {
        "kelas_1": {"meterai_asli": 1, "ttd_asli": 1},
        "kelas_2": {"meterai_asli": 0, "ttd_asli": 1},
        "kelas_3": {"meterai_asli": 1, "ttd_asli": 0},
        "kelas_4": {"meterai_asli": 0, "ttd_asli": 0},
    }

    data_rekapan   = []
    detail_results = []

    for nama_folder, kunci in kunci_jawaban.items():
        path_folder = os.path.join(DATASET_FOLDER, nama_folder)
        if not os.path.exists(path_folder):
            print(f"[WARNING] Folder tidak ditemukan: {path_folder}")
            continue

        daftar_pdf = sorted([f for f in os.listdir(path_folder) if f.endswith('.pdf')])
        print(f"\nMemproses {nama_folder} ({len(daftar_pdf)} file) "
              f"[GT: Meterai={kunci['meterai_asli']}, TTD={kunci['ttd_asli']}]")

        for file_pdf in tqdm(daftar_pdf):
            path_pdf = os.path.join(path_folder, file_pdf)
            try:
                prediksi, page_results = analyze_multipage(path_pdf)
                data_rekapan.append({
                    "nama_file":        file_pdf,
                    "kelas":            nama_folder,
                    "meterai_asli":     kunci["meterai_asli"],
                    "meterai_prediksi": prediksi["meterai_prediksi"],
                    "ttd_asli":         kunci["ttd_asli"],
                    "ttd_prediksi":     prediksi["ttd_prediksi"],
                })
                detail_results.append({
                    "file":         file_pdf,
                    "kelas":        nama_folder,
                    "aggregated":   prediksi,
                    "page_by_page": page_results,
                })
            except Exception as e:
                print(f"\n[ERROR] {file_pdf}: {e}")
                data_rekapan.append({
                    "nama_file":        file_pdf,
                    "kelas":            nama_folder,
                    "meterai_asli":     kunci["meterai_asli"],
                    "meterai_prediksi": -1,
                    "ttd_asli":         kunci["ttd_asli"],
                    "ttd_prediksi":     -1,
                })

    waktu_selesai = datetime.now()
    durasi_total  = (waktu_selesai - waktu_mulai).total_seconds()

    df = pd.DataFrame(data_rekapan)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[INFO] CSV disimpan: {OUTPUT_CSV}")

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(detail_results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] JSON detail disimpan: {OUTPUT_JSON}")

    df_valid = df[df['meterai_prediksi'] != -1].copy()
    n_error  = len(df) - len(df_valid)

    m_meterai = hitung_metrik(df_valid, "meterai_asli", "meterai_prediksi")
    m_ttd     = hitung_metrik(df_valid, "ttd_asli",     "ttd_prediksi")

    df_valid["keduanya_benar"] = (
        (df_valid["meterai_asli"]  == df_valid["meterai_prediksi"]) &
        (df_valid["ttd_asli"]      == df_valid["ttd_prediksi"])
    )
    acc_gabungan     = df_valid["keduanya_benar"].mean()
    n_keduanya_benar = df_valid["keduanya_benar"].sum()

    it = sorted(inference_times)
    it_stats = {}
    if it:
        n = len(it)
        it_stats = {
            "total_calls":  n,
            "total_detik":  sum(it),
            "total_menit":  sum(it) / 60,
            "avg":          sum(it) / n,
            "min":          it[0],
            "max":          it[-1],
            "p50":          it[n // 2],
            "p90":          it[int(n * 0.9)],
            "p95":          it[int(n * 0.95)],
            "avg_per_file": sum(it) / len(df_valid) if len(df_valid) > 0 else 0,
        }

    cetak_metrik_terminal("METERAI",      m_meterai)
    cetak_metrik_terminal("TANDA TANGAN", m_ttd)

    print(f"\n{'='*55}")
    print(f"AKURASI GABUNGAN (kedua atribut benar)")
    print(f"{'='*55}")
    print(f"  Benar keduanya : {n_keduanya_benar} / {len(df_valid)}")
    print(f"  Accuracy       : {acc_gabungan:.4f}")

    print(f"\n{'='*55}")
    print(f"INFERENCE TIME STATISTICS")
    print(f"{'='*55}")
    if it_stats:
        print(f"  Total API calls   : {it_stats['total_calls']} (1 call/halaman)")
        print(f"  Total waktu       : {it_stats['total_detik']:.2f} detik "
              f"({it_stats['total_menit']:.1f} menit)")
        print(f"  Rata-rata/call    : {it_stats['avg']:.3f} detik")
        print(f"  Min               : {it_stats['min']:.3f} detik")
        print(f"  Max               : {it_stats['max']:.3f} detik")
        print(f"  P50 (median)      : {it_stats['p50']:.3f} detik")
        print(f"  P90               : {it_stats['p90']:.3f} detik")
        print(f"  P95               : {it_stats['p95']:.3f} detik")
        print(f"  Rata-rata/file    : {it_stats['avg_per_file']:.3f} detik")
    if n_error > 0:
        print(f"\n  [WARNING] {n_error} file gagal diproses (tidak masuk metrik)")

    print(f"\n  Mulai  : {waktu_mulai.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Selesai: {waktu_selesai.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Durasi : {durasi_total/60:.1f} menit")

    def tulis_metrik_report(rf, label, m):
        rf.write(f"\n{'='*55}\n")
        rf.write(f"HASIL EVALUASI {label}\n")
        rf.write(f"{'='*55}\n")
        rf.write(f"  Accuracy  : {m['accuracy']:.4f}\n")
        rf.write(f"  Precision : {m['precision']:.4f}\n")
        rf.write(f"  Recall    : {m['recall']:.4f}\n")
        rf.write(f"  F1-Score  : {m['f1']:.4f}\n")
        cm = m['cm']
        rf.write(f"\n  Confusion Matrix:\n")
        rf.write(f"                Pred=0   Pred=1\n")
        rf.write(f"    Actual=0    TN={cm['tn']:4d}   FP={cm['fp']:4d}\n")
        rf.write(f"    Actual=1    FN={cm['fn']:4d}   TP={cm['tp']:4d}\n")
        rf.write(f"\n  Rincian per kelas:\n")
        for kelas, k in m['per_kelas'].items():
            rf.write(f"    [{kelas}] n={k['n']} | "
                     f"Benar={k['benar']} Salah={k['salah']} "
                     f"Acc={k['accuracy_kelas']:.2%} | "
                     f"TP={k['tp']} FN={k['fn']} FP={k['fp']} TN={k['tn']}\n")
            for sf in k['file_salah']:
                rf.write(f"      [SALAH] {sf['file']} "
                         f"(GT={sf['gt']}, Pred={sf['pred']})\n")

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as rf:
        rf.write("=" * 60 + "\n")
        rf.write("LAPORAN EVALUASI LENGKAP\n")
        rf.write(f"Konfigurasi : {KONFIGURASI}\n")
        rf.write(f"Model       : {MODEL_NAME}\n")
        rf.write(f"Skema       : One-Pass (1 API call per halaman)\n")
        rf.write(f"Prompting   : Zero-Shot (tanpa contoh)\n")
        rf.write(f"Dataset     : {DATASET_FOLDER}\n")
        rf.write(f"Total file  : {len(df)}\n")
        rf.write(f"File valid  : {len(df_valid)}\n")
        rf.write(f"File error  : {n_error}\n")
        rf.write(f"Mulai       : {waktu_mulai.strftime('%Y-%m-%d %H:%M:%S')}\n")
        rf.write(f"Selesai     : {waktu_selesai.strftime('%Y-%m-%d %H:%M:%S')}\n")
        rf.write(f"Durasi      : {durasi_total/60:.1f} menit\n")
        rf.write("=" * 60 + "\n")

        rf.write("\nRINGKASAN DATASET\n")
        rf.write("-" * 40 + "\n")
        for kelas, kunci in kunci_jawaban.items():
            sub = df[df['kelas'] == kelas]
            rf.write(f"  {kelas}: {len(sub)} file "
                     f"[GT Meterai={kunci['meterai_asli']}, "
                     f"GT TTD={kunci['ttd_asli']}]\n")

        if it_stats:
            rf.write("\nINFERENCE TIME STATISTICS\n")
            rf.write("-" * 40 + "\n")
            rf.write(f"  Total API calls   : {it_stats['total_calls']} (1 call/halaman)\n")
            rf.write(f"  Total waktu       : {it_stats['total_detik']:.2f} detik "
                     f"({it_stats['total_menit']:.1f} menit)\n")
            rf.write(f"  Rata-rata/call    : {it_stats['avg']:.3f} detik\n")
            rf.write(f"  Min               : {it_stats['min']:.3f} detik\n")
            rf.write(f"  Max               : {it_stats['max']:.3f} detik\n")
            rf.write(f"  P50 (median)      : {it_stats['p50']:.3f} detik\n")
            rf.write(f"  P90               : {it_stats['p90']:.3f} detik\n")
            rf.write(f"  P95               : {it_stats['p95']:.3f} detik\n")
            rf.write(f"  Rata-rata/file    : {it_stats['avg_per_file']:.3f} detik\n")

        tulis_metrik_report(rf, "METERAI",      m_meterai)
        tulis_metrik_report(rf, "TANDA TANGAN", m_ttd)

        rf.write(f"\n{'='*55}\n")
        rf.write("AKURASI GABUNGAN (kedua atribut benar sekaligus)\n")
        rf.write(f"{'='*55}\n")
        rf.write(f"  Benar keduanya : {n_keduanya_benar} / {len(df_valid)}\n")
        rf.write(f"  Accuracy       : {acc_gabungan:.4f}\n")

        rf.write(f"\n{'='*55}\n")
        rf.write("TABEL RINGKASAN METRIK\n")
        rf.write(f"{'='*55}\n")
        rf.write(f"{'Metrik':<20} {'Meterai':>10} {'Tanda Tangan':>14}\n")
        rf.write("-" * 46 + "\n")
        rf.write(f"{'Accuracy':<20} {m_meterai['accuracy']:>10.4f} {m_ttd['accuracy']:>14.4f}\n")
        rf.write(f"{'Precision':<20} {m_meterai['precision']:>10.4f} {m_ttd['precision']:>14.4f}\n")
        rf.write(f"{'Recall':<20} {m_meterai['recall']:>10.4f} {m_ttd['recall']:>14.4f}\n")
        rf.write(f"{'F1-Score':<20} {m_meterai['f1']:>10.4f} {m_ttd['f1']:>14.4f}\n")
        rf.write(f"{'Acc. Gabungan':<20} {acc_gabungan:>10.4f}\n")

    print(f"\n[INFO] Report disimpan: {OUTPUT_REPORT}")
    print("[SELESAI]")