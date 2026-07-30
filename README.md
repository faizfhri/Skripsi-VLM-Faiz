# Implementasi Vision-Language Model untuk Klasifikasi Keberadaan Meterai dan Tanda Tangan pada Dokumen Invoice

Skripsi S1 Teknik Informatika yang menguji VLM open-source secara *training-free* untuk mendeteksi keberadaan meterai dan tanda tangan pada dokumen invoice, lewat ablation study tiga tahap yang menghasilkan lima eksperimen pengujian. Repo ini berisi skrip evaluasinya dan dashboard Streamlit untuk menjelajahi hasilnya.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.38%2B-FF4B4B)

## Daftar Isi

- [Deskripsi Proyek](#deskripsi-proyek)
- [Pemetaan Kode Eksperimen (Lama ke Baru)](#pemetaan-kode-eksperimen-lama-ke-baru)
- [Struktur Repository](#struktur-repository)
- [Fitur Dashboard](#fitur-dashboard)
- [Instalasi & Setup](#instalasi--setup)
- [Menjalankan Dashboard](#menjalankan-dashboard)
- [Menjalankan Skrip Evaluasi](#menjalankan-skrip-evaluasi)
- [Ringkasan Hasil Utama](#ringkasan-hasil-utama)
- [Tech Stack](#tech-stack)
- [Kredit](#kredit)
- [Lisensi](#lisensi)

## Deskripsi Proyek

Verifikasi dokumen invoice pada praktiknya mengecek dua hal: keberadaan meterai sebagai bukti bea, dan keberadaan tanda tangan sebagai bukti otorisasi. Proses ini umumnya dikerjakan manual, membuka tiap berkas PDF satu demi satu. Skripsi ini menguji kemampuan Vision-Language Model mengambil alih tugas tersebut tanpa proses pelatihan tambahan, murni melalui prompting terhadap model yang sudah tersedia.

Pendekatan training-free dipilih karena dua alasan. Dataset yang tersedia, 400 dokumen invoice internal perusahaan, terlalu kecil untuk fine-tuning. Penelitian ini juga ingin mengukur kemampuan out-of-the-box VLM open-source pada dokumen keuangan berbahasa Indonesia, bukan kemampuan model setelah dilatih ulang.

Evaluasi disusun sebagai ablation study tiga tahap. Tiap tahap mengubah satu variabel saja, supaya perbandingan antar konfigurasi tetap adil:

1. Strategi prompting: zero-shot dibandingkan few-shot tekstual, pada skema inferensi yang sama (K1 vs K2)
2. Skema inferensi: one-pass (meterai dan tanda tangan dideteksi dalam satu panggilan API) dibandingkan two-pass (dua panggilan terpisah), pada strategi prompting yang sama (K2 vs K3)
3. Pilihan model: model dasar dibandingkan varian kuantisasi AWQ dan generasi model yang lebih baru, pada kombinasi prompting dan skema inferensi terbaik (K3 vs K4 vs K5)

Ketiga tahap ini menghasilkan lima eksperimen pengujian:

| Eksperimen | Prompting | Skema Inferensi | Model |
|---|---|---|---|
| K1 | Zero-Shot | One-Pass | Qwen2.5-VL-7B-Instruct |
| K2 | Few-Shot Tekstual | One-Pass | Qwen2.5-VL-7B-Instruct |
| K3 | Few-Shot Tekstual | Two-Pass | Qwen2.5-VL-7B-Instruct |
| K4 | Few-Shot Tekstual | Two-Pass | Qwen2.5-VL-7B-AWQ |
| K5 | Few-Shot Tekstual | Two-Pass | Qwen3-VL-8B |

Dataset terbagi menjadi 4 kelas berdasarkan kombinasi ada/tidaknya kedua atribut:

| Kelas | Meterai | Tanda Tangan | Jumlah Dokumen |
|---|---|---|---|
| kelas_1 | Ada | Ada | 99 |
| kelas_2 | Tidak Ada | Ada | 101 |
| kelas_3 | Ada | Tidak Ada | 100 |
| kelas_4 | Tidak Ada | Tidak Ada | 100 |

## Struktur Repository

```
Skripsi-VLM-Faiz/
├── K1inszero.py              # Evaluasi K1: zero-shot, one-pass
├── K2insfewone.py            # Evaluasi K2: few-shot tekstual, one-pass
├── K3ins.py                   # Evaluasi K3: few-shot tekstual, two-pass
├── K4awqfewtwo.py             # Evaluasi K4: few-shot tekstual, two-pass, varian AWQ
├── K5qwen3fewtwo.py           # Evaluasi K5: few-shot tekstual, two-pass, Qwen3-VL-8B
├── dataset/                   # 400 dokumen invoice, dibagi 4 kelas ground truth
│   ├── kelas_1/
│   ├── kelas_2/
│   ├── kelas_3/
│   └── kelas_4/
├── hasil/                     # Output evaluasi mentah (CSV, JSON detail, laporan .txt)
└── skripsi-dashboard/         # Dashboard Streamlit untuk menjelajahi hasil
    ├── app.py                 # Halaman utama (ringkasan)
    ├── pages/
    │   ├── 1_Perbandingan_Konfigurasi.py
    │   ├── 2_Detail_Konfigurasi.py
    │   ├── 3_Analisis_Kesalahan.py
    │   ├── 4_Tentang_Penelitian.py
    │   └── 5_Coba_Inferensi.py
    ├── utils/                 # Loading data, metrik, tema visual, prompt, klien inferensi
    ├── data/                  # Salinan hasil evaluasi yang dipakai dashboard
    └── requirements.txt
```

Tiap file di `hasil/` dan `skripsi-dashboard/data/` datang dalam tiga bentuk: `.csv` (prediksi per dokumen), `_detail.json` (keluaran mentah model per halaman), dan `_report.txt` (laporan metrik lengkap plus daftar kesalahan klasifikasi).

## Fitur Dashboard

Dashboard punya enam halaman:

- **Ringkasan** (`app.py`): kartu metrik utama (total dokumen, akurasi gabungan terbaik, F1 meterai terbaik, konfigurasi tercepat), komposisi dataset per kelas, dan tabel ringkasan kelima eksperimen.
- **Perbandingan Konfigurasi**: grafik F1-score dan akurasi gabungan per konfigurasi, grafik kecepatan inferensi, serta breakdown tiga tahap ablation study berdampingan dengan tabel metrik lengkap (precision, recall, F1 untuk kedua atribut).
- **Detail Konfigurasi**: pilih satu konfigurasi untuk melihat confusion matrix, akurasi per kelas, dan statistik waktu inferensi (total call, rata-rata, min/p50/p90/max).
- **Analisis Kesalahan**: telusuri dokumen yang salah diklasifikasikan, dengan filter konfigurasi/atribut/kelas dan pencarian nama file. Untuk tiap dokumen, dashboard menampilkan pratinjau halaman PDF berdampingan dengan keluaran mentah model per halaman (butuh dataset lokal tersedia).
- **Tentang Penelitian**: ringkasan metodologi, spesifikasi perangkat pengembangan dan inferensi, daftar model yang dibandingkan, temuan kunci, dan daftar referensi.
- **Coba Inferensi**: unggah PDF invoice baru, pilih salah satu dari lima eksperimen, dan jalankan inferensi langsung ke server VLM lokal memakai prompt dan logika yang identik dengan skrip evaluasi aslinya.

## Instalasi & Setup

Dibutuhkan Python 3.10 ke atas (dikembangkan dan diuji dengan Python 3.12).

Dependencies untuk skrip evaluasi di root repo:

```bash
pip install pymupdf requests pandas tqdm scikit-learn
```

Dependencies untuk dashboard:

```bash
cd skripsi-dashboard
pip install -r requirements.txt
```

Skrip evaluasi dan halaman "Coba Inferensi" pada dashboard memanggil server VLM lokal berformat OpenAI-compatible (mis. vLLM) di `localhost:8001`-`8003`, tergantung model yang dipakai konfigurasi tersebut. Tanpa server ini keduanya tidak akan berfungsi, tapi lima halaman dashboard lainnya tetap bisa dijalankan karena datanya sudah tersedia di `skripsi-dashboard/data/`.

## Menjalankan Dashboard

```bash
cd skripsi-dashboard
streamlit run app.py
```

## Menjalankan Skrip Evaluasi

Tiap skrip membaca PDF dari folder dataset, mengonversi tiap halaman jadi gambar lewat PyMuPDF, mengirimkannya ke server VLM, mem-parsing jawaban, lalu menghitung metrik dan menulis tiga output (`.csv`, `_detail.json`, `_report.txt`).

| Skrip | Eksperimen | Model | Endpoint |
|---|---|---|---|
| `K1inszero.py` | K1 (zero-shot, one-pass) | Qwen2.5-VL-7B-Instruct | `localhost:8003` |
| `K2insfewone.py` | K2 (few-shot, one-pass) | Qwen2.5-VL-7B-Instruct | `localhost:8003` |
| `K3ins.py` | K3 (few-shot, two-pass) | Qwen2.5-VL-7B-Instruct | `localhost:8003` |
| `K4awqfewtwo.py` | K4 (few-shot, two-pass) | Qwen2.5-VL-7B-AWQ | `localhost:8002` |
| `K5qwen3fewtwo.py` | K5 (few-shot, two-pass) | Qwen3-VL-8B | `localhost:8001` |

Sebelum menjalankan, sesuaikan konstanta `DATASET_FOLDER` di bagian atas tiap skrip. Nilai defaultnya (`../datasetlengkap`) mengacu ke path di mesin pengembangan asli, bukan folder `dataset/` yang ada di repo ini.

```bash
python K1inszero.py
```

## Ringkasan Hasil Utama

Angka di bawah diambil langsung dari `_report.txt` masing-masing konfigurasi, dievaluasi pada 400 dokumen yang sama.

| Eksperimen | F1 Meterai | F1 Tanda Tangan | Akurasi Gabungan | Rata-rata/File |
|---|---|---|---|---|
| K1 | 0.9526 | 0.9191 | **0.8900** | 3.54 d |
| K2 | 0.9583 | 0.8786 | 0.8400 | 2.77 d |
| K3 | 0.9526 | 0.9091 | 0.8800 | 4.73 d |
| K4 | 0.9608 | 0.8859 | 0.8650 | **2.42 d** |
| K5 | **0.9975** | 0.8818 | 0.8775 | 3.96 d |

Beberapa temuan yang cukup mengejutkan dari hasil ini:

- **Zero-shot (K1) unggul pada akurasi gabungan**, mengalahkan semua varian few-shot walau tidak diberi contoh sama sekali, bertentangan dengan asumsi awal bahwa few-shot akan selalu lebih baik.
- **Few-shot one-pass (K2) menaikkan false positive tanda tangan** (31 kasus, jauh di atas 7 kasus pada K1), mengindikasikan contoh deskriptif dalam prompt bisa memperkenalkan bias, bukan cuma membantu.
- **Memisahkan meterai dan tanda tangan jadi dua panggilan terpisah (K3, two-pass) menaikkan precision tanda tangan** dari 0.8538 (K2, one-pass) ke 0.9770, pada strategi prompting few-shot yang sama.
- **Qwen3-VL-8B (K5) mendominasi deteksi meterai** (F1 0.9975) tapi tidak ikut terbaik di tanda tangan, menunjukkan performa antar atribut tidak selalu berjalan searah untuk model yang sama.
- **Varian AWQ (K4) adalah yang tercepat** sekaligus mempertahankan metrik yang kompetitif terhadap model dasarnya. Trade-off kuantisasi di sini terbilang murah.

Rincian lengkap tiap konfigurasi (termasuk confusion matrix, breakdown per kelas, dan daftar file yang salah diklasifikasikan) bisa dilihat di halaman **Detail Konfigurasi** dan **Analisis Kesalahan** pada dashboard.

## Tech Stack

- **Python**: bahasa utama skrip evaluasi dan dashboard
- **PyMuPDF (fitz)**: konversi halaman PDF ke gambar
- **Streamlit**: framework dashboard
- **Plotly**: visualisasi interaktif pada dashboard
- **Pandas**: pengolahan data hasil evaluasi
- **scikit-learn**: perhitungan metrik klasifikasi (accuracy, precision, recall, F1)
- **Qwen2.5-VL / Qwen3-VL**: model VLM yang dievaluasi, diakses lewat API lokal berformat OpenAI-compatible

## Kredit

**Penulis**
Muhammad Faiz Fahri (NPM 140810220002)

**Dosen Pembimbing**
1. Erick Paulus, S.Si., M.Kom.
2. Dr. Asep Sholahuddin, M.T.

Program Studi S1 Teknik Informatika, Universitas Padjadjaran
