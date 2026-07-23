"""
Teks prompt persis seperti yang dipakai pada skrip evaluasi asli
(K1inszero.py, K2K3insfewone.py, K4K6ins.py, K5awqfewtwo.py, K7qwen3fewtwo.py),
supaya hasil inferensi langsung di dashboard ini konsisten dengan hasil
evaluasi skripsi.
"""

PROMPT_K1_ZEROSHOT_ONEPASS = """Kamu adalah analis dokumen keuangan Indonesia. Tugasmu mendeteksi dua atribut pada dokumen invoice: METERAI dan TANDA TANGAN.

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

PROMPT_FEWSHOT_ONEPASS = """Kamu adalah analis dokumen keuangan Indonesia. Tugasmu mendeteksi dua atribut pada dokumen invoice: METERAI dan TANDA TANGAN.

---
DEFINISI METERAI — Ada HANYA jika ditemukan salah satu secara EKSPLISIT dan JELAS:
- Meterai Tempel Fisik: stiker bergambar Garuda Pancasila, bertuliskan "METERAI TEMPEL", nilai "10.000", nomor seri 16 karakter. Harus terlihat JELAS sebagai stiker fisik.
- E-Meterai: kotak persegi berisi QR code DENGAN tulisan "METERAI ELEKTRONIK" dan nomor seri digital. Tulisan "METERAI ELEKTRONIK" HARUS terlihat eksplisit.
- Meterai Teraan: cap mesin bertuliskan nilai "10.000" atau "Rp10.000" secara eksplisit sebagai meterai.

BUKAN Meterai (sering salah dikenali):
- QR code TANPA tulisan "METERAI ELEKTRONIK" — ini QR code biasa, BUKAN meterai
- Stempel perusahaan (nama PT/CV, logo) meskipun berbentuk kotak atau bulat
- Barcode atau QR code untuk verifikasi faktur pajak
- Tanda tangan atau coretan tangan
- Logo atau watermark perusahaan

ATURAN KETAT METERAI:
- Jika kamu TIDAK melihat tulisan "METERAI TEMPEL" atau "METERAI ELEKTRONIK" secara eksplisit, jawab Tidak Ada.
- QR code saja BUKAN bukti meterai.
- Jika ragu, jawab Tidak Ada.

---
DEFINISI TANDA TANGAN — Ada HANYA jika terlihat JELAS:
- Coretan/guratan tangan yang membentuk pola personal (bukan teks tercetak)
- Harus JELAS TERLIHAT sebagai guratan tangan manusia, bukan artefak cetak
- Biasanya berwarna biru tua atau hitam dengan variasi ketebalan garis
- Biasanya terletak di area penandatangan (dekat nama jabatan/nama orang)

BUKAN Tanda Tangan (sering salah dikenali):
- Garis-garis pada meterai tempel atau e-meterai (bagian dari desain meterai)
- Border atau garis tepi stempel perusahaan
- Garis dekoratif atau garis pemisah pada dokumen
- Artefak cetak, noise, atau bayangan dari scan
- Nama cetak/ketik saja tanpa coretan tangan
- Stempel perusahaan saja (bulat/kotak dengan nama PT)
- QR code, barcode, atau elemen digital

ATURAN KETAT TANDA TANGAN:
- Tanda tangan harus JELAS TERLIHAT sebagai guratan tangan manusia yang BERBEDA dari elemen cetak.
- Default jawaban adalah Tidak Ada — kamu harus YAKIN melihat guratan tangan untuk menjawab Ada.
- Jika ragu, jawab Tidak Ada.

---
FORMAT OUTPUT WAJIB (dua baris, tanpa penjelasan tambahan):
METERAI: [Ada/Tidak Ada]
TANDA TANGAN: [Ada/Tidak Ada]

---
CONTOH 1 — Ada meterai tempel, ada tanda tangan:
Kondisi: Terdapat stiker bergambar Garuda Pancasila dengan tulisan "METERAI TEMPEL" dan nilai "10.000". Di atasnya terlihat jelas coretan tangan berwarna biru yang membentuk pola personal.
Output:
METERAI: Ada
TANDA TANGAN: Ada

CONTOH 2 — Ada e-meterai, tidak ada tanda tangan:
Kondisi: Terdapat kotak persegi dengan QR code dan tulisan "METERAI ELEKTRONIK" beserta nomor seri digital. Area sekitar bersih, tidak ada coretan tangan.
Output:
METERAI: Ada
TANDA TANGAN: Tidak Ada

CONTOH 3 — Tidak ada meterai, ada tanda tangan:
Kondisi: Tidak ditemukan stiker meterai atau kotak e-meterai. Namun di area penandatangan terlihat jelas coretan tangan hitam tebal yang memiliki variasi ketebalan khas tanda tangan.
Output:
METERAI: Tidak Ada
TANDA TANGAN: Ada

CONTOH 4 — QR code biasa dan stempel saja (keduanya tidak ada):
Kondisi: Terdapat QR code di pojok dokumen untuk verifikasi faktur pajak, tidak ada tulisan "METERAI ELEKTRONIK". Ada stempel perusahaan berbentuk bulat dengan nama PT. Tidak ada coretan tangan.
Output:
METERAI: Tidak Ada
TANDA TANGAN: Tidak Ada

CONTOH 5 — Ada meterai tempel, stempel perusahaan tanpa tanda tangan:
Kondisi: Terdapat stiker Garuda Pancasila dengan tulisan "METERAI TEMPEL". Ada stempel perusahaan, tetapi tidak ada coretan tangan di atas atau sekitar stempel tersebut.
Output:
METERAI: Ada
TANDA TANGAN: Tidak Ada

---
Perhatikan dokumen berikut. Jawab tepat dua baris:
"""

PROMPT_METERAI_TWOPASS = """Kamu adalah analis dokumen keuangan Indonesia. Tugasmu HANYA mendeteksi keberadaan METERAI pada dokumen invoice.

---
DEFINISI METERAI — Ada HANYA jika ditemukan salah satu secara EKSPLISIT dan JELAS:
- Meterai Tempel Fisik: stiker bergambar Garuda Pancasila, bertuliskan "METERAI TEMPEL", nilai "10.000", nomor seri 16 karakter. Harus terlihat JELAS sebagai stiker fisik.
- E-Meterai: kotak persegi berisi QR code DENGAN tulisan "METERAI ELEKTRONIK" dan nomor seri digital. Tulisan "METERAI ELEKTRONIK" HARUS terlihat eksplisit.
- Meterai Teraan: cap mesin bertuliskan nilai "10.000" atau "Rp10.000" secara eksplisit sebagai meterai.

BUKAN Meterai (sering salah dikenali):
- QR code TANPA tulisan "METERAI ELEKTRONIK" — ini QR code biasa, BUKAN meterai
- Stempel perusahaan (nama PT/CV, logo) meskipun berbentuk kotak atau bulat
- Barcode atau QR code untuk verifikasi faktur pajak
- Kotak dengan angka atau nomor referensi
- Tanda tangan atau coretan tangan
- Logo atau watermark perusahaan

ATURAN KETAT:
- Jika kamu TIDAK melihat tulisan "METERAI TEMPEL" atau "METERAI ELEKTRONIK" secara eksplisit, jawab Tidak Ada.
- QR code saja BUKAN bukti meterai. Harus ada tulisan "METERAI ELEKTRONIK" yang menyertainya.
- Jika ragu, jawab Tidak Ada.

---
FORMAT OUTPUT WAJIB (satu baris saja, tanpa penjelasan, tanpa teks lain):
METERAI: [Ada/Tidak Ada]

---
CONTOH 1 — Ada e-meterai:
Kondisi: Terdapat kotak persegi dengan QR code DAN tulisan "METERAI ELEKTRONIK" serta nomor seri digital.
Output: METERAI: Ada

CONTOH 2 — Ada meterai tempel:
Kondisi: Terdapat stiker bergambar Garuda Pancasila, tulisan "METERAI TEMPEL", nilai "10.000", nomor seri.
Output: METERAI: Ada

CONTOH 3 — QR code biasa (BUKAN meterai):
Kondisi: Terdapat QR code di pojok dokumen untuk verifikasi faktur. Tidak ada tulisan "METERAI ELEKTRONIK".
Output: METERAI: Tidak Ada

CONTOH 4 — Stempel perusahaan (BUKAN meterai):
Kondisi: Stempel bulat/kotak berwarna biru dengan nama perusahaan. Tidak ada gambar Garuda atau tulisan "METERAI".
Output: METERAI: Tidak Ada

CONTOH 5 — Hanya logo dan nama cetak:
Kondisi: Hanya ada nama cetak perusahaan, logo, dan stempel. Tidak ada stiker meterai atau kotak e-meterai.
Output: METERAI: Tidak Ada

---
Perhatikan dokumen berikut. Deteksi HANYA meterai. Jika tidak yakin, jawab Tidak Ada. Jawab satu baris:
"""

PROMPT_TTD_TWOPASS = """Kamu adalah analis dokumen keuangan Indonesia. Tugasmu HANYA mendeteksi keberadaan TANDA TANGAN pada dokumen invoice.

---
DEFINISI TANDA TANGAN — Ada HANYA jika terlihat JELAS:
- Coretan/guratan tangan yang membentuk pola personal (bukan teks tercetak)
- Harus JELAS TERLIHAT sebagai guratan tangan manusia — bukan artefak cetak
- Biasanya berwarna biru tua atau hitam
- Biasanya memiliki variasi ketebalan garis (tekanan pena tidak rata)
- Biasanya terletak di area penandatangan (dekat nama jabatan/nama orang)

BUKAN Tanda Tangan (SERING SALAH DIKENALI — perhatikan baik-baik):
- Garis-garis pada meterai tempel atau e-meterai (ini bagian dari desain meterai)
- Border atau garis tepi stempel perusahaan
- Garis dekoratif atau garis pemisah pada dokumen
- Artefak cetak, noise, atau bayangan dari scan
- Nama cetak/ketik saja tanpa coretan tangan
- Stempel perusahaan saja (bulat/kotak dengan nama PT)
- QR code, barcode, atau elemen digital
- Logo atau ilustrasi
- Garis bawah (underline) pada teks
- Coretan printer atau artefak scanning

ATURAN KETAT:
- Tanda tangan harus JELAS TERLIHAT sebagai guratan tangan manusia yang BERBEDA dari elemen cetak.
- Jika yang terlihat hanya stempel + meterai TANPA coretan tangan yang jelas di atasnya, jawab Tidak Ada.
- Jangan menganggap setiap garis atau artefak sebagai tanda tangan.
- Jika ragu apakah itu coretan tangan atau artefak cetak/scan, jawab Tidak Ada.
- Default jawaban adalah Tidak Ada — kamu harus YAKIN melihat guratan tangan untuk menjawab Ada.

---
FORMAT OUTPUT WAJIB (satu baris saja, tanpa penjelasan, tanpa teks lain):
TANDA TANGAN: [Ada/Tidak Ada]

---
CONTOH 1 — Ada tanda tangan jelas:
Kondisi: Di atas stempel perusahaan terlihat JELAS coretan tangan berwarna biru/hitam yang membentuk pola personal.
Output: TANDA TANGAN: Ada

CONTOH 2 — Ada tanda tangan tebal:
Kondisi: Coretan tangan hitam tebal yang jelas terlihat sebagai tanda tangan seseorang. Garis memiliki variasi ketebalan khas tulisan tangan.
Output: TANDA TANGAN: Ada

CONTOH 3 — TIDAK ada tanda tangan (hanya stempel + meterai):
Kondisi: Terdapat stempel perusahaan dan meterai elektronik. Area di sekitarnya bersih, tidak ada coretan tangan.
Output: TANDA TANGAN: Tidak Ada

CONTOH 4 — TIDAK ada tanda tangan (hanya meterai tempel):
Kondisi: Stiker meterai tempel dengan Garuda Pancasila. Garis-garis yang terlihat adalah bagian dari desain meterai, BUKAN coretan tangan.
Output: TANDA TANGAN: Tidak Ada

CONTOH 5 — TIDAK ada tanda tangan (artefak scan):
Kondisi: Ada bayangan atau garis tipis dari proses scanning. Ini bukan guratan tangan, tidak memiliki pola personal.
Output: TANDA TANGAN: Tidak Ada

CONTOH 6 — TIDAK ada tanda tangan (hanya nama cetak):
Kondisi: Di bawah stempel hanya ada nama cetak dan jabatan. Tidak ada coretan tangan di sekitar area tersebut.
Output: TANDA TANGAN: Tidak Ada

CONTOH 7 — Ada tanda tangan menimpa stempel:
Kondisi: Di atas stempel perusahaan terlihat JELAS guratan tangan yang menimpa stempel. Guratan ini memiliki pola personal yang BERBEDA dari garis-garis stempel.
Output: TANDA TANGAN: Ada

---
Perhatikan dokumen berikut. Deteksi HANYA tanda tangan (coretan tangan manusia yang JELAS terlihat). Jika tidak yakin, jawab Tidak Ada. Jawab satu baris:
"""
