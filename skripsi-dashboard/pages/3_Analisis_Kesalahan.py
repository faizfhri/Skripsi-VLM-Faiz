import streamlit as st

from utils import data as D
from utils import theme as T

st.set_page_config(page_title="Analisis Kesalahan", layout="wide")
T.inject_base_css()

T.page_header(
    eyebrow="Bab IV - Hasil dan Pembahasan",
    title="Analisis Kesalahan Klasifikasi",
    subtitle=(
        "Telusuri berkas yang salah diklasifikasikan pada tiap konfigurasi, lengkap "
        "dengan keluaran mentah model dan pratinjau dokumen invoice asli per halaman."
    ),
)

filter_cols = st.columns([1, 1, 1.4])
with filter_cols[0]:
    config_id = st.selectbox("Konfigurasi", D.CONFIG_ORDER, index=0)
with filter_cols[1]:
    atribut = st.radio("Atribut", ["Meterai", "Tanda Tangan"], horizontal=True)

metrics = D.compute_metrics(config_id)

if atribut == "Meterai":
    mis_df = metrics["misclassified_meterai"].copy()
    col_true, col_pred = "meterai_asli", "meterai_prediksi"
else:
    mis_df = metrics["misclassified_ttd"].copy()
    col_true, col_pred = "ttd_asli", "ttd_prediksi"

mis_df = mis_df.rename(columns={
    "nama_file": "Nama File", "kelas_label": "Kelas",
    col_true: "Ground Truth", col_pred: "Prediksi",
})
# Kolom "kelas" (id folder mentah, mis. kelas_1) sengaja tidak diganti nama supaya
# tetap bisa dipakai untuk menyusun path berkas PDF, tapi disembunyikan dari tabel.

with filter_cols[2]:
    kelas_opsi = sorted(mis_df["Kelas"].unique().tolist())
    kelas_pilih = st.multiselect("Filter kelas", kelas_opsi, default=kelas_opsi)

mis_df = mis_df[mis_df["Kelas"].isin(kelas_pilih)] if kelas_pilih else mis_df.iloc[0:0]

search = st.text_input("Cari nama file", "")
if search:
    mis_df = mis_df[mis_df["Nama File"].str.contains(search, case=False, na=False)]

total_n = metrics["n_total"]
salah_n = len(metrics["misclassified_meterai"]) if atribut == "Meterai" else len(metrics["misclassified_ttd"])

T.section_label(f"Berkas Salah Klasifikasi \u2014 {atribut}")

st.markdown(
    f'<span class="badge badge-error">{salah_n} dari {total_n} salah</span>&nbsp;&nbsp;'
    f'<span class="badge badge-muted">{len(mis_df)} ditampilkan setelah filter</span>',
    unsafe_allow_html=True,
)
st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

display_cols = ["Nama File", "Kelas", "Ground Truth", "Prediksi"]
st.dataframe(mis_df[display_cols], use_container_width=True, hide_index=True, height=280)

T.section_label("Detail Berkas")

if mis_df.empty:
    st.markdown(
        f'<p style="color:{T.INK_MUTED}; font-size:0.88rem;">Tidak ada berkas pada filter saat ini.</p>',
        unsafe_allow_html=True,
    )
else:
    pilihan_file = st.selectbox("Pilih berkas untuk melihat detail", mis_df["Nama File"].tolist())
    detail = D.find_file_detail(config_id, pilihan_file)
    row = mis_df[mis_df["Nama File"] == pilihan_file].iloc[0]

    info_cols = st.columns(4)
    with info_cols[0]:
        st.markdown(T.metric_card("Kelas", row["Kelas"]), unsafe_allow_html=True)
    with info_cols[1]:
        st.markdown(T.metric_card("Ground Truth", str(row["Ground Truth"])), unsafe_allow_html=True)
    with info_cols[2]:
        st.markdown(T.metric_card("Prediksi Akhir", str(row["Prediksi"])), unsafe_allow_html=True)
    with info_cols[3]:
        n_halaman = len(detail.get("page_by_page", [])) if detail else 0
        st.markdown(T.metric_card("Jumlah Halaman", str(n_halaman)), unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

    # Path PDF dicari otomatis di DATASET_ROOT/<kelas>/<nama_file>; sesuaikan DATASET_ROOT di utils/data.py bila struktur foldermu berbeda.
    pdf_path = D.DATASET_ROOT / row["kelas"] / pilihan_file
    pdf_module_ok = D.pdf_support_available()
    pdf_ready = pdf_module_ok and pdf_path.exists()

    if not pdf_module_ok:
        st.markdown(
            f'<p style="color:{T.ERROR}; font-size:0.82rem;">'
            "Pratinjau PDF memerlukan paket PyMuPDF, yang belum terpasang. Jalankan "
            "<code>pip install pymupdf</code> pada environment yang sama lalu muat ulang halaman."
            "</p>",
            unsafe_allow_html=True,
        )
    elif not pdf_path.exists():
        st.markdown(
            f'<p style="color:{T.INK_FAINT}; font-size:0.8rem;">'
            f"Pratinjau PDF tidak ditemukan di <code>{pdf_path}</code>. Dashboard mencari "
            f"berkas secara otomatis di <code>{D.DATASET_ROOT}/&lt;kelas&gt;/&lt;nama_file&gt;</code> "
            "relatif terhadap folder aplikasi ini."
            "</p>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="cm-title">Keluaran Model per Halaman</div>', unsafe_allow_html=True)

    if detail is None:
        st.markdown(
            f'<p style="color:{T.INK_MUTED}; font-size:0.88rem;">Detail keluaran model tidak ditemukan untuk berkas ini.</p>',
            unsafe_allow_html=True,
        )
    else:
        for page in detail.get("page_by_page", []):
            page_idx = page.get("page", 0)
            raw = page.get("raw", "")
            parsed = page.get("parsed", {})

            with st.expander(f"Halaman {page_idx}"):
                if pdf_ready:
                    col_img, col_text = st.columns([1, 1.1])
                    with col_img:
                        status, payload = D.render_pdf_page(str(pdf_path), page_idx)
                        if status == "ok":
                            st.image(payload["image"])
                        elif status == "error":
                            st.markdown(
                                f'<p style="color:{T.ERROR}; font-size:0.78rem;">Gagal merender halaman: {payload}</p>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<p style="color:{T.INK_FAINT}; font-size:0.78rem;">Halaman tidak dapat ditampilkan.</p>',
                                unsafe_allow_html=True,
                            )
                    with col_text:
                        st.code(raw, language=None)
                        st.markdown(
                            f'<span class="badge badge-accent">Meterai: {parsed.get("meterai_prediksi", "-")}</span>&nbsp;'
                            f'<span class="badge badge-accent">TTD: {parsed.get("ttd_prediksi", "-")}</span>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.code(raw, language=None)
                    st.markdown(
                        f'<span class="badge badge-accent">Meterai: {parsed.get("meterai_prediksi", "-")}</span>&nbsp;'
                        f'<span class="badge badge-accent">TTD: {parsed.get("ttd_prediksi", "-")}</span>',
                        unsafe_allow_html=True,
                    )

        agg = detail.get("aggregated", {})
        st.markdown(
            '<p style="color:' + T.INK_FAINT + '; font-size:0.78rem; margin-top:0.8rem;">'
            f'Hasil agregasi akhir: meterai = {agg.get("meterai_prediksi", "-")}, '
            f'tanda tangan = {agg.get("ttd_prediksi", "-")}. Agregasi dilakukan mengikuti '
            f'aturan gabungan antar halaman pada skrip evaluasi konfigurasi {config_id}.'
            "</p>",
            unsafe_allow_html=True,
        )
