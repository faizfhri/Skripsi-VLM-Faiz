import streamlit as st

from utils import data as D
from utils import inference as I
from utils import theme as T

st.set_page_config(page_title="Coba Inferensi", layout="wide")
T.inject_base_css()

T.page_header(
    eyebrow="Coba Sendiri",
    title="Inferensi Langsung",
    subtitle=(
        "Pilih salah satu dari tujuh konfigurasi, unggah dokumen invoice dalam "
        "format PDF, lalu jalankan inferensi langsung ke server VLM untuk melihat "
        "hasil deteksi meterai dan tanda tangan secara nyata."
    ),
)

sel_col, info_col = st.columns([1, 1.6])
with sel_col:
    config_id = st.selectbox("Konfigurasi", D.CONFIG_ORDER, index=0, key="inference_config")
    uploaded_file = st.file_uploader("Unggah invoice (PDF)", type=["pdf"])
    run_clicked = st.button("Jalankan Inferensi", type="primary", disabled=uploaded_file is None)

with info_col:
    meta = D.resolve_config(config_id)
    endpoint = I.INFERENCE_ENDPOINTS[config_id]
    plate_rows = [
        ("Prompting", meta.prompting),
        ("Skema Inferensi", meta.inference),
        ("Model", meta.model),
        ("Endpoint Server", endpoint.url),
    ]
    st.markdown(T.config_plate(config_id, plate_rows), unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{T.INK_FAINT}; font-size:0.78rem; margin-top:0.5rem;">'
        "Endpoint di atas mengikuti alamat server pada skrip evaluasi asli. Bila "
        "server VLM-mu berjalan di alamat berbeda, ubah <code>INFERENCE_ENDPOINTS</code> "
        "di <code>utils/inference.py</code>."
        "</p>",
        unsafe_allow_html=True,
    )

if run_clicked and uploaded_file is not None:
    progress_bar = st.progress(0.0, text="Membaca berkas PDF...")

    def _on_progress(page_idx: int, n_pages: int) -> None:
        frac = page_idx / max(n_pages, 1)
        progress_bar.progress(frac, text=f"Memproses halaman {page_idx + 1} dari {n_pages}...")

    pdf_bytes = uploaded_file.getvalue()
    result = I.run_full_inference(config_id, pdf_bytes, progress_callback=_on_progress)
    progress_bar.progress(1.0, text="Selesai.")
    progress_bar.empty()

    st.session_state["inference_result"] = {
        "config_id": config_id,
        "filename": uploaded_file.name,
        "result": result,
    }

# Disimpan di session_state agar hasil tetap tampil walau halaman rerun karena interaksi lain (mis. expander).
saved = st.session_state.get("inference_result")

if saved is None:
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{T.INK_FAINT}; font-size:0.85rem;">'
        "Belum ada hasil inferensi. Unggah berkas PDF lalu klik \"Jalankan Inferensi\" di atas."
        "</p>",
        unsafe_allow_html=True,
    )
else:
    result = saved["result"]
    T.section_label(f"Hasil untuk \u201c{saved['filename']}\u201d \u2014 Konfigurasi {saved['config_id']}")

    if result.get("fatal_error"):
        st.markdown(
            f'<div class="panel" style="border-left:3px solid {T.ERROR};">'
            f'<span class="badge badge-error">Gagal</span>'
            f'<p style="margin-top:0.6rem; font-size:0.88rem; color:{T.INK};">{result["fatal_error"]}</p>'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        agg = result["aggregated"]

        verdict_cols = st.columns(4)
        with verdict_cols[0]:
            st.markdown(T.verdict_card("Meterai", agg["meterai_prediksi"] == 1), unsafe_allow_html=True)
        with verdict_cols[1]:
            st.markdown(T.verdict_card("Tanda Tangan", agg["ttd_prediksi"] == 1), unsafe_allow_html=True)
        with verdict_cols[2]:
            st.markdown(T.metric_card("Jumlah Halaman", str(result["n_pages"])), unsafe_allow_html=True)
        with verdict_cols[3]:
            st.markdown(T.metric_card("Total Waktu Inferensi", f"{result['total_elapsed']:.2f} d"), unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:{T.INK_FAINT}; font-size:0.78rem;">'
            "Hasil akhir memakai agregasi OR antar halaman: bila salah satu atribut "
            "terdeteksi \u201cAda\u201d pada halaman manapun, hasil akhir dokumen adalah \u201cAda\u201d "
            "\u2014 sama seperti aturan yang dipakai pada evaluasi skripsi."
            "</p>",
            unsafe_allow_html=True,
        )

        T.section_label("Rincian per Halaman")

        for page in result["pages"]:
            page_idx = page["page"]
            has_error = bool(page.get("error"))
            title = f"Halaman {page_idx}" + (" \u2014 gagal diproses" if has_error else "")
            with st.expander(title, expanded=has_error):
                col_img, col_text = st.columns([1, 1.1])
                with col_img:
                    st.image(page["image"])
                with col_text:
                    if has_error:
                        st.markdown(
                            f'<p style="color:{T.ERROR}; font-size:0.82rem;">{page["error"]}</p>',
                            unsafe_allow_html=True,
                        )
                    if page.get("raw_meterai") is not None:
                        st.markdown('<div class="cm-title">Keluaran Meterai</div>', unsafe_allow_html=True)
                        st.code(page["raw_meterai"], language=None)
                        st.markdown('<div class="cm-title">Keluaran Tanda Tangan</div>', unsafe_allow_html=True)
                        st.code(page["raw_ttd"], language=None)
                    else:
                        st.code(page["raw"], language=None)

                    parsed = page["parsed"]
                    st.markdown(
                        f'<span class="badge badge-accent">Meterai: {parsed["meterai_prediksi"]}</span>&nbsp;'
                        f'<span class="badge badge-accent">TTD: {parsed["ttd_prediksi"]}</span>&nbsp;'
                        + (
                            f'<span class="badge badge-muted">{page["elapsed"]:.2f} d</span>'
                            if page.get("elapsed") else ""
                        ),
                        unsafe_allow_html=True,
                    )
