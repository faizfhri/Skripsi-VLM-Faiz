import plotly.graph_objects as go
import streamlit as st

from utils import data as D
from utils import theme as T
from utils.charts import apply_theme

st.set_page_config(page_title="Detail Konfigurasi", layout="wide")
T.inject_base_css()

T.page_header(
    eyebrow="Bab IV - Hasil dan Pembahasan",
    title="Detail per Konfigurasi",
    subtitle="Confusion matrix, rincian per kelas, dan statistik waktu inferensi untuk satu konfigurasi terpilih.",
)

config_id = st.selectbox("Pilih konfigurasi", D.CONFIG_ORDER, index=0)
meta = D.resolve_config(config_id)
metrics = D.compute_metrics(config_id)
timing = D.parse_timing_stats(config_id)

plate_rows = [
    ("Prompting", meta.prompting),
    ("Skema Inferensi", meta.inference),
    ("Model", meta.model),
    ("Jumlah Dokumen", str(metrics["n_total"])),
]
col_plate, col_note = st.columns([1, 2])
with col_plate:
    st.markdown(T.config_plate(config_id, plate_rows), unsafe_allow_html=True)
with col_note:
    if meta.duplicate_of:
        st.markdown(
            f"""
            <div class="panel" style="border-left:3px solid {T.GOLD};">
                <span class="badge badge-gold">Data Bersama</span>
                <p style="margin-top:0.5rem; font-size:0.85rem; color:{T.INK_MUTED};">
                {meta.catatan} Angka pada halaman ini identik dengan konfigurasi
                <b>{meta.duplicate_of}</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="panel">
                <p style="font-size:0.85rem; color:{T.INK_MUTED}; margin:0;">
                Seluruh metrik pada halaman ini dihitung langsung dari berkas CSV hasil
                prediksi untuk konfigurasi {config_id}, dibandingkan terhadap label ground
                truth pada empat kelas dataset.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

T.section_label("Ringkasan Metrik")

cols = st.columns(5)
cards = [
    ("F1 Meterai", f"{metrics['meterai']['f1']:.4f}", f"Precision {metrics['meterai']['precision']:.4f}"),
    ("F1 Tanda Tangan", f"{metrics['ttd']['f1']:.4f}", f"Precision {metrics['ttd']['precision']:.4f}"),
    ("Akurasi Meterai", f"{metrics['meterai']['accuracy']:.4f}", f"Recall {metrics['meterai']['recall']:.4f}"),
    ("Akurasi TTD", f"{metrics['ttd']['accuracy']:.4f}", f"Recall {metrics['ttd']['recall']:.4f}"),
    ("Akurasi Gabungan", f"{metrics['combined_accuracy']:.4f}", f"{metrics['combined_benar']} / {metrics['n_total']} benar"),
]
for col, (label, value, note) in zip(cols, cards):
    with col:
        st.markdown(T.metric_card(label, value, note), unsafe_allow_html=True)

T.section_label("Confusion Matrix")

cm_col1, cm_col2 = st.columns(2)
with cm_col1:
    m = metrics["meterai"]
    st.markdown(
        T.confusion_matrix_html("Meterai", m["tp"], m["fp"], m["fn"], m["tn"]),
        unsafe_allow_html=True,
    )
with cm_col2:
    t = metrics["ttd"]
    st.markdown(
        T.confusion_matrix_html("Tanda Tangan", t["tp"], t["fp"], t["fn"], t["tn"]),
        unsafe_allow_html=True,
    )

T.section_label("Rincian per Kelas")

kelas_col1, kelas_col2 = st.columns(2)


def render_kelas_chart(df, attribut_label, color):
    fig = go.Figure()
    fig.add_bar(
        x=df["label"], y=df["akurasi"], marker_color=color,
        text=df["akurasi"].map(lambda x: f"{x:.2%}"), textposition="outside",
    )
    apply_theme(fig, height=270, yaxis=dict(range=[0, 1.15]), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"kelas-{attribut_label}-{config_id}")


with kelas_col1:
    st.markdown(f'<div class="cm-title">Akurasi per Kelas &ndash; Meterai</div>', unsafe_allow_html=True)
    render_kelas_chart(metrics["meterai_per_kelas"], "meterai", T.ACCENT)
    st.dataframe(
        metrics["meterai_per_kelas"][["label", "deskripsi", "n", "benar", "salah"]].rename(
            columns={"label": "Kelas", "deskripsi": "Deskripsi", "n": "n", "benar": "Benar", "salah": "Salah"}
        ),
        use_container_width=True, hide_index=True,
    )

with kelas_col2:
    st.markdown(f'<div class="cm-title">Akurasi per Kelas &ndash; Tanda Tangan</div>', unsafe_allow_html=True)
    render_kelas_chart(metrics["ttd_per_kelas"], "ttd", T.CLASS_COLORS["kelas_2"])
    st.dataframe(
        metrics["ttd_per_kelas"][["label", "deskripsi", "n", "benar", "salah"]].rename(
            columns={"label": "Kelas", "deskripsi": "Deskripsi", "n": "n", "benar": "Benar", "salah": "Salah"}
        ),
        use_container_width=True, hide_index=True,
    )

T.section_label("Statistik Waktu Inferensi")

t_cols = st.columns(4)
timing_cards = [
    ("Total API Call", f"{int(timing['total_api_calls']) if timing['total_api_calls'] else '-'}"),
    ("Total Waktu", f"{timing['total_waktu_detik']:.1f} d" if timing["total_waktu_detik"] else "-"),
    ("Rata-rata / Call", f"{timing['rata_rata_per_call']:.3f} d" if timing["rata_rata_per_call"] else "-"),
    ("Rata-rata / File", f"{timing['rata_rata_per_file']:.3f} d" if timing["rata_rata_per_file"] else "-"),
]
for col, (label, value) in zip(t_cols, timing_cards):
    with col:
        st.markdown(T.metric_card(label, value), unsafe_allow_html=True)

st.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)

t_cols2 = st.columns(4)
timing_cards2 = [
    ("Min", f"{timing['min_detik']:.3f} d" if timing["min_detik"] else "-"),
    ("P50 (median)", f"{timing['p50_detik']:.3f} d" if timing["p50_detik"] else "-"),
    ("P90", f"{timing['p90_detik']:.3f} d" if timing["p90_detik"] else "-"),
    ("Max", f"{timing['max_detik']:.3f} d" if timing["max_detik"] else "-"),
]
for col, (label, value) in zip(t_cols2, timing_cards2):
    with col:
        st.markdown(T.metric_card(label, value), unsafe_allow_html=True)
