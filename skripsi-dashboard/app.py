import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import data as D
from utils import theme as T
from utils.charts import apply_theme

st.set_page_config(
    page_title="Dashboard Evaluasi VLM - Meterai & Tanda Tangan",
    layout="wide",
    initial_sidebar_state="expanded",
)
T.inject_base_css()

T.page_header(
    eyebrow="Skripsi - Teknik Informatika, Universitas Padjadjaran",
    title="Klasifikasi Keberadaan Meterai dan Tanda Tangan pada Dokumen Invoice",
    subtitle=(
        "Evaluasi training-free menggunakan Vision-Language Model untuk mendeteksi "
        "keberadaan meterai dan tanda tangan pada dokumen invoice, dibandingkan pada "
        "tujuh konfigurasi melalui ablation study tiga tahap: strategi prompting, "
        "skema inferensi, dan pilihan model."
    ),
)

summary = D.all_configs_summary()
best_combined = summary.loc[summary["akurasi_gabungan"].idxmax()]
best_f1_meterai = summary.loc[summary["f1_meterai"].idxmax()]
best_f1_ttd = summary.loc[summary["f1_ttd"].idxmax()]
fastest = summary.loc[summary["rata_rata_per_file_detik"].idxmin()]

kelas_counts = D.load_csv("K1")["kelas"].value_counts().to_dict()
total_dokumen = sum(kelas_counts.values())

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        T.metric_card("Total Dokumen Diuji", f"{total_dokumen}", "4 kelas, dataset yang sama untuk tiap konfigurasi"),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        T.metric_card(
            "Akurasi Gabungan Terbaik", f"{best_combined['akurasi_gabungan']:.4f}",
            f"Konfigurasi {best_combined['id']}", highlight=True,
        ),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        T.metric_card(
            "F1 Meterai Terbaik", f"{best_f1_meterai['f1_meterai']:.4f}",
            f"Konfigurasi {best_f1_meterai['id']}", highlight=True,
        ),
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        T.metric_card(
            "Konfigurasi Tercepat", f"{fastest['rata_rata_per_file_detik']:.2f} d",
            f"Konfigurasi {fastest['id']}, rata-rata per file", highlight=True,
        ),
        unsafe_allow_html=True,
    )

T.section_label("Komposisi Dataset")

col_donut, col_desc = st.columns([1, 1.3], gap="large")
with col_donut:
    labels = [D.CLASS_INFO[k]["label"] for k in kelas_counts.keys()]
    values = list(kelas_counts.values())
    colors = [T.CLASS_COLORS.get(k, T.ACCENT) for k in kelas_counts.keys()]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels, values=values, hole=0.62,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+value", textfont=dict(family="IBM Plex Mono, monospace", size=12),
            )
        ]
    )
    apply_theme(fig, showlegend=False, height=280, annotations=[
        dict(text=f"{total_dokumen}<br><span style='font-size:11px;color:#5B6472'>dokumen</span>",
             x=0.5, y=0.5, showarrow=False, font=dict(size=22, family="IBM Plex Mono, monospace"))
    ])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col_desc:
    st.markdown('<div class="panel panel-accent">', unsafe_allow_html=True)
    for kelas_id, info in D.CLASS_INFO.items():
        n = kelas_counts.get(kelas_id, 0)
        dot_color = T.CLASS_COLORS.get(kelas_id, T.ACCENT)
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:0.6rem; padding:0.35rem 0;">
                <span style="width:10px; height:10px; border-radius:2px; background:{dot_color}; flex-shrink:0;"></span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.82rem; min-width:70px; color:{T.INK_MUTED};">{info['label']}</span>
                <span style="font-size:0.87rem;">{info['deskripsi']}</span>
                <span style="margin-left:auto; font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:{T.INK};">n = {n}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

T.section_label("Ringkasan Tujuh Konfigurasi")

st.markdown(
    '<p style="color:#5B6472; font-size:0.9rem; margin-top:-0.5rem;">'
    "K2/K3 dan K4/K6 berbagi data evaluasi yang sama karena keduanya identik dalam "
    "desain ablation study; ditampilkan sebagai baris terpisah agar posisi perbandingan tetap jelas."
    "</p>",
    unsafe_allow_html=True,
)

display_df = summary.copy()
display_df["Konfigurasi"] = display_df["id"]
display_df["Prompting"] = display_df["prompting"]
display_df["Inferensi"] = display_df["inference"]
display_df["Model"] = display_df["model"]
display_df["F1 Meterai"] = display_df["f1_meterai"].map(lambda x: f"{x:.4f}")
display_df["F1 TTD"] = display_df["f1_ttd"].map(lambda x: f"{x:.4f}")
display_df["Akurasi Gabungan"] = display_df["akurasi_gabungan"].map(lambda x: f"{x:.4f}")
display_df["Rata-rata/file (d)"] = display_df["rata_rata_per_file_detik"].map(lambda x: f"{x:.2f}")

st.dataframe(
    display_df[[
        "Konfigurasi", "Prompting", "Inferensi", "Model",
        "F1 Meterai", "F1 TTD", "Akurasi Gabungan", "Rata-rata/file (d)",
    ]],
    use_container_width=True,
    hide_index=True,
    height=282,
)

T.section_label("Jelajahi Dashboard")

nav_cols = st.columns(4)
nav_items = [
    ("Perbandingan Konfigurasi", "Grafik perbandingan metrik dan tiga tahap ablation study secara berdampingan."),
    ("Detail Konfigurasi", "Confusion matrix, rincian per kelas, dan statistik waktu inferensi untuk satu konfigurasi."),
    ("Analisis Kesalahan", "Telusuri file yang salah diklasifikasikan, termasuk keluaran mentah model per halaman."),
    ("Tentang Penelitian", "Metodologi, infrastruktur, dan referensi yang digunakan dalam penelitian."),
]
for col, (title, desc) in zip(nav_cols, nav_items):
    with col:
        st.markdown(
            f"""
            <div class="panel" style="height:150px;">
                <div style="font-family:'Source Serif 4',serif; font-weight:600; font-size:1.02rem; margin-bottom:0.4rem;">{title}</div>
                <div style="font-size:0.83rem; color:{T.INK_MUTED}; line-height:1.45;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    f"""
    <div style="margin-top:2.2rem; padding-top:1rem; border-top:1px solid {T.BORDER};
                font-size:0.78rem; color:{T.INK_FAINT};">
        Dataset bersumber dari dokumen invoice internal; nama perusahaan sumber dokumen
        tidak ditampilkan pada dashboard ini. Inferensi dijalankan pada NVIDIA DGX Spark
        (GB10 Grace Blackwell, 128GB unified memory) melalui API lokal.
    </div>
    """,
    unsafe_allow_html=True,
)
