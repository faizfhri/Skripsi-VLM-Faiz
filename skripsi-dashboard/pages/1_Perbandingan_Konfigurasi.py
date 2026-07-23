import plotly.graph_objects as go
import streamlit as st

from utils import data as D
from utils import theme as T
from utils.charts import apply_theme

st.set_page_config(page_title="Perbandingan Konfigurasi", layout="wide")
T.inject_base_css()

T.page_header(
    eyebrow="Bab IV - Hasil dan Pembahasan",
    title="Perbandingan Tujuh Konfigurasi",
    subtitle=(
        "Perbandingan metrik F1, akurasi gabungan, dan kecepatan inferensi di seluruh "
        "konfigurasi, disusun mengikuti tiga tahap ablation study pada skripsi."
    ),
)

summary = D.all_configs_summary()

T.section_label("F1-Score dan Akurasi Gabungan per Konfigurasi")

fig = go.Figure()
fig.add_bar(name="F1 Meterai", x=summary["id"], y=summary["f1_meterai"], marker_color=T.ACCENT)
fig.add_bar(name="F1 Tanda Tangan", x=summary["id"], y=summary["f1_ttd"], marker_color=T.CLASS_COLORS["kelas_2"])
fig.add_bar(name="Akurasi Gabungan", x=summary["id"], y=summary["akurasi_gabungan"], marker_color=T.GOLD)
apply_theme(fig, barmode="group", height=380, yaxis=dict(range=[0.75, 1.02], showgrid=True, gridcolor=T.BORDER))
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown(
    f'<p style="color:{T.INK_MUTED}; font-size:0.85rem; margin-top:-0.6rem;">'
    "Sumbu-y dipotong mulai 0.75 agar perbedaan antar konfigurasi lebih terlihat jelas."
    "</p>",
    unsafe_allow_html=True,
)

T.section_label("Kecepatan Inferensi Rata-Rata per File")

fig_speed = go.Figure()
fig_speed.add_bar(
    x=summary["id"], y=summary["rata_rata_per_file_detik"],
    marker_color=[T.CONFIG_COLORS.get(c, T.ACCENT) for c in summary["id"]],
    text=summary["rata_rata_per_file_detik"].map(lambda x: f"{x:.2f}s"),
    textposition="outside",
)
apply_theme(fig_speed, height=320, showlegend=False)
st.plotly_chart(fig_speed, use_container_width=True, config={"displayModeBar": False})

T.section_label("Ablation Study Tiga Tahap")

for stage in D.ABLATION_STAGES:
    st.markdown(
        f"""
        <div style="margin-top:1.1rem;">
            <span class="badge badge-accent">{stage['sub_bab']}</span>
            <span style="font-family:'Source Serif 4',serif; font-weight:600; font-size:1.15rem; margin-left:0.6rem;">{stage['judul']}</span>
        </div>
        <p style="color:{T.INK_MUTED}; font-size:0.88rem; max-width:820px; margin-top:0.4rem;">{stage['deskripsi']}</p>
        """,
        unsafe_allow_html=True,
    )

    members = stage["anggota"]
    sub = summary[summary["id"].isin(members)].set_index("id").loc[members].reset_index()

    cols = st.columns([1.1, 1])
    with cols[0]:
        fig_stage = go.Figure()
        fig_stage.add_bar(name="F1 Meterai", x=sub["id"], y=sub["f1_meterai"], marker_color=T.ACCENT)
        fig_stage.add_bar(name="F1 Tanda Tangan", x=sub["id"], y=sub["f1_ttd"], marker_color=T.CLASS_COLORS["kelas_2"])
        fig_stage.add_bar(name="Akurasi Gabungan", x=sub["id"], y=sub["akurasi_gabungan"], marker_color=T.GOLD)
        apply_theme(fig_stage, barmode="group", height=260, yaxis=dict(range=[0.75, 1.02]))
        st.plotly_chart(fig_stage, use_container_width=True, config={"displayModeBar": False}, key=f"stage-{stage['sub_bab']}")

    with cols[1]:
        row_fragments = []
        for _, row in sub.iterrows():
            row_fragments.append(
                '<div style="display:flex; justify-content:space-between; '
                f'padding:0.35rem 0; border-top:1px dashed {T.BORDER};">'
                '<span style="font-family:\'IBM Plex Mono\',monospace; font-size:0.82rem;">'
                f'{row["id"]}</span>'
                f'<span style="font-size:0.8rem; color:{T.INK_MUTED};">'
                f'{row["prompting"]} / {row["inference"]} / {row["model"]}</span>'
                '<span style="font-family:\'IBM Plex Mono\',monospace; font-size:0.82rem;">'
                f'{row["akurasi_gabungan"]:.4f}</span>'
                "</div>"
            )
        rows_html = "".join(row_fragments)
        st.markdown(f'<div class="panel">{rows_html}</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)

T.section_label("Tabel Metrik Lengkap")

full = summary.copy()
for col in ["f1_meterai", "f1_ttd", "acc_meterai", "acc_ttd", "prec_meterai", "prec_ttd",
            "rec_meterai", "rec_ttd", "akurasi_gabungan"]:
    full[col] = full[col].map(lambda x: f"{x:.4f}")
full["rata_rata_per_file_detik"] = full["rata_rata_per_file_detik"].map(lambda x: f"{x:.2f}")

full = full.rename(columns={
    "id": "Konfigurasi", "prompting": "Prompting", "inference": "Inferensi", "model": "Model",
    "prec_meterai": "Prec. Meterai", "rec_meterai": "Rec. Meterai", "f1_meterai": "F1 Meterai",
    "prec_ttd": "Prec. TTD", "rec_ttd": "Rec. TTD", "f1_ttd": "F1 TTD",
    "akurasi_gabungan": "Akurasi Gabungan", "rata_rata_per_file_detik": "Rata-rata/file (d)",
})

st.dataframe(
    full[[
        "Konfigurasi", "Prompting", "Inferensi", "Model",
        "Prec. Meterai", "Rec. Meterai", "F1 Meterai",
        "Prec. TTD", "Rec. TTD", "F1 TTD",
        "Akurasi Gabungan", "Rata-rata/file (d)",
    ]],
    use_container_width=True,
    hide_index=True,
    height=282,
)
