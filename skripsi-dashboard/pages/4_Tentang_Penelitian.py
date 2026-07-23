import streamlit as st

from utils import data as D
from utils import theme as T

st.set_page_config(page_title="Tentang Penelitian", layout="wide")
T.inject_base_css()

T.page_header(
    eyebrow="Metodologi",
    title="Tentang Penelitian",
    subtitle=(
        "Ringkasan metodologi, infrastruktur, dan referensi yang digunakan dalam "
        "penelitian klasifikasi keberadaan meterai dan tanda tangan berbasis VLM."
    ),
)

T.section_label("Ringkasan Metodologi")

st.markdown(
    f"""
    <div class="panel panel-accent">
    <p style="font-size:0.92rem; line-height:1.7; margin:0;">
    Penelitian ini bersifat <i>training-free</i>: model Vision-Language Model (VLM)
    tidak dilatih ulang, melainkan digunakan langsung melalui prompting untuk menilai
    keberadaan meterai dan tanda tangan pada 400 dokumen invoice. Dataset terbagi
    menjadi empat kelas berdasarkan kombinasi ada/tidaknya kedua atribut tersebut.
    Perbandingan dilakukan melalui ablation study tiga tahap yang menghasilkan tujuh
    konfigurasi pengujian, mengubah satu variabel pada satu waktu: strategi prompting,
    skema inferensi, dan pilihan model.
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

for stage in D.ABLATION_STAGES:
    st.markdown(
        f"""
        <div class="panel" style="margin-bottom:0.7rem;">
            <span class="badge badge-accent">{stage['sub_bab']}</span>
            <span style="font-family:'Source Serif 4',serif; font-weight:600; font-size:1rem; margin-left:0.6rem;">{stage['judul']}</span>
            <p style="font-size:0.85rem; color:{T.INK_MUTED}; margin:0.5rem 0 0 0; line-height:1.6;">{stage['deskripsi']}</p>
            <p style="font-size:0.8rem; color:{T.INK_FAINT}; margin:0.4rem 0 0 0;">Konfigurasi terkait: {', '.join(stage['anggota'])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

T.section_label("Infrastruktur")

infra_col1, infra_col2 = st.columns(2)
with infra_col1:
    st.markdown(
        f"""
        <div class="panel">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:{T.INK_MUTED}; margin-bottom:0.6rem;">Perangkat Pengembangan (Laptop)</div>
        <ol style="font-size:0.88rem; line-height:1.8; margin:0; padding-left:1.2rem;">
            <li>Processor: Intel Core i5-11400H</li>
            <li>GPU: NVIDIA RTX 3050 Ti</li>
            <li>RAM: 16GB</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
with infra_col2:
    st.markdown(
        f"""
        <div class="panel">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:{T.INK_MUTED}; margin-bottom:0.6rem;">Perangkat Inferensi</div>
        <ol style="font-size:0.88rem; line-height:1.8; margin:0; padding-left:1.2rem;">
            <li>NVIDIA DGX Spark (GB10 Grace Blackwell)</li>
            <li>128GB unified memory</li>
            <li>Diakses via HTTP API pada localhost:8001</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="panel">
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:{T.INK_MUTED}; margin-bottom:0.6rem;">Model yang Dibandingkan</div>
    <ol style="font-size:0.88rem; line-height:1.8; margin:0; padding-left:1.2rem;">
        <li>Qwen2.5-VL-7B-Instruct &mdash; model dasar</li>
        <li>Qwen2.5-VL-7B-AWQ &mdash; varian terkuantisasi dari model dasar</li>
        <li>Qwen3-VL-8B &mdash; generasi model yang lebih baru</li>
    </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

T.section_label("Temuan Kunci")

findings = [
    ("Zero-shot unggul pada akurasi gabungan",
     "Konfigurasi K1 (zero-shot) mencatat akurasi gabungan tertinggi (0.8900), hasil yang "
     "tidak sejalan dengan asumsi awal bahwa few-shot akan selalu lebih baik."),
    ("Few-shot one-pass menurunkan deteksi tanda tangan",
     "K2/K3 (few-shot, one-pass) menghasilkan 31 false positive pada tanda tangan, jauh "
     "lebih banyak dibanding 7 false positive pada K1, mengindikasikan few-shot dapat "
     "memperkenalkan bias."),
    ("Two-pass memperbaiki presisi tanda tangan",
     "K4/K6 (two-pass) mencatat precision tanda tangan 0.9770, dibandingkan 0.8538 pada "
     "K2/K3 (one-pass), pada strategi prompting few-shot yang sama."),
    ("Qwen3-VL dominan pada meterai, lebih lemah pada tanda tangan",
     "K7 mencatat F1 meterai tertinggi (0.9975), namun performa tanda tangannya tidak "
     "menjadi yang terbaik, menunjukkan trade-off yang bergantung pada model."),
    ("AWQ menawarkan kecepatan dengan performa kompetitif",
     "K5 (varian AWQ) merupakan konfigurasi tercepat dalam pengujian sambil tetap "
     "mempertahankan metrik yang kompetitif terhadap model dasarnya."),
]

for title, desc in findings:
    st.markdown(
        f"""
        <div class="panel" style="margin-bottom:0.6rem; border-left:3px solid {T.ACCENT};">
        <div style="font-weight:600; font-size:0.92rem; margin-bottom:0.3rem;">{title}</div>
        <div style="font-size:0.85rem; color:{T.INK_MUTED}; line-height:1.6;">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

T.section_label("Referensi Utama")

references = [
    "Zhang, J., Huang, J., Jin, S., & Lu, S. (2024). Vision-Language Models for Vision Tasks: A Survey. IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI).",
    "Bai, S., Cai, Y., Chen, R., et al. (2025). Qwen3-VL Technical Report. arXiv.",
    "Al Rahhal, M. M., Bazi, Y., Elgibreen, H., & Zuair, M. (2023). Vision-Language Models for Zero-Shot Classification of Remote Sensing Images. Applied Sciences, 13(22), 12462.",
    "Saha, S., Kulinski, A., Poudel, S., et al. (2024). Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).",
    "Boonstra, L. (2025). Prompt Engineering. Google Cloud.",
    "Ye, Q., Axmed, M., Pryzant, R., & Khani, F. (2024). Prompt Engineering a Prompt Engineer. Findings of the Association for Computational Linguistics: ACL 2024, 355\u2013385.",
    "Vujovi\u0107, \u017d. \u0110. (2021). Classification Model Evaluation Metrics. International Journal of Advanced Computer Science and Applications (IJACSA), 12(6), 599\u2013606.",
    "Cutting, G. A., & Cutting-Decelle, A.-F. (2021). Intelligent Document Processing \u2013 Methods and Tools in the Real World. arXiv Preprint.",
    "Ojika, F. U., Owobu, W. O., Abieba, O. A., et al. (2024). The Role of Artificial Intelligence in Business Process Automation: A Model for Reducing Operational Costs and Enhancing Efficiency. International Journal of Advanced Multidisciplinary Research and Studies, 4(6), 1449\u20131462.",
]

ref_html = "".join(
    f'<li style="margin-bottom:0.55rem; font-size:0.85rem; line-height:1.55; color:{T.INK};">{r}</li>'
    for r in references
)
st.markdown(f'<div class="panel"><ol style="padding-left:1.2rem; margin:0;">{ref_html}</ol></div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <p style="font-size:0.8rem; color:{T.INK_FAINT}; margin-top:0.8rem;">
    Catatan: daftar pustaka skripsi masih dalam proses finalisasi. Referensi mengenai
    DECOMP (Khot et al., 2023) sebagai justifikasi skema two-pass, serta laporan teknis
    Qwen2.5-VL, digunakan dalam pembahasan tetapi belum tercantum lengkap pada berkas
    referensi ini &mdash; mohon diverifikasi kembali sitasi persisnya sebelum sidang.
    </p>
    """,
    unsafe_allow_html=True,
)
