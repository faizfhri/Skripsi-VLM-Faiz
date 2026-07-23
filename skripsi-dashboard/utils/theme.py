import streamlit as st

BG_PAGE = "#F5F6F3"
BG_PANEL = "#FFFFFF"
BG_PANEL_ALT = "#F0F2EE"
BORDER = "#DDE1DA"
INK = "#1B2430"
INK_MUTED = "#5B6472"
INK_FAINT = "#8A93A0"

ACCENT = "#14636B"          # teal utama
ACCENT_DARK = "#0D4A50"
ACCENT_TINT = "#E4EEEE"
ACCENT_TINT_STRONG = "#CBE0E1"

GOLD = "#B8863E"            # brass/amber untuk penanda "terbaik"
GOLD_TINT = "#F6EDDC"

ERROR = "#B3492F"
ERROR_TINT = "#F5E4DE"
SUCCESS = "#3C7A5D"
SUCCESS_TINT = "#E4EFE8"

CLASS_COLORS = {
    "kelas_1": "#14636B",
    "kelas_2": "#5FA0A6",
    "kelas_3": "#B8863E",
    "kelas_4": "#B7AF9C",
}

CONFIG_COLORS = {
    "K1": "#14636B",
    "K2": "#3E8E94",
    "K3": "#3E8E94",
    "K4": "#7EB2B4",
    "K5": "#B8863E",
    "K6": "#7EB2B4",
    "K7": "#8A4B3B",
}

FONT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600;8..60,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
"""

BASE_CSS = """
<style>
%(fonts)s

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

.stApp {
    background-color: %(bg_page)s;
}

header[data-testid="stHeader"] {
    display: none;
}

section[data-testid="stSidebar"] {
    background-color: %(bg_panel)s;
    border-right: 1px solid %(border)s;
}

section[data-testid="stSidebar"] .stMarkdown p {
    color: %(ink_muted)s;
}

h1, h2, h3 {
    font-family: 'Source Serif 4', Georgia, serif !important;
    color: %(ink)s !important;
    letter-spacing: -0.01em;
}

h1 { font-weight: 600 !important; }
h2 { font-weight: 600 !important; }
h3 { font-weight: 600 !important; }

p, li, span, label, div {
    color: %(ink)s;
}

.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
    max-width: 1180px;
}

hr {
    border: none;
    border-top: 1px solid %(border)s;
    margin: 1.6rem 0;
}

/* ---- Kartu / panel umum ---- */
.panel {
    background-color: %(bg_panel)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
    padding: 1.3rem 1.5rem;
}

.panel-accent {
    border-left: 3px solid %(accent)s;
}

/* ---- Header halaman ---- */
.page-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: %(accent)s;
    margin-bottom: 0.3rem;
}

.page-title {
    font-family: 'Source Serif 4', serif;
    font-size: 2.05rem;
    font-weight: 600;
    color: %(ink)s;
    margin-bottom: 0.35rem;
    line-height: 1.15;
}

.page-subtitle {
    color: %(ink_muted)s;
    font-size: 1rem;
    max-width: 760px;
    line-height: 1.55;
    margin-bottom: 0.4rem;
}

/* ---- Kartu metrik (readout) ---- */
.metric-card {
    background-color: %(bg_panel)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
    padding: 1rem 1.1rem;
    height: 100%%;
}

.metric-card .metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: %(ink_muted)s;
    margin-bottom: 0.45rem;
}

.metric-card .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    color: %(ink)s;
    line-height: 1;
}

.metric-card .metric-note {
    font-size: 0.78rem;
    color: %(ink_faint)s;
    margin-top: 0.4rem;
}

.metric-card.highlight {
    border-color: %(gold)s;
    background-color: %(gold_tint)s;
}

.metric-card.highlight .metric-value { color: %(accent_dark)s; }

/* ---- Kartu konfigurasi (nameplate) ---- */
.config-plate {
    background-color: %(bg_panel)s;
    border: 1px solid %(border)s;
    border-left: 4px solid %(accent)s;
    border-radius: 4px;
    padding: 0.95rem 1.15rem;
    height: 100%%;
}

.config-plate .config-id {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    font-weight: 600;
    color: %(accent_dark)s;
    margin-bottom: 0.5rem;
}

.config-plate .config-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    padding: 0.18rem 0;
    border-top: 1px dashed %(border)s;
}

.config-plate .config-row:first-of-type { border-top: none; }

.config-plate .config-row .k { color: %(ink_faint)s; }
.config-plate .config-row .v { color: %(ink)s; font-weight: 500; text-align: right; }

/* ---- Badge ---- */
.badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.04em;
    padding: 0.15rem 0.55rem;
    border-radius: 3px;
    font-weight: 500;
}

.badge-accent { background-color: %(accent_tint)s; color: %(accent_dark)s; }
.badge-gold { background-color: %(gold_tint)s; color: %(gold)s; }
.badge-error { background-color: %(error_tint)s; color: %(error)s; }
.badge-success { background-color: %(success_tint)s; color: %(success)s; }
.badge-muted { background-color: %(bg_panel_alt)s; color: %(ink_muted)s; }

/* ---- Confusion matrix ---- */
.cm-wrap { display: flex; flex-direction: column; gap: 2px; margin-top: 0.6rem; }
.cm-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: %(ink_muted)s; margin-bottom: 0.3rem; text-transform: uppercase; letter-spacing: 0.06em; }
.cm-row { display: flex; gap: 2px; }
.cm-cell {
    flex: 1;
    text-align: center;
    padding: 0.7rem 0.4rem;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
}
.cm-cell .cm-value { font-size: 1.35rem; font-weight: 600; display: block; }
.cm-cell .cm-label { font-size: 0.65rem; color: %(ink_muted)s; text-transform: uppercase; letter-spacing: 0.04em; }
.cm-cell.tp, .cm-cell.tn { background-color: %(success_tint)s; }
.cm-cell.tp .cm-value, .cm-cell.tn .cm-value { color: %(success)s; }
.cm-cell.fp, .cm-cell.fn { background-color: %(error_tint)s; }
.cm-cell.fp .cm-value, .cm-cell.fn .cm-value { color: %(error)s; }
.cm-axis { font-size: 0.68rem; color: %(ink_faint)s; text-align: center; margin-top: 0.25rem; }

/* ---- Section label ---- */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: %(ink_muted)s;
    border-bottom: 1px solid %(border)s;
    padding-bottom: 0.45rem;
    margin: 1.6rem 0 0.9rem 0;
}

/* ---- Tabel ---- */
[data-testid="stDataFrame"] {
    border: 1px solid %(border)s;
    border-radius: 6px;
}

/* ---- Sidebar nav polish ---- */
[data-testid="stSidebarNav"] ul { padding-top: 0.4rem; }

/* ---- Tombol ---- */
.stButton > button, .stDownloadButton > button {
    border-radius: 4px;
    border: 1px solid %(border)s;
    font-family: 'Inter', sans-serif;
}

.stButton > button:hover {
    border-color: %(accent)s;
    color: %(accent_dark)s;
}

/* ---- Selectbox / radio label ---- */
.stSelectbox label, .stRadio label, .stMultiSelect label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: %(ink_muted)s !important;
}

[data-testid="stImage"] img {
    width: 100%%;
    height: auto;
    border-radius: 4px;
    border: 1px solid %(border)s;
}

/* ---- Kartu hasil inferensi (verdict) ---- */
.verdict-card {
    border-radius: 6px;
    padding: 1.5rem 1.6rem;
    text-align: center;
    border: 1px solid %(border)s;
    background-color: %(bg_panel_alt)s;
}
.verdict-card.present {
    background-color: %(success_tint)s;
    border-color: %(success)s;
}
.verdict-card .verdict-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: %(ink_muted)s;
    margin-bottom: 0.5rem;
}
.verdict-card .verdict-value {
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 1.8rem;
    color: %(ink_muted)s;
}
.verdict-card.present .verdict-value {
    color: %(success)s;
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
"""


def inject_base_css():
    css = BASE_CSS % {
        "fonts": FONT_CSS,
        "bg_page": BG_PAGE, "bg_panel": BG_PANEL, "bg_panel_alt": BG_PANEL_ALT,
        "border": BORDER, "ink": INK, "ink_muted": INK_MUTED, "ink_faint": INK_FAINT,
        "accent": ACCENT, "accent_dark": ACCENT_DARK, "accent_tint": ACCENT_TINT,
        "gold": GOLD, "gold_tint": GOLD_TINT,
        "error": ERROR, "error_tint": ERROR_TINT,
        "success": SUCCESS, "success_tint": SUCCESS_TINT,
    }
    st.markdown(css, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = ""):
    sub_html = f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="page-eyebrow">{eyebrow}</div>
        <div class="page-title">{title}</div>
        {sub_html}
        <hr />
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, note: str = "", highlight: bool = False) -> str:
    cls = "metric-card highlight" if highlight else "metric-card"
    note_html = f'<div class="metric-note">{note}</div>' if note else ""
    return (
        f'<div class="{cls}"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>{note_html}</div>'
    )


def badge(text: str, kind: str = "accent") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def config_plate(config_id: str, rows: list[tuple[str, str]]) -> str:
    row_html = "".join(
        f'<div class="config-row"><span class="k">{k}</span><span class="v">{v}</span></div>'
        for k, v in rows
    )
    return (
        f'<div class="config-plate"><div class="config-id">{config_id}</div>{row_html}</div>'
    )


def verdict_card(label: str, present: bool) -> str:
    cls = "verdict-card present" if present else "verdict-card"
    value = "ADA" if present else "TIDAK ADA"
    return (
        f'<div class="{cls}"><div class="verdict-label">{label}</div>'
        f'<div class="verdict-value">{value}</div></div>'
    )


def confusion_matrix_html(title: str, tp: int, fp: int, fn: int, tn: int) -> str:
    return f"""
    <div class="cm-wrap">
        <div class="cm-title">{title}</div>
        <div class="cm-row">
            <div class="cm-cell tp"><span class="cm-value">{tp}</span><span class="cm-label">TP</span></div>
            <div class="cm-cell fp"><span class="cm-value">{fp}</span><span class="cm-label">FP</span></div>
        </div>
        <div class="cm-row">
            <div class="cm-cell fn"><span class="cm-value">{fn}</span><span class="cm-label">FN</span></div>
            <div class="cm-cell tn"><span class="cm-value">{tn}</span><span class="cm-label">TN</span></div>
        </div>
        <div class="cm-axis">Baris: prediksi Ada / Tidak Ada &nbsp;|&nbsp; Kolom: aktual Ada / Tidak Ada</div>
    </div>
    """
