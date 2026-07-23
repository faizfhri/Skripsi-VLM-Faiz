"""Helper Plotly yang konsisten dengan tema dashboard (tanpa gridline ramai,
tanpa warna default Plotly, font selaras dengan CSS kustom)."""

import plotly.graph_objects as go

from utils.theme import ACCENT, BORDER, GOLD, INK, INK_MUTED

FONT_FAMILY = "Inter, sans-serif"
MONO_FAMILY = "IBM Plex Mono, monospace"


def base_layout(**overrides) -> dict:
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=INK, size=13),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=12, color=INK_MUTED),
        ),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=BORDER, tickfont=dict(color=INK_MUTED)),
        yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=False, tickfont=dict(color=INK_MUTED)),
        hoverlabel=dict(bgcolor="white", font_family=FONT_FAMILY, bordercolor=BORDER),
    )
    layout.update(overrides)
    return layout


def apply_theme(fig: go.Figure, **overrides) -> go.Figure:
    fig.update_layout(**base_layout(**overrides))
    return fig
