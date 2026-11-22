"""Visual identity for Sentinel: corporate navy + safety orange.

Navy sidebar and hero banners, white bordered panels, one orange accent.
"""

import base64
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ASSETS = Path(__file__).parent / "assets"

INK = "#13233a"
MUTED = "#667085"
FAINT = "#cfd6e0"
BG = "#eef1f5"
PANEL = "#ffffff"
LINE = "#e1e6ee"
ACCENT = "#f26a21"  # safety orange
NAVY = "#12294a"
NAVY2 = "#1d3d6b"
GOOD, WATCH, SOON, BAD = "#1f9d62", "#e8a317", "#ee7623", "#d63b3b"


def _fan_uri() -> str:
    try:
        svg = (ASSETS / "fan.svg").read_bytes()
        return "data:image/svg+xml;base64," + base64.b64encode(svg).decode()
    except OSError:
        return ""


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Manrope', system-ui, sans-serif; }}
.stApp {{ background: {BG}; color: {INK}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"], [data-testid="stMainMenu"],
[data-testid="stDecoration"], #MainMenu, footer {{ display: none !important; }}
/* the white logo is unreadable on the light header when the sidebar is collapsed */
header[data-testid="stHeader"] img, header[data-testid="stHeader"] [data-testid="stLogo"] {{ display: none !important; }}
/* the button that reopens a collapsed sidebar must stay visible */
[data-testid="stExpandSidebarButton"] {{ display: flex !important; background: {NAVY}; border-radius: 10px; }}
[data-testid="stExpandSidebarButton"] * {{ color: #fff !important; fill: #fff !important; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1280px; }}

h1, h2, h3, h4 {{ font-family: 'Sora', sans-serif; color: {INK}; letter-spacing: -0.01em; }}
h1, h2, h3, h4, [data-testid="stHeading"] * {{ font-family: 'Sora', sans-serif !important; }}
h3 {{ font-size: 1.05rem !important; font-weight: 600 !important; margin: 0.2rem 0 0.5rem 0 !important;
      padding: 0.15rem 0 0.15rem 0.7rem !important; border-left: 4px solid {ACCENT}; line-height: 1.25 !important; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: {NAVY}; border-right: none; }}
[data-testid="stSidebar"] * {{ color: #dbe4f0; }}
[data-testid="stSidebarNavLink"] {{ border-radius: 8px; margin: 1px 6px; }}
[data-testid="stSidebarNavLink"]:hover {{ background: rgba(255,255,255,0.07); }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: {ACCENT}; }}
[data-testid="stSidebarNavLink"][aria-current="page"] * {{ color: #fff !important; font-weight: 700; }}
[data-testid="stNavSectionHeader"] {{ color: #7f96b6 !important; text-transform: uppercase; font-size: 0.68rem !important; letter-spacing: 0.14em; }}
.side-box {{ background: rgba(255,255,255,0.06); border-radius: 10px; padding: 0.8rem 0.9rem; margin: 0.6rem 0.4rem; font-size: 0.82rem; line-height: 1.55; }}
.side-box b {{ color: #fff; }}
.side-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; background: #3ddc97; }}

/* hero banner */
.hero {{ position: relative; overflow: hidden; background: linear-gradient(115deg, {NAVY} 0%, {NAVY2} 100%);
         border-radius: 16px; padding: 1.9rem 2.2rem; margin-bottom: 1.3rem; color: #fff; }}
.hero.big {{ padding: 2.6rem 2.4rem; }}
.hero .tag {{ display: inline-block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
              color: {ACCENT}; background: rgba(242,106,33,0.14); padding: 0.25rem 0.7rem; border-radius: 99px; margin-bottom: 0.7rem; }}
.hero h1 {{ font-family: 'Sora'; font-size: 2rem; font-weight: 700; color: #fff; margin: 0 0 0.45rem 0; line-height: 1.15; max-width: 40rem; }}
.hero.big h1 {{ font-size: 2.55rem; }}
.hero p {{ color: #b9c8dd; font-size: 1rem; margin: 0; max-width: 38rem; line-height: 1.55; }}
.hero::after {{ content: ""; position: absolute; right: -70px; top: 50%; width: 330px; height: 330px; margin-top: -165px;
                background: url("{_fan_uri()}") center/contain no-repeat; opacity: 0.28; animation: spin 40s linear infinite; }}
.hero.big::after {{ width: 440px; height: 440px; margin-top: -220px; right: -40px; opacity: 0.4; animation-duration: 28s; }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.hero .cta {{ margin-top: 1.1rem; display: flex; gap: 0.7rem; flex-wrap: wrap; }}
.hero .cta span {{ background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.18); padding: 0.35rem 0.85rem; border-radius: 99px; font-size: 0.85rem; }}

/* panels (st.container(border=True)) */
[data-testid="stVerticalBlockBorderWrapper"] {{ background: {PANEL} !important; border: 1px solid {LINE} !important; border-radius: 14px;
        box-shadow: 0 1px 2px rgba(16,24,40,0.05); padding: 0.4rem 0.5rem; }}

/* kpi cards */
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1.2rem; }}
.kpi {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 14px; padding: 1rem 1.2rem; position: relative; overflow: hidden;
        box-shadow: 0 1px 2px rgba(16,24,40,0.05); }}
.kpi::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: var(--c); }}
.kpi .l {{ font-size: 0.78rem; font-weight: 600; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.06em; }}
.kpi .v {{ font-family: 'Sora'; font-size: 2.1rem; font-weight: 700; line-height: 1.2; margin: 0.15rem 0; }}
.kpi .n {{ font-size: 0.82rem; color: {MUTED}; }}

.verdict {{ font-family: 'Sora'; font-size: 1.3rem; font-weight: 500; line-height: 1.4; padding: 1rem 1.3rem; border-radius: 12px;
            background: {PANEL}; border: 1px solid {LINE}; border-left: 5px solid {ACCENT}; margin: 0.3rem 0 1rem 0; }}
.pill {{ display: inline-block; padding: 0.18rem 0.75rem; border-radius: 99px; font-weight: 700; font-size: 0.82rem; color: #fff; vertical-align: middle; }}
.item {{ display: flex; justify-content: space-between; padding: 0.6rem 0; border-bottom: 1px solid {LINE}; font-size: 0.95rem; }}
.item span:last-child {{ color: {MUTED}; }}
.log {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 12px; padding: 0.7rem 1rem; height: 13rem; overflow-y: auto; font-size: 0.93rem; }}
.log div {{ padding: 0.3rem 0; border-bottom: 1px solid {LINE}; }}
.log div:last-child {{ border-bottom: none; }}
.hint {{ color: {MUTED}; font-size: 0.88rem; }}
.foot {{ text-align: center; color: {MUTED}; font-size: 0.8rem; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid {LINE}; }}

/* controls */
.stButton > button, .stDownloadButton > button {{ border-radius: 10px; border: 1px solid {FAINT}; background: #fff; color: {INK}; font-weight: 600; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
.stButton > button[kind="primary"] {{ background: {ACCENT}; border-color: {ACCENT}; color: #fff; }}
.stButton > button[kind="primary"]:hover {{ background: #d9571a; color: #fff; }}
[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 10px; overflow: hidden; }}
[data-testid="stExpander"] {{ border: 1px solid {LINE}; border-radius: 12px; background: {PANEL}; }}
[data-baseweb="tab-list"] {{ gap: 0.3rem; }}
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, sub: str, tag: str = "", big: bool = False, chips=()):
    tag_html = f'<div class="tag">{tag}</div><br>' if tag else ""
    chips_html = f'<div class="cta">{"".join(f"<span>{c}</span>" for c in chips)}</div>' if chips else ""
    st.markdown(
        f'<div class="hero{" big" if big else ""}">{tag_html}<h1>{title}</h1><p>{sub}</p>{chips_html}</div>',
        unsafe_allow_html=True,
    )


def kpi_cards(items):
    """items: list of (label, value, note, colour)."""
    html = "".join(
        f'<div class="kpi" style="--c:{c}"><div class="l">{l}</div><div class="v">{v}</div><div class="n">{n}</div></div>'
        for l, v, n, c in items
    )
    st.markdown(f'<div class="kpis">{html}</div>', unsafe_allow_html=True)


def pill(label: str, color: str) -> str:
    return f'<span class="pill" style="background:{color}">{label}</span>'


def verdict(html: str):
    st.markdown(f'<div class="verdict">{html}</div>', unsafe_allow_html=True)


def footer():
    st.markdown(
        '<div class="foot">Sentinel · Predictive Maintenance Platform · Built on NASA C-MAPSS FD001 turbofan data · '
        "For academic demonstration</div>",
        unsafe_allow_html=True,
    )


# ── Plotly template ──────────────────────────────────────────
pio.templates["sentinel"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Manrope", color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[ACCENT, NAVY],
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=MUTED)),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=48, r=16, t=20, b=40),
        hoverlabel=dict(bgcolor="#fff", bordercolor=LINE, font=dict(family="Manrope", color=INK)),
    )
)
pio.templates.default = "sentinel"


def style(fig, height=340, **kw):
    fig.update_layout(template="sentinel", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", **kw)
    return fig
