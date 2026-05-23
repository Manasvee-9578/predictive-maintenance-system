"""
╔══════════════════════════════════════════════════════════════╗
║   Theme — Dark Industrial AI Operations Center              ║
╚══════════════════════════════════════════════════════════════╝

Futuristic glassmorphism CSS, color tokens, and layout constants
for the Predictive Maintenance enterprise dashboard.
"""

# ── Color Palette ────────────────────────────────────────────

COLORS = {
    "bg_primary": "#0a0e1a",
    "bg_secondary": "#111827",
    "bg_card": "rgba(17, 24, 39, 0.7)",
    "bg_card_hover": "rgba(30, 41, 59, 0.8)",
    "accent": "#3b82f6",
    "accent_glow": "rgba(59, 130, 246, 0.25)",
    "accent_cyan": "#06b6d4",
    "accent_violet": "#8b5cf6",
    "accent_emerald": "#10b981",
    "accent_amber": "#f59e0b",
    "accent_rose": "#f43f5e",
    "text_primary": "#f1f5f9",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
    "border": "rgba(148, 163, 184, 0.08)",
    "border_glow": "rgba(59, 130, 246, 0.15)",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "critical": "#dc2626",
    "glass": "rgba(15, 23, 42, 0.6)",
    "glass_border": "rgba(148, 163, 184, 0.12)",
}

PRIORITY_COLORS = {
    "Low": "#10b981",
    "Medium": "#f59e0b",
    "High": "#f97316",
    "Critical": "#ef4444",
}

HEALTH_COLORS = {
    "Normal": "#10b981",
    "Warning": "#f59e0b",
    "Critical": "#ef4444",
}

PLOTLY_TEMPLATE = "plotly_dark"

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color="#e2e8f0", size=12),
    margin=dict(l=50, r=30, t=60, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor="#1e293b",
        font_size=12,
        font_family="Inter, system-ui, sans-serif",
        bordercolor="#334155",
    ),
    xaxis=dict(
        gridcolor="rgba(148,163,184,0.06)", zerolinecolor="rgba(148,163,184,0.1)"
    ),
    yaxis=dict(
        gridcolor="rgba(148,163,184,0.06)", zerolinecolor="rgba(148,163,184,0.1)"
    ),
)


# ── CSS Theme ────────────────────────────────────────────────


def get_css() -> str:
    """Return the complete dashboard CSS theme."""
    return """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.7);
    --accent: #3b82f6;
    --accent-glow: rgba(59, 130, 246, 0.25);
    --accent-cyan: #06b6d4;
    --accent-violet: #8b5cf6;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: rgba(148, 163, 184, 0.08);
    --border-glow: rgba(59, 130, 246, 0.15);
    --glass: rgba(15, 23, 42, 0.6);
    --glass-border: rgba(148, 163, 184, 0.12);
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --radius: 16px;
    --radius-sm: 10px;
}

/* ── Global ──────────────────────────────────── */
.stApp {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 50%, #0a0e1a 100%);
}

html, body, .stApp { color: var(--text-primary); }
h1 { font-weight: 800; letter-spacing: -1px; font-size: 2rem !important; }
h2 { font-weight: 700; letter-spacing: -0.5px; font-size: 1.5rem !important; color: var(--text-primary) !important; }
h3 { font-weight: 600; font-size: 1.15rem !important; color: var(--text-secondary) !important; }

/* ── Sidebar ─────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    border-right: 1px solid var(--glass-border);
}
section[data-testid="stSidebar"] .stRadio > label { display: none; }
section[data-testid="stSidebar"] .stRadio > div {
    display: flex; flex-direction: column; gap: 2px;
}
section[data-testid="stSidebar"] .stRadio > div > label {
    padding: 10px 16px;
    border-radius: var(--radius-sm);
    transition: all 0.2s ease;
    font-weight: 500;
    font-size: 0.92rem;
    border: 1px solid transparent;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(59, 130, 246, 0.08);
    border-color: var(--border-glow);
}
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio > div [data-checked="true"] {
    background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(6,182,212,0.08));
    border-color: var(--accent);
    color: var(--accent) !important;
}

/* ── Metrics (st.metric) ─────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, var(--bg-card), rgba(30, 41, 59, 0.5));
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 20px 24px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
}
div[data-testid="stMetric"]:hover {
    border-color: var(--border-glow);
    box-shadow: 0 4px 30px var(--accent-glow);
    transform: translateY(-2px);
}
div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-weight: 500; font-size: 0.85rem !important;
    text-transform: uppercase; letter-spacing: 0.5px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 700 !important; font-size: 1.8rem !important;
    background: linear-gradient(135deg, var(--accent), var(--accent-cyan));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-weight: 600;
}

/* ── Glass Cards (custom HTML) ───────────────── */
.glass-card {
    background: linear-gradient(145deg, var(--glass), rgba(30,41,59,0.4));
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 24px;
    transition: all 0.3s ease;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.glass-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 8px 40px var(--accent-glow);
    transform: translateY(-2px);
}

/* ── Alert Cards ─────────────────────────────── */
.alert-card {
    background: linear-gradient(145deg, var(--glass), rgba(30,41,59,0.4));
    backdrop-filter: blur(12px);
    border-radius: var(--radius);
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.alert-critical { border-left: 4px solid var(--danger); }
.alert-high { border-left: 4px solid #f97316; }
.alert-medium { border-left: 4px solid var(--warning); }
.alert-low { border-left: 4px solid var(--success); }

/* ── KPI Hero Card ───────────────────────────── */
.kpi-hero {
    background: linear-gradient(145deg, var(--glass), rgba(30,41,59,0.3));
    backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 28px;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}
.kpi-hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent-cyan), var(--accent-violet));
    border-radius: var(--radius) var(--radius) 0 0;
}
.kpi-hero:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px var(--accent-glow);
    border-color: var(--border-glow);
}
.kpi-hero .kpi-icon { font-size: 2rem; margin-bottom: 8px; }
.kpi-hero .kpi-label {
    font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px;
    color: var(--text-secondary); margin-bottom: 6px;
}
.kpi-hero .kpi-value {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, var(--accent), var(--accent-cyan));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.kpi-hero .kpi-delta {
    font-size: 0.82rem; font-weight: 600; margin-top: 4px;
}

/* ── Section Headers ─────────────────────────── */
.section-header {
    display: flex; align-items: center; gap: 12px;
    margin: 32px 0 20px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--glass-border);
}
.section-header .section-icon { font-size: 1.4rem; }
.section-header .section-title {
    font-size: 1.2rem; font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.3px;
}

/* ── Status Badge ────────────────────────────── */
.status-badge {
    display: inline-block; padding: 4px 14px;
    border-radius: 20px; font-size: 0.78rem;
    font-weight: 600; letter-spacing: 0.3px;
    text-transform: uppercase;
}
.badge-critical { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-high { background: rgba(249,115,22,0.15); color: #f97316; border: 1px solid rgba(249,115,22,0.3); }
.badge-warning { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-normal { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }

/* ── Streamlit Overrides ─────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-weight: 600;
    color: var(--text-secondary);
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.1);
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent);
}

div[data-testid="stExpander"] {
    background: var(--bg-card);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-sm);
}
div[data-testid="stSelectbox"] > div { border-color: var(--glass-border); }

/* ── Scrollbar ───────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(148,163,184,0.2);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(148,163,184,0.35); }

/* ── Top Gradient Bar ────────────────────────── */
.top-bar {
    height: 3px; width: 100%;
    background: linear-gradient(90deg, #3b82f6, #06b6d4, #8b5cf6, #3b82f6);
    background-size: 300% 100%;
    animation: gradient-flow 8s ease infinite;
    margin-bottom: 24px; border-radius: 2px;
}
@keyframes gradient-flow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Pulse Animation ─────────────────────────── */
.pulse-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}
.pulse-green { background: #10b981; box-shadow: 0 0 8px rgba(16,185,129,0.5); }
.pulse-amber { background: #f59e0b; box-shadow: 0 0 8px rgba(245,158,11,0.5); }
.pulse-red { background: #ef4444; box-shadow: 0 0 8px rgba(239,68,68,0.5); }

/* ── Hide Streamlit defaults ─────────────────── */
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }
"""
