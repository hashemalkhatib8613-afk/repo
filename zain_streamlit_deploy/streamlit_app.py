import base64
import html
import os
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "zain-logo.png"
DB_PATH = APP_DIR / "zain_customer_360_ai_demo.db"


def load_streamlit_secret() -> None:
    """Load OpenAI key from Streamlit secrets when deployed on Streamlit Cloud."""
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None

    if key:
        os.environ["OPENAI_API_KEY"] = str(key)


load_streamlit_secret()

from class3_sql_agent_backend import (  # noqa: E402
    ask_sql_agent_payload,
    build_chart_from_question,
    execute_sql_query,
    get_database_overview,
)

st.set_page_config(
    page_title="Zain 360 Copilot",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Theme and UI helpers
# ============================================================

THEME_TOKENS: Dict[str, Dict[str, str]] = {
    "Dark": {
        "app_bg": "#060916",
        "app_bg_2": "#0B1022",
        "sidebar_bg": "rgba(9, 13, 31, 0.82)",
        "surface": "rgba(15, 23, 42, 0.82)",
        "surface_2": "rgba(20, 30, 55, 0.74)",
        "surface_3": "#121A31",
        "text": "#F8FAFC",
        "text_soft": "#D9E3F0",
        "muted": "#A7B3C5",
        "faint": "#64748B",
        "border": "rgba(255, 255, 255, 0.12)",
        "border_strong": "rgba(167, 139, 250, 0.42)",
        "accent": "#7C3AED",
        "accent_2": "#A855F7",
        "accent_3": "#38BDF8",
        "success": "#22C55E",
        "warning": "#FBBF24",
        "danger": "#F87171",
        "shadow": "0 28px 80px rgba(0, 0, 0, 0.46)",
        "shadow_soft": "0 18px 48px rgba(0, 0, 0, 0.32)",
        "plot_template": "plotly_dark",
        "plot_bg": "rgba(0,0,0,0)",
        "paper_bg": "rgba(0,0,0,0)",
    },
}


def get_theme_name() -> str:
    st.session_state.appearance = "Dark"
    return "Dark"


def theme() -> Dict[str, str]:
    return THEME_TOKENS[get_theme_name()]


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def logo_data_uri() -> str:
    if not LOGO_PATH.exists():
        return ""
    image_bytes = LOGO_PATH.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


LOGO_DATA_URI = logo_data_uri()


def inject_css() -> None:
    t = theme()
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {{
  --app-bg: {t['app_bg']};
  --app-bg-2: {t['app_bg_2']};
  --sidebar-bg: {t['sidebar_bg']};
  --surface: {t['surface']};
  --surface-2: {t['surface_2']};
  --surface-3: {t['surface_3']};
  --text: {t['text']};
  --text-soft: {t['text_soft']};
  --muted: {t['muted']};
  --faint: {t['faint']};
  --border: {t['border']};
  --border-strong: {t['border_strong']};
  --accent: {t['accent']};
  --accent-2: {t['accent_2']};
  --accent-3: {t['accent_3']};
  --success: {t['success']};
  --warning: {t['warning']};
  --danger: {t['danger']};
  --shadow: {t['shadow']};
  --shadow-soft: {t['shadow_soft']};
  --radius-xl: 30px;
  --radius-lg: 22px;
  --radius-md: 16px;
}}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {{
  background:
    radial-gradient(circle at 12% 2%, color-mix(in srgb, var(--accent-3) 12%, transparent), transparent 30%),
    radial-gradient(circle at 94% 8%, color-mix(in srgb, var(--accent) 18%, transparent), transparent 29%),
    linear-gradient(135deg, var(--app-bg), var(--app-bg-2) 48%, var(--app-bg));
  color: var(--text) !important;
  font-family: "Inter", sans-serif !important;
}}

* {{
  font-family: "Inter", sans-serif !important;
}}

header[data-testid="stHeader"] {{
  background: transparent !important;
}}

[data-testid="stToolbar"] {{
  color: var(--text) !important;
}}

.block-container {{
  max-width: 1280px !important;
  padding-top: 1.15rem !important;
  padding-bottom: 6.5rem !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
}}

h1, h2, h3, h4, h5, h6 {{
  color: var(--text) !important;
  letter-spacing: -0.035em !important;
}}

p, li, label, span, div {{
  color: inherit;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
  width: 340px !important;
  min-width: 340px !important;
  background: transparent !important;
  border-right: 0 !important;
}}

section[data-testid="stSidebar"] > div {{
  padding: 1rem 0.8rem !important;
  background: transparent !important;
}}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
  min-height: calc(100vh - 2rem) !important;
  padding: 1rem !important;
  border-radius: 34px !important;
  background:
    radial-gradient(circle at 82% 0%, color-mix(in srgb, var(--accent) 24%, transparent), transparent 34%),
    var(--sidebar-bg) !important;
  border: 1px solid var(--border) !important;
  box-shadow: var(--shadow) !important;
  backdrop-filter: blur(24px) saturate(1.4) !important;
}}

.sidebar-brand-card {{
  position: relative;
  overflow: hidden;
  border-radius: 28px;
  padding: 1rem;
  margin-bottom: 1rem;
  background:
    radial-gradient(circle at 85% 12%, color-mix(in srgb, var(--accent-2) 34%, transparent), transparent 36%),
    linear-gradient(135deg, color-mix(in srgb, var(--surface-3) 88%, transparent), color-mix(in srgb, var(--surface-2) 72%, transparent));
  border: 1px solid var(--border);
  box-shadow: var(--shadow-soft);
}}

.sidebar-brand-row {{
  position: relative;
  z-index: 2;
  display: flex;
  align-items: center;
  gap: 0.78rem;
}}

.brand-mark {{
  width: 46px;
  height: 46px;
  min-width: 46px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 18px;
  color: #fff;
  font-weight: 900;
  font-size: 0.88rem;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  box-shadow: 0 16px 34px color-mix(in srgb, var(--accent) 34%, transparent);
}}

.brand-title {{
  font-size: 1.02rem;
  font-weight: 900;
  letter-spacing: -0.035em;
  line-height: 1.1;
  color: var(--text);
}}

.brand-subtitle {{
  font-size: 0.72rem;
  font-weight: 800;
  color: var(--muted);
  margin-top: 0.14rem;
}}

.brand-copy {{
  position: relative;
  z-index: 2;
  margin-top: 0.82rem;
  color: var(--text-soft);
  font-size: 0.78rem;
  line-height: 1.55;
}}

.sidebar-label {{
  margin: 1.05rem 0 0.45rem 0.2rem;
  color: var(--muted);
  font-size: 0.68rem;
  font-weight: 900;
  text-transform: uppercase;
  letter-spacing: 0.13em;
}}

.sidebar-divider {{
  height: 1px;
  margin: 1rem 0;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
}}

.status-pill {{
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  margin-top: 0.8rem;
  padding: 0.42rem 0.62rem;
  border-radius: 999px;
  color: var(--text-soft);
  font-size: 0.72rem;
  font-weight: 800;
  background: color-mix(in srgb, var(--surface-3) 68%, transparent);
  border: 1px solid var(--border);
}}

.status-dot {{
  width: 8px;
  height: 8px;
  border-radius: 99px;
  background: var(--success);
  box-shadow: 0 0 0 5px color-mix(in srgb, var(--success) 15%, transparent);
}}

.sidebar-footnote {{
  margin-top: 1rem;
  padding: 0.85rem;
  border-radius: 20px;
  color: var(--muted);
  font-size: 0.72rem;
  line-height: 1.48;
  background: color-mix(in srgb, var(--surface-3) 55%, transparent);
  border: 1px solid var(--border);
}}

/* Buttons */
section[data-testid="stSidebar"] div.stButton > button {{
  width: 100% !important;
  min-height: 42px !important;
  border-radius: 16px !important;
  border: 1px solid transparent !important;
  color: var(--text-soft) !important;
  background: transparent !important;
  font-size: 0.84rem !important;
  font-weight: 800 !important;
  text-align: left !important;
  justify-content: flex-start !important;
  padding: 0.55rem 0.72rem !important;
  box-shadow: none !important;
  transition: 150ms ease !important;
}}

section[data-testid="stSidebar"] div.stButton > button:hover {{
  color: var(--text) !important;
  border-color: var(--border) !important;
  background: color-mix(in srgb, var(--surface-3) 68%, transparent) !important;
}}

section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
  color: #fff !important;
  background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
  border-color: color-mix(in srgb, var(--accent) 34%, transparent) !important;
  box-shadow: 0 14px 34px color-mix(in srgb, var(--accent) 30%, transparent) !important;
}}

.main div.stButton > button, .stDownloadButton > button {{
  min-height: 42px !important;
  border-radius: 15px !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  background: color-mix(in srgb, var(--surface-3) 70%, transparent) !important;
  font-weight: 800 !important;
  box-shadow: var(--shadow-soft) !important;
  transition: 150ms ease !important;
}}

.main div.stButton > button:hover, .stDownloadButton > button:hover {{
  transform: translateY(-1px);
  border-color: var(--border-strong) !important;
}}

.main div.stButton > button[kind="primary"] {{
  color: #fff !important;
  border-color: color-mix(in srgb, var(--accent) 42%, transparent) !important;
  background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
  box-shadow: 0 18px 38px color-mix(in srgb, var(--accent) 28%, transparent) !important;
}}

/* Inputs */
input, textarea, [data-baseweb="select"] > div, [data-baseweb="tag"] {{
  border-radius: 15px !important;
}}

input, textarea {{
  color: var(--text) !important;
  background: color-mix(in srgb, var(--surface-3) 72%, transparent) !important;
  border-color: var(--border) !important;
}}

textarea:focus, input:focus {{
  border-color: var(--border-strong) !important;
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 12%, transparent) !important;
}}

/* Cards */
.hero-card {{
  position: relative;
  overflow: hidden;
  border-radius: 34px;
  padding: 1.45rem;
  margin-bottom: 1.25rem;
  background:
    radial-gradient(circle at 86% 26%, color-mix(in srgb, var(--accent) 22%, transparent), transparent 30%),
    radial-gradient(circle at 100% 4%, color-mix(in srgb, var(--accent-3) 14%, transparent), transparent 27%),
    linear-gradient(135deg, color-mix(in srgb, var(--surface-3) 86%, transparent), color-mix(in srgb, var(--surface-2) 74%, transparent));
  border: 1px solid var(--border);
  box-shadow: var(--shadow-soft);
  backdrop-filter: blur(20px) saturate(1.3);
}}

.hero-eyebrow {{
  color: var(--accent);
  font-size: 0.72rem;
  font-weight: 900;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  margin-bottom: 0.42rem;
}}

.hero-title {{
  color: var(--text);
  font-size: clamp(1.65rem, 4vw, 2.65rem);
  line-height: 1.02;
  font-weight: 900;
  letter-spacing: -0.06em;
  max-width: 850px;
}}

.hero-copy {{
  color: var(--text-soft);
  font-size: 0.96rem;
  line-height: 1.62;
  max-width: 850px;
  margin-top: 0.66rem;
}}

.hero-logo {{
  position: absolute;
  right: 1.5rem;
  top: 50%;
  transform: translateY(-50%);
  width: 148px;
  max-width: 24%;
  opacity: 0.14;
  filter: grayscale(1);
}}

.stat-card, .insight-card, .mini-card {{
  height: 100%;
  border-radius: 24px;
  padding: 1rem;
  background: color-mix(in srgb, var(--surface-3) 72%, transparent);
  border: 1px solid var(--border);
  box-shadow: var(--shadow-soft);
  backdrop-filter: blur(18px);
}}

.stat-label {{
  color: var(--muted);
  font-size: 0.76rem;
  font-weight: 850;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}}

.stat-value {{
  color: var(--text);
  font-size: 1.72rem;
  font-weight: 950;
  letter-spacing: -0.05em;
  margin-top: 0.38rem;
}}

.stat-help {{
  color: var(--muted);
  font-size: 0.78rem;
  line-height: 1.45;
  margin-top: 0.35rem;
}}

.insight-card strong {{ color: var(--text); }}
.insight-card {{ color: var(--text-soft); font-size: 0.86rem; line-height: 1.55; }}

.section-title {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin: 1.15rem 0 0.65rem;
}}

.section-title h3 {{
  margin: 0;
  font-size: 1.08rem;
  font-weight: 900;
}}

.section-title p {{
  margin: 0.16rem 0 0;
  color: var(--muted);
  font-size: 0.84rem;
}}

.soft-caption {{
  color: var(--muted);
  font-size: 0.82rem;
  line-height: 1.55;
}}

.chip-row {{
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin: 0.75rem 0 0.5rem;
}}

.fake-chip {{
  border-radius: 999px;
  padding: 0.42rem 0.72rem;
  font-size: 0.76rem;
  font-weight: 850;
  color: var(--text-soft);
  background: color-mix(in srgb, var(--surface-3) 68%, transparent);
  border: 1px solid var(--border);
}}

/* Native components */
div[data-testid="stMetric"] {{
  border-radius: 24px !important;
  background: color-mix(in srgb, var(--surface-3) 72%, transparent) !important;
  border: 1px solid var(--border) !important;
  padding: 1rem !important;
  box-shadow: var(--shadow-soft) !important;
}}

div[data-testid="stMetric"] label, div[data-testid="stMetricDelta"] {{
  color: var(--muted) !important;
}}

div[data-testid="stMetricValue"] {{
  color: var(--text) !important;
  font-weight: 900 !important;
}}

.stPlotlyChart, div[data-testid="stDataFrame"], div[data-testid="stExpander"], div[data-testid="stCodeBlock"] {{
  border-radius: 24px !important;
  overflow: hidden !important;
  border: 1px solid var(--border) !important;
  background: color-mix(in srgb, var(--surface-3) 72%, transparent) !important;
  box-shadow: var(--shadow-soft) !important;
}}

div[data-testid="stExpander"] summary {{
  font-weight: 850 !important;
  color: var(--text) !important;
}}

div[data-testid="stTabs"] button {{
  font-weight: 850 !important;
}}

/* Chat */
div[data-testid="stChatMessage"] {{
  padding: 0.6rem 0 !important;
  background: transparent !important;
}}

div[data-testid="stChatMessageContent"] {{
  color: var(--text-soft) !important;
  line-height: 1.65 !important;
  font-size: 0.95rem !important;
}}

div[data-testid="stChatInput"] {{
  border-radius: 26px !important;
}}

[data-testid="stBottomBlockContainer"] {{
  background: linear-gradient(180deg, transparent, var(--app-bg) 36%) !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: color-mix(in srgb, var(--muted) 36%, transparent); border-radius: 999px; }}
::-webkit-scrollbar-thumb:hover {{ background: color-mix(in srgb, var(--muted) 52%, transparent); }}

@media (max-width: 980px) {{
  section[data-testid="stSidebar"] {{ width: auto !important; min-width: auto !important; }}
  .block-container {{ padding-left: 1rem !important; padding-right: 1rem !important; }}
  .hero-card {{ padding: 1.15rem; border-radius: 26px; }}
  .hero-logo {{ display: none; }}
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, eyebrow: str = "ZAIN 360 COPILOT") -> None:
    logo = f'<img class="hero-logo" src="{LOGO_DATA_URI}" alt="Zain logo">' if LOGO_DATA_URI else ""
    st.markdown(
        f"""
        <div class="hero-card">
          <div class="hero-eyebrow">{esc(eyebrow)}</div>
          <div class="hero-title">{esc(title)}</div>
          <div class="hero-copy">{esc(subtitle)}</div>
          {logo}
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: Any, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="stat-card">
          <div class="stat-label">{esc(label)}</div>
          <div class="stat-value">{esc(value)}</div>
          <div class="stat-help">{esc(help_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-title">
          <div>
            <h3>{esc(title)}</h3>
            <p>{esc(subtitle)}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="insight-card">
          <strong>{esc(title)}</strong><br>
          {esc(body)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_number(value: float, suffix: str = "") -> str:
    if pd.isna(value):
        return "—"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M{suffix}"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K{suffix}"
    if value.is_integer():
        return f"{int(value):,}{suffix}"
    return f"{value:,.2f}{suffix}"


def apply_plot_style(fig: go.Figure, height: int = 410) -> go.Figure:
    t = theme()
    fig.update_layout(
        template=t["plot_template"],
        height=height,
        paper_bgcolor=t["paper_bg"],
        plot_bgcolor=t["plot_bg"],
        margin=dict(l=18, r=18, t=58, b=18),
        font=dict(family="Inter", color=t["text_soft"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.16)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.16)", zeroline=False)
    return fig


def render_plot(fig: go.Figure, height: int = 410) -> None:
    st.plotly_chart(apply_plot_style(fig, height), use_container_width=True)


# ============================================================
# Database helpers
# ============================================================


def connect_db() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=300, show_spinner=False)
def query_df(sql: str, params: Tuple[Any, ...] = ()) -> pd.DataFrame:
    with connect_db() as conn:
        return pd.read_sql_query(sql, conn, params=params)


@st.cache_data(ttl=300, show_spinner=False)
def get_options() -> Dict[str, List[str]]:
    options_sql = {
        "cities": "SELECT DISTINCT city AS value FROM customers WHERE city IS NOT NULL ORDER BY city",
        "customer_segments": "SELECT DISTINCT customer_segment AS value FROM customers WHERE customer_segment IS NOT NULL ORDER BY customer_segment",
        "value_segments": "SELECT DISTINCT value_segment AS value FROM customer_value_segments WHERE value_segment IS NOT NULL ORDER BY value_segment",
        "risk_levels": "SELECT DISTINCT risk_level AS value FROM customer_churn_scores WHERE risk_level IS NOT NULL ORDER BY risk_level",
        "service_types": "SELECT DISTINCT service_type AS value FROM subscriptions WHERE service_type IS NOT NULL ORDER BY service_type",
        "plan_categories": "SELECT DISTINCT plan_category AS value FROM plans WHERE plan_category IS NOT NULL ORDER BY plan_category",
        "statuses": "SELECT DISTINCT status AS value FROM customers WHERE status IS NOT NULL ORDER BY status",
    }
    out: Dict[str, List[str]] = {}
    for key, sql in options_sql.items():
        df = query_df(sql)
        out[key] = df["value"].dropna().astype(str).tolist() if not df.empty else []
    return out


@st.cache_data(ttl=300, show_spinner=False)
def get_month_bounds() -> Tuple[date, date]:
    df = query_df(
        """
        SELECT
          MIN(date(summary_month || '-01')) AS min_month,
          MAX(date(summary_month || '-01')) AS max_month
        FROM customer_monthly_summary
        """
    )
    if df.empty or pd.isna(df.loc[0, "min_month"]):
        today = date.today()
        return today, today
    start = datetime.strptime(str(df.loc[0, "min_month"]), "%Y-%m-%d").date()
    end = datetime.strptime(str(df.loc[0, "max_month"]), "%Y-%m-%d").date()
    return start, end


def in_clause(column: str, values: Sequence[str], params: List[Any], clauses: List[str]) -> None:
    values = [v for v in values if v]
    if not values:
        return
    placeholders = ", ".join(["?"] * len(values))
    clauses.append(f"{column} IN ({placeholders})")
    params.extend(values)


@st.cache_data(ttl=300, show_spinner=False)
def load_filtered_monthly_data(
    start_date: str,
    end_date: str,
    cities: Tuple[str, ...],
    customer_segments: Tuple[str, ...],
    value_segments: Tuple[str, ...],
    risk_levels: Tuple[str, ...],
    service_types: Tuple[str, ...],
    plan_categories: Tuple[str, ...],
    statuses: Tuple[str, ...],
) -> pd.DataFrame:
    clauses = ["date(cms.summary_month || '-01') BETWEEN ? AND ?"]
    params: List[Any] = [start_date, end_date]

    in_clause("c.city", cities, params, clauses)
    in_clause("c.customer_segment", customer_segments, params, clauses)
    in_clause("cvs.value_segment", value_segments, params, clauses)
    in_clause("ch.risk_level", risk_levels, params, clauses)
    in_clause("s.service_type", service_types, params, clauses)
    in_clause("p.plan_category", plan_categories, params, clauses)
    in_clause("c.status", statuses, params, clauses)

    where_sql = " AND ".join(clauses)
    sql = f"""
    SELECT
      cms.summary_id,
      cms.summary_month,
      cms.customer_id,
      cms.subscription_id,
      c.full_name,
      c.city,
      c.governorate,
      c.customer_type,
      c.customer_segment,
      c.status AS customer_status,
      cvs.value_segment,
      ch.risk_level,
      ch.churn_score,
      ch.main_risk_reason,
      ch.recommended_action,
      s.service_type,
      s.msisdn,
      s.status AS subscription_status,
      p.plan_name,
      p.plan_category,
      p.monthly_fee_jod,
      cms.total_revenue_jod,
      cms.voice_minutes,
      cms.data_used_gb,
      cms.sms_count,
      cms.support_interactions_count,
      cms.complaints_count,
      cms.payment_delay_days
    FROM customer_monthly_summary cms
    JOIN customers c ON c.customer_id = cms.customer_id
    LEFT JOIN customer_churn_scores ch ON ch.customer_id = cms.customer_id
    LEFT JOIN customer_value_segments cvs ON cvs.customer_id = cms.customer_id
    LEFT JOIN subscriptions s ON s.subscription_id = cms.subscription_id
    LEFT JOIN plans p ON p.plan_id = s.plan_id
    WHERE {where_sql}
    """
    df = query_df(sql, tuple(params))
    numeric_cols = [
        "churn_score",
        "monthly_fee_jod",
        "total_revenue_jod",
        "voice_minutes",
        "data_used_gb",
        "sms_count",
        "support_interactions_count",
        "complaints_count",
        "payment_delay_days",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def search_customers(term: str, limit: int = 30) -> pd.DataFrame:
    term = term.strip()
    if not term:
        sql = """
        SELECT customer_id, full_name, phone_number, email, city, customer_segment, status
        FROM customers
        ORDER BY customer_id
        LIMIT ?
        """
        return query_df(sql, (limit,))

    if term.isdigit():
        sql = """
        SELECT customer_id, full_name, phone_number, email, city, customer_segment, status
        FROM customers
        WHERE customer_id = ? OR phone_number LIKE ?
        ORDER BY customer_id
        LIMIT ?
        """
        return query_df(sql, (int(term), f"%{term}%", limit))

    like = f"%{term}%"
    sql = """
    SELECT customer_id, full_name, phone_number, email, city, customer_segment, status
    FROM customers
    WHERE full_name LIKE ? OR email LIKE ? OR phone_number LIKE ? OR city LIKE ?
    ORDER BY full_name
    LIMIT ?
    """
    return query_df(sql, (like, like, like, like, limit))


@st.cache_data(ttl=300, show_spinner=False)
def load_customer_bundle(customer_id: int) -> Dict[str, pd.DataFrame]:
    queries = {
        "profile": """
            SELECT c.*, ch.churn_score, ch.risk_level, ch.main_risk_reason, ch.recommended_action,
                   cvs.arpu_jod, cvs.total_revenue_6m_jod, cvs.value_segment, cvs.lifetime_months
            FROM customers c
            LEFT JOIN customer_churn_scores ch ON ch.customer_id = c.customer_id
            LEFT JOIN customer_value_segments cvs ON cvs.customer_id = c.customer_id
            WHERE c.customer_id = ?
        """,
        "subscriptions": """
            SELECT s.subscription_id, s.msisdn, s.service_type, s.activation_date, s.contract_end_date,
                   s.status, s.auto_renewal_flag, s.primary_subscription_flag,
                   p.plan_name, p.plan_category, p.monthly_fee_jod, p.data_allowance_gb, p.technology
            FROM subscriptions s
            LEFT JOIN plans p ON p.plan_id = s.plan_id
            WHERE s.customer_id = ?
            ORDER BY s.primary_subscription_flag DESC, s.activation_date DESC
        """,
        "invoices": """
            SELECT i.invoice_id, i.issue_date, i.due_date, i.total_amount_jod, i.payment_status, i.days_overdue
            FROM invoices i
            JOIN accounts a ON a.account_id = i.account_id
            WHERE a.customer_id = ?
            ORDER BY i.issue_date DESC
            LIMIT 12
        """,
        "complaints": """
            SELECT complaint_date, complaint_category, severity, status, compensation_amount_jod, complaint_description
            FROM complaints
            WHERE customer_id = ?
            ORDER BY complaint_date DESC
            LIMIT 12
        """,
        "support": """
            SELECT interaction_datetime, channel, reason_category, issue_type, priority,
                   resolution_status, resolution_time_minutes, customer_sentiment
            FROM support_interactions
            WHERE customer_id = ?
            ORDER BY interaction_datetime DESC
            LIMIT 12
        """,
        "monthly": """
            SELECT summary_month, total_revenue_jod, data_used_gb, voice_minutes, sms_count,
                   support_interactions_count, complaints_count, payment_delay_days, churn_score
            FROM customer_monthly_summary
            WHERE customer_id = ?
            ORDER BY summary_month
        """,
        "campaigns": """
            SELECT cr.sent_date, ca.campaign_name, ca.campaign_type, cr.channel, cr.response_status,
                   cr.converted_flag, cr.revenue_generated_jod
            FROM customer_campaign_responses cr
            JOIN campaigns ca ON ca.campaign_id = cr.campaign_id
            WHERE cr.customer_id = ?
            ORDER BY cr.sent_date DESC
            LIMIT 12
        """,
        "devices": """
            SELECT device_type, brand, model, os, purchase_date, device_5g_capable_flag,
                   installment_flag, monthly_installment_jod
            FROM devices
            WHERE customer_id = ?
            ORDER BY purchase_date DESC
        """,
    }
    return {name: query_df(sql, (customer_id,)) for name, sql in queries.items()}


# ============================================================
# State and chat management
# ============================================================

SUGGESTED_QUESTIONS = [
    "Show me the full profile, plan, complaints, churn risk, invoices, and recommended action for customer ID 9.",
    "Find the top 10 customers with the highest churn score and explain why they are at risk.",
    "Which customers have overdue invoices and high churn risk?",
    "What are the most common complaint categories and how many are still unresolved?",
    "Which marketing campaigns have the best conversion rate?",
    "Which customer segments bring the most revenue in the last 6 months?",
    "Which cities have the highest number of affected customers from network events?",
    "Which customers have 5G capable devices but are not on a 5G plan?",
]

NAV_ITEMS = [
    ("Executive Overview", "📊", "Overview"),
    ("Dynamic Analytics", "🎛️", "Analytics"),
    ("Customer Explorer", "👤", "Customer"),
    ("Chart Builder", "📈", "Charts"),
    ("SQL Console", "🧮", "SQL"),
]


def default_assistant_message() -> Dict[str, str]:
    return {
        "role": "assistant",
        "content": (
            "Hello. I can answer business questions about customers, churn, billing, complaints, "
            "campaigns, network events, usage, and revenue."
        ),
    }


def title_from_question(question: str) -> str:
    cleaned = " ".join(question.split())
    return cleaned[:36] + "..." if len(cleaned) > 36 else cleaned or "New Chat"


def init_state() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "Ask AI"
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = [
            {"id": f"chat_1_{int(time.time())}", "title": "New Chat", "messages": [default_assistant_message()]}
        ]
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = st.session_state.chat_sessions[0]["id"]
    if "rename_chat_id" not in st.session_state:
        st.session_state.rename_chat_id = None
    if "rename_chat_value" not in st.session_state:
        st.session_state.rename_chat_value = ""
    if "chat_search" not in st.session_state:
        st.session_state.chat_search = ""


def current_chat() -> Dict[str, Any]:
    init_state()
    for chat in st.session_state.chat_sessions:
        if chat["id"] == st.session_state.current_chat_id:
            return chat
    st.session_state.current_chat_id = st.session_state.chat_sessions[0]["id"]
    return st.session_state.chat_sessions[0]


def create_new_chat() -> None:
    next_id = f"chat_{len(st.session_state.chat_sessions) + 1}_{int(time.time())}"
    chat = {"id": next_id, "title": "New Chat", "messages": [default_assistant_message()]}
    st.session_state.chat_sessions.insert(0, chat)
    st.session_state.current_chat_id = next_id
    st.session_state.page = "Ask AI"
    st.session_state.rename_chat_id = None
    st.session_state.rename_chat_value = ""


def delete_chat(chat_id: str) -> None:
    st.session_state.chat_sessions = [c for c in st.session_state.chat_sessions if c["id"] != chat_id]
    if not st.session_state.chat_sessions:
        st.session_state.chat_sessions = [
            {"id": f"chat_1_{int(time.time())}", "title": "New Chat", "messages": [default_assistant_message()]}
        ]
    if st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat_id = st.session_state.chat_sessions[0]["id"]
    if st.session_state.rename_chat_id == chat_id:
        st.session_state.rename_chat_id = None
        st.session_state.rename_chat_value = ""


def rename_chat(chat_id: str, new_title: str) -> None:
    clean = " ".join(new_title.split()).strip()
    if not clean:
        return
    for chat in st.session_state.chat_sessions:
        if chat["id"] == chat_id:
            chat["title"] = clean
            break
    st.session_state.rename_chat_id = None
    st.session_state.rename_chat_value = ""


def chat_to_markdown(chat: Dict[str, Any]) -> str:
    lines = [f"# {chat['title']}", ""]
    for msg in chat.get("messages", []):
        lines.append(f"## {msg.get('role', '').title()}")
        lines.append(str(msg.get("content", "")))
        if msg.get("sql"):
            lines.extend(["", "```sql", msg["sql"], "```"])
        lines.append("")
    return "\n".join(lines)


# ============================================================
# Sidebar
# ============================================================


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand-card">
              <div class="sidebar-brand-row">
                <div class="brand-mark">AI</div>
                <div>
                  <div class="brand-title">Zain 360 Copilot</div>
                  <div class="brand-subtitle">Customer intelligence workspace</div>
                </div>
              </div>
              <div class="brand-copy">
                Chat, inspect SQL, explore analytics, and move from insight to action using one clean interface.
              </div>
              <div class="status-pill"><span class="status-dot"></span>Connected to local Customer 360 DB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-label">Workspace</div>', unsafe_allow_html=True)
        for page_name, icon, short_label in NAV_ITEMS:
            button_type = "primary" if st.session_state.page == page_name else "secondary"
            if st.button(f"{icon}  {short_label}", key=f"nav_{page_name}", type=button_type):
                st.session_state.page = page_name
                st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">Chats</div>', unsafe_allow_html=True)
        top_col_1, top_col_2 = st.columns([1, 1])
        with top_col_1:
            if st.button("＋", key="new_chat", type="primary", help="New chat"):
                create_new_chat()
                st.rerun()
        with top_col_2:
            if st.button("↻", key="refresh_app", help="Refresh data"):
                st.cache_data.clear()
                st.rerun()

        st.text_input("Search conversations", key="chat_search", placeholder="Search chats...", label_visibility="collapsed")
        search_text = st.session_state.chat_search.lower().strip()

        visible_chats = [
            chat
            for chat in st.session_state.chat_sessions
            if not search_text or search_text in chat["title"].lower()
        ]

        for chat in visible_chats[:10]:
            selected = chat["id"] == st.session_state.current_chat_id and st.session_state.page == "Ask AI"
            row_col, action_col = st.columns([8, 1.35])
            with row_col:
                if st.button(
                    chat["title"],
                    key=f"select_chat_{chat['id']}",
                    type="primary" if selected else "secondary",
                    help="Open chat",
                ):
                    st.session_state.current_chat_id = chat["id"]
                    st.session_state.page = "Ask AI"
                    st.rerun()
            with action_col:
                if st.button("✎", key=f"rename_open_{chat['id']}", help="Rename chat"):
                    st.session_state.rename_chat_id = chat["id"]
                    st.session_state.rename_chat_value = chat["title"]

            if st.session_state.rename_chat_id == chat["id"]:
                new_title = st.text_input(
                    "Rename chat",
                    value=st.session_state.rename_chat_value,
                    key=f"rename_input_{chat['id']}",
                    label_visibility="collapsed",
                )
                save_col, delete_col = st.columns(2)
                with save_col:
                    if st.button("✓", key=f"save_rename_{chat['id']}", type="primary", help="Save name"):
                        rename_chat(chat["id"], new_title)
                        st.rerun()
                with delete_col:
                    if st.button("🗑", key=f"delete_{chat['id']}", help="Delete chat"):
                        delete_chat(chat["id"])
                        st.rerun()

        st.markdown(
            """
            <div class="sidebar-footnote">
              Tip: use Dynamic Analytics for filters and thresholds, then use Ask AI for deeper explanations.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Page: Ask AI
# ============================================================


def process_chat_prompt(prompt: str) -> None:
    chat = current_chat()
    prompt = prompt.strip()
    if not prompt:
        return

    if chat["title"] == "New Chat":
        chat["title"] = title_from_question(prompt)

    chat["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking through the data..."):
            payload = ask_sql_agent_payload(prompt)
        placeholder = st.empty()
        rendered = ""
        for token in payload["answer"].split(" "):
            rendered += token + " "
            placeholder.markdown(rendered)
            time.sleep(0.006)

        if payload.get("sql"):
            with st.expander("SQL used for this answer"):
                st.code(payload["sql"], language="sql")

    chat["messages"].append(
        {"role": "assistant", "content": payload["answer"], "sql": payload.get("sql", "")}
    )


def show_chat() -> None:
    chat = current_chat()
    render_hero(
        "Customer 360 Chat",
        f"{chat['title']} · Ask about customers, churn, billing, complaints, campaigns, network events, usage, and revenue.",
    )

    chip_text = " ".join(["Customer profile", "Churn risk", "Overdue invoices", "Campaign conversion", "Network impact"])
    st.markdown(
        f'<div class="chip-row"><span class="fake-chip">{chip_text}</span></div>',
        unsafe_allow_html=True,
    )

    with st.expander("Quick questions", expanded=False):
        cols = st.columns(2)
        for idx, question in enumerate(SUGGESTED_QUESTIONS):
            with cols[idx % 2]:
                if st.button(question, key=f"quick_q_{idx}"):
                    process_chat_prompt(question)
                    st.rerun()

    for message in chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sql"):
                with st.expander("SQL used for this answer"):
                    st.code(message["sql"], language="sql")

    export_col, clear_col, _ = st.columns([0.12, 0.12, 0.76])
    with export_col:
        st.download_button(
            "↓",
            data=chat_to_markdown(chat),
            file_name=f"{chat['title'].replace(' ', '_')[:40]}_chat.md",
            mime="text/markdown",
            help="Download chat",
        )
    with clear_col:
        if st.button("↺", help="Clear current chat"):
            chat["messages"] = [default_assistant_message()]
            chat["title"] = "New Chat"
            st.rerun()

    prompt = st.chat_input("Ask Zain 360 Copilot anything about the database...")
    if prompt:
        process_chat_prompt(prompt)
        st.rerun()


# ============================================================
# Page: Executive Overview
# ============================================================


def render_backend_chart(chart: Dict[str, Any]) -> None:
    rows = chart.get("rows") or []
    if not rows:
        st.info(chart.get("summary") or "No chart data found.")
        return
    df = pd.DataFrame(rows)
    chart_type = chart.get("chart_type", "bar")
    title = chart.get("title", "Chart")
    if chart_type in {"pie", "doughnut"}:
        fig = px.pie(
            df,
            names="label",
            values="value",
            hole=0.48 if chart_type == "doughnut" else 0,
            title=title,
        )
    elif chart_type == "horizontal_bar":
        fig = px.bar(df, x="value", y="label", orientation="h", title=title)
    elif chart_type == "line":
        fig = px.line(df, x="label", y="value", markers=True, title=title)
    elif chart_type == "area":
        fig = px.area(df, x="label", y="value", title=title)
    else:
        fig = px.bar(df, x="label", y="value", title=title)
    render_plot(fig)
    if chart.get("summary"):
        st.caption(chart["summary"])
    with st.expander("View chart data"):
        st.dataframe(df, use_container_width=True, hide_index=True)


def show_executive_overview() -> None:
    render_hero(
        "Executive Overview",
        "A polished high-level view of customer volume, churn risk, complaint activity, campaigns, network events, and major database areas.",
    )
    data = get_database_overview()

    cols = st.columns(len(data["kpis"]))
    for col, item in zip(cols, data["kpis"]):
        with col:
            stat_card(item["label"], f"{item['value']:,}", "Current records in the Customer 360 database")

    section_title("Core distribution", "Clean executive charts generated from the database.")
    chart_cols = st.columns(2)
    for index, chart in enumerate(data["charts"]):
        with chart_cols[index % 2]:
            render_backend_chart(chart)

    section_title("Database coverage", "Tables available for AI answers, analytics, and SQL exploration.")
    tables_df = pd.DataFrame(data["tables"]).rename(columns={"label": "Table", "value": "Rows"})
    st.dataframe(tables_df, use_container_width=True, hide_index=True)


# ============================================================
# Page: Dynamic Analytics
# ============================================================


def as_date_range(value: Any, default_start: date, default_end: date) -> Tuple[date, date]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, list) and len(value) == 2:
        return value[0], value[1]
    return default_start, default_end


def customer_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg = (
        df.groupby(["customer_id", "full_name", "city", "customer_segment", "value_segment", "risk_level"], dropna=False)
        .agg(
            revenue_jod=("total_revenue_jod", "sum"),
            data_gb=("data_used_gb", "sum"),
            voice_minutes=("voice_minutes", "sum"),
            complaints=("complaints_count", "sum"),
            support_interactions=("support_interactions_count", "sum"),
            avg_payment_delay=("payment_delay_days", "mean"),
            churn_score=("churn_score", "max"),
        )
        .reset_index()
    )
    return agg.sort_values(["churn_score", "revenue_jod"], ascending=[False, False])


def show_dynamic_analytics() -> None:
    render_hero(
        "Dynamic Analytics Lab",
        "Adjust segments, markets, risk levels, dates, plan categories, and thresholds. The KPIs, charts, action queue, and exports update immediately.",
    )

    options = get_options()
    min_month, max_month = get_month_bounds()

    with st.expander("Filters and adjustable parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            selected_range = st.date_input("Month range", value=(min_month, max_month), min_value=min_month, max_value=max_month)
            start_dt, end_dt = as_date_range(selected_range, min_month, max_month)
            cities = st.multiselect("City", options["cities"], placeholder="All cities")
            customer_segments = st.multiselect("Customer segment", options["customer_segments"], placeholder="All segments")
        with c2:
            value_segments = st.multiselect("Value segment", options["value_segments"], placeholder="All value segments")
            risk_levels = st.multiselect("Risk level", options["risk_levels"], placeholder="All risk levels")
            statuses = st.multiselect("Customer status", options["statuses"], placeholder="All statuses")
        with c3:
            service_types = st.multiselect("Service type", options["service_types"], placeholder="All services")
            plan_categories = st.multiselect("Plan category", options["plan_categories"], placeholder="All plan categories")
            high_churn_threshold = st.slider("Action queue churn threshold", 0, 100, 70, 5)
            min_revenue_threshold = st.number_input("Minimum customer revenue for action queue", min_value=0.0, value=0.0, step=10.0)
            min_complaints_threshold = st.slider("Minimum complaints for action queue", 0, 20, 1, 1)

    df = load_filtered_monthly_data(
        start_dt.isoformat(),
        end_dt.isoformat(),
        tuple(cities),
        tuple(customer_segments),
        tuple(value_segments),
        tuple(risk_levels),
        tuple(service_types),
        tuple(plan_categories),
        tuple(statuses),
    )

    if df.empty:
        st.warning("No records match the selected filters. Adjust the filters and try again.")
        return

    unique_customers = int(df["customer_id"].nunique())
    total_revenue = float(df["total_revenue_jod"].sum())
    avg_revenue = total_revenue / unique_customers if unique_customers else 0
    avg_churn = float(df.drop_duplicates("customer_id")["churn_score"].mean())
    total_complaints = int(df["complaints_count"].sum())
    total_data = float(df["data_used_gb"].sum())
    avg_delay = float(df["payment_delay_days"].mean())

    metric_cols = st.columns(6)
    metrics = [
        ("Customers", f"{unique_customers:,}", "Unique customers in selected slice"),
        ("Revenue", format_number(total_revenue, " JOD"), "Total monthly summary revenue"),
        ("Avg / Customer", format_number(avg_revenue, " JOD"), "Revenue divided by unique customers"),
        ("Avg Churn", f"{avg_churn:.1f}", "Average churn score"),
        ("Complaints", f"{total_complaints:,}", "Total complaint count"),
        ("Data Usage", format_number(total_data, " GB"), "Total selected data usage"),
    ]
    for col, item in zip(metric_cols, metrics):
        with col:
            stat_card(*item)

    section_title("Auto insights", "Quick reading of the current filter selection.")
    insight_cols = st.columns(3)
    top_city = df.groupby("city")["total_revenue_jod"].sum().sort_values(ascending=False).head(1)
    top_segment = df.groupby("customer_segment")["total_revenue_jod"].sum().sort_values(ascending=False).head(1)
    risk_counts = df.drop_duplicates("customer_id")["risk_level"].value_counts()
    with insight_cols[0]:
        city_text = f"{top_city.index[0]} leads the selected slice with {format_number(top_city.iloc[0], ' JOD')} revenue." if not top_city.empty else "No city insight available."
        insight_card("Top market", city_text)
    with insight_cols[1]:
        seg_text = f"{top_segment.index[0]} is the strongest segment by revenue in this view." if not top_segment.empty else "No segment insight available."
        insight_card("Segment signal", seg_text)
    with insight_cols[2]:
        high_count = int(risk_counts.get("High", 0)) if not risk_counts.empty else 0
        insight_card("Risk watch", f"{high_count:,} high-risk customers are present. Average payment delay is {avg_delay:.1f} days.")

    tab1, tab2, tab3, tab4 = st.tabs(["Revenue", "Risk & Segments", "Action Queue", "Data Export"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            trend = df.groupby("summary_month", as_index=False)["total_revenue_jod"].sum()
            render_plot(px.line(trend, x="summary_month", y="total_revenue_jod", markers=True, title="Revenue Trend"))
        with c2:
            city_rev = df.groupby("city", as_index=False)["total_revenue_jod"].sum().sort_values("total_revenue_jod", ascending=False).head(10)
            render_plot(px.bar(city_rev, x="city", y="total_revenue_jod", title="Top Cities by Revenue"))

        c3, c4 = st.columns(2)
        with c3:
            service_rev = df.groupby("service_type", as_index=False)["total_revenue_jod"].sum().sort_values("total_revenue_jod", ascending=False)
            render_plot(px.bar(service_rev, x="service_type", y="total_revenue_jod", title="Revenue by Service Type"))
        with c4:
            plan_rev = df.groupby("plan_category", as_index=False)["total_revenue_jod"].sum().sort_values("total_revenue_jod", ascending=False).head(10)
            render_plot(px.bar(plan_rev, x="plan_category", y="total_revenue_jod", title="Revenue by Plan Category"))

    with tab2:
        unique_customer_df = df.drop_duplicates("customer_id")
        c1, c2 = st.columns(2)
        with c1:
            risk_dist = unique_customer_df.groupby("risk_level", as_index=False)["customer_id"].count().rename(columns={"customer_id": "customers"})
            render_plot(px.pie(risk_dist, names="risk_level", values="customers", hole=0.52, title="Risk Level Distribution"))
        with c2:
            segment_rev = df.groupby("value_segment", as_index=False)["total_revenue_jod"].sum().sort_values("total_revenue_jod", ascending=False)
            render_plot(px.bar(segment_rev, x="value_segment", y="total_revenue_jod", title="Revenue by Value Segment"))

        c3, c4 = st.columns(2)
        with c3:
            complaints_segment = df.groupby("customer_segment", as_index=False)["complaints_count"].sum().sort_values("complaints_count", ascending=False)
            render_plot(px.bar(complaints_segment, x="customer_segment", y="complaints_count", title="Complaints by Segment"))
        with c4:
            churn_by_service = df.groupby("service_type", as_index=False)["churn_score"].mean().sort_values("churn_score", ascending=False)
            render_plot(px.bar(churn_by_service, x="service_type", y="churn_score", title="Average Churn by Service"))

    with tab3:
        agg = customer_aggregate(df)
        queue = agg[
            (agg["churn_score"] >= high_churn_threshold)
            & (agg["revenue_jod"] >= min_revenue_threshold)
            & (agg["complaints"] >= min_complaints_threshold)
        ].copy()
        queue = queue.sort_values(["churn_score", "complaints", "revenue_jod"], ascending=[False, False, False])
        queue["recommended_focus"] = queue.apply(
            lambda row: "Retention call + complaint review" if row["complaints"] > 0 else "Retention offer review",
            axis=1,
        )
        st.caption(f"{len(queue):,} customers match the current action thresholds.")
        st.dataframe(queue.head(100), use_container_width=True, hide_index=True)
        st.download_button(
            "Download action queue CSV",
            data=queue.to_csv(index=False).encode("utf-8"),
            file_name="zain_360_action_queue.csv",
            mime="text/csv",
        )

    with tab4:
        preview_cols = [
            "summary_month", "customer_id", "full_name", "city", "customer_segment", "value_segment", "risk_level",
            "service_type", "plan_category", "total_revenue_jod", "data_used_gb", "complaints_count", "payment_delay_days", "churn_score"
        ]
        st.dataframe(df[preview_cols].head(500), use_container_width=True, hide_index=True)
        st.download_button(
            "Download filtered analytics CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="zain_360_filtered_analytics.csv",
            mime="text/csv",
        )


# ============================================================
# Page: Customer Explorer
# ============================================================


def show_customer_explorer() -> None:
    render_hero(
        "Customer Explorer",
        "Search a customer and view profile, plan, value, churn reason, invoices, complaints, support activity, campaigns, and monthly behavior.",
    )

    search_col, count_col = st.columns([3, 1])
    with search_col:
        term = st.text_input("Search customer", placeholder="Customer ID, name, phone, email, or city")
    with count_col:
        limit = st.number_input("Result limit", min_value=5, max_value=100, value=30, step=5)

    matches = search_customers(term, int(limit))
    if matches.empty:
        st.warning("No customers found for this search.")
        return

    labels = {
        int(row.customer_id): f"{int(row.customer_id)} · {row.full_name} · {row.city} · {row.customer_segment} · {row.status}"
        for row in matches.itertuples()
    }
    selected_id = st.selectbox("Select customer", list(labels.keys()), format_func=lambda cid: labels[cid])
    bundle = load_customer_bundle(int(selected_id))
    profile = bundle["profile"]

    if profile.empty:
        st.error("Could not load this customer profile.")
        return

    p = profile.iloc[0]
    metric_cols = st.columns(5)
    with metric_cols[0]:
        stat_card("Risk level", p.get("risk_level", "—"), f"Churn score {float(p.get('churn_score', 0)):.1f}")
    with metric_cols[1]:
        stat_card("Value segment", p.get("value_segment", "—"), f"ARPU {format_number(float(p.get('arpu_jod', 0)), ' JOD')}")
    with metric_cols[2]:
        stat_card("6M revenue", format_number(float(p.get("total_revenue_6m_jod", 0)), " JOD"), "Customer value")
    with metric_cols[3]:
        stat_card("City", p.get("city", "—"), p.get("governorate", ""))
    with metric_cols[4]:
        stat_card("Status", p.get("status", "—"), p.get("customer_segment", ""))

    c1, c2 = st.columns([1.15, 1])
    with c1:
        section_title("Profile snapshot")
        profile_cols = [
            "customer_id", "full_name", "gender", "age_group", "nationality", "preferred_language",
            "email", "phone_number", "signup_date", "customer_type", "customer_segment", "status"
        ]
        profile_df = profile[profile_cols].T.rename(columns={0: "Value"})
        st.dataframe(profile_df, use_container_width=True)
    with c2:
        section_title("Recommended action")
        insight_card("Main risk reason", str(p.get("main_risk_reason", "No risk reason available.")))
        st.write("")
        insight_card("Next best action", str(p.get("recommended_action", "No action available.")))
        if st.button("Ask AI to explain this customer", type="primary"):
            chat = current_chat()
            question = f"Explain customer ID {int(selected_id)} profile, churn risk, complaints, invoices, and recommended action."
            chat["messages"].append({"role": "user", "content": question})
            with st.spinner("Preparing customer explanation..."):
                payload = ask_sql_agent_payload(question)
            chat["messages"].append({"role": "assistant", "content": payload["answer"], "sql": payload.get("sql", "")})
            st.session_state.page = "Ask AI"
            st.rerun()

    tabs = st.tabs(["Monthly", "Subscriptions", "Invoices", "Complaints", "Support", "Campaigns", "Devices"])
    with tabs[0]:
        monthly = bundle["monthly"]
        if monthly.empty:
            st.info("No monthly summary found.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                render_plot(px.line(monthly, x="summary_month", y="total_revenue_jod", markers=True, title="Monthly Revenue"), height=360)
            with c2:
                render_plot(px.line(monthly, x="summary_month", y="churn_score", markers=True, title="Monthly Churn Score"), height=360)
            st.dataframe(monthly, use_container_width=True, hide_index=True)
    tab_names = ["subscriptions", "invoices", "complaints", "support", "campaigns", "devices"]
    for tab, name in zip(tabs[1:], tab_names):
        with tab:
            df = bundle[name]
            if df.empty:
                st.info(f"No {name} records found for this customer.")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================
# Page: Chart Builder
# ============================================================

CHART_TYPES = {
    "Bar chart": "bar",
    "Horizontal bar": "horizontal_bar",
    "Pie chart": "pie",
    "Doughnut chart": "doughnut",
    "Line chart": "line",
    "Area chart": "area",
}


def show_chart_builder() -> None:
    render_hero(
        "Natural Language Chart Builder",
        "Describe the chart you need. The app turns your request into a safe database chart and shows the result with chart data.",
    )
    default_q = "Build a chart based on customer with ID = 9 by their complaints type and number."
    q_col, type_col = st.columns([3, 1])
    with q_col:
        question = st.text_area("Chart inquiry", value=default_q, height=120)
    with type_col:
        chart_label = st.selectbox("Chart type", list(CHART_TYPES.keys()))
        st.write("")
        create_chart = st.button("Create chart", type="primary")

    if create_chart:
        with st.spinner("Building chart from database..."):
            st.session_state.last_chart = build_chart_from_question(question, CHART_TYPES[chart_label])

    if st.session_state.get("last_chart"):
        render_backend_chart(st.session_state.last_chart)


# ============================================================
# Page: SQL Console
# ============================================================

SAFE_SQL_EXAMPLES = {
    "Total customers": "SELECT COUNT(*) AS total_customers FROM customers",
    "Top churn customers": """SELECT c.customer_id, c.full_name, c.city, ch.churn_score, ch.risk_level, ch.recommended_action
FROM customer_churn_scores ch
JOIN customers c ON c.customer_id = ch.customer_id
ORDER BY ch.churn_score DESC
LIMIT 10""",
    "Revenue by value segment": """SELECT value_segment, COUNT(*) AS total_customers, ROUND(AVG(arpu_jod), 2) AS avg_arpu, ROUND(AVG(total_revenue_6m_jod), 2) AS avg_revenue_6m
FROM customer_value_segments
GROUP BY value_segment
ORDER BY avg_revenue_6m DESC""",
    "Complaints by category": """SELECT complaint_category, COUNT(*) AS total_complaints
FROM complaints
GROUP BY complaint_category
ORDER BY total_complaints DESC""",
}


def show_sql_console() -> None:
    render_hero(
        "SQL Console",
        "Run safe read-only SELECT queries, inspect the rows, and export results. Destructive SQL is blocked by the backend validation.",
    )
    example = st.selectbox("Load example", list(SAFE_SQL_EXAMPLES.keys()))
    if "sql_console_text" not in st.session_state:
        st.session_state.sql_console_text = SAFE_SQL_EXAMPLES[example]
    if st.button("Use selected example"):
        st.session_state.sql_console_text = SAFE_SQL_EXAMPLES[example]

    sql = st.text_area("SQL", key="sql_console_text", height=220)
    run = st.button("Run query", type="primary")
    if run:
        try:
            result = execute_sql_query(sql, limit=100)
            rows = result.get("rows", [])
            st.success(f"Returned {len(rows):,} rows.")
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download result CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="sql_result.csv",
                    mime="text/csv",
                )
            with st.expander("Final SQL executed"):
                st.code(result.get("sql", sql), language="sql")
        except Exception as exc:
            st.error(f"Query failed: {type(exc).__name__}: {exc}")


# ============================================================
# App router
# ============================================================

init_state()
inject_css()
render_sidebar()

if not DB_PATH.exists():
    st.error("Database file is missing. Place zain_customer_360_ai_demo.db next to streamlit_app.py.")
    st.stop()

page = st.session_state.page

if page == "Ask AI":
    show_chat()
elif page == "Executive Overview":
    show_executive_overview()
elif page == "Dynamic Analytics":
    show_dynamic_analytics()
elif page == "Customer Explorer":
    show_customer_explorer()
elif page == "Chart Builder":
    show_chart_builder()
else:
    show_sql_console()
