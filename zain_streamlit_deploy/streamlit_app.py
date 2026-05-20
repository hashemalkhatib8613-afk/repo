import base64
import html
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "zain-logo.png"

DB_CANDIDATES = [
    APP_DIR / "zain_customer_360_ai_demo.db",
    APP_DIR / "zain_customer_360_ai_demo(1).db",
]


def resolve_db_path() -> Path:
    for candidate in DB_CANDIDATES:
        if candidate.exists():
            return candidate
    return DB_CANDIDATES[0]


DB_PATH = resolve_db_path()


def load_streamlit_secret() -> None:
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
)

st.set_page_config(
    page_title="Zain 360 Copilot",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Data access
# -----------------------------------------------------------------------------


def connect_db() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database file was not found. Expected one of: {', '.join(str(p.name) for p in DB_CANDIDATES)}"
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(show_spinner=False)
def query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    with connect_db() as conn:
        return pd.read_sql_query(sql, conn, params=params)


@st.cache_data(show_spinner=False)
def scalar(sql: str, params: tuple = ()):  # noqa: ANN001
    with connect_db() as conn:
        row = conn.execute(sql, params).fetchone()
    if row is None:
        return None
    return list(dict(row).values())[0]


@st.cache_data(show_spinner=False)
def get_table_counts() -> pd.DataFrame:
    with connect_db() as conn:
        tables = [
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ]
        rows = []
        for table in tables:
            count = conn.execute(f"SELECT COUNT(*) AS total FROM {table}").fetchone()["total"]
            rows.append({"Table": table, "Rows": count})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_filter_options() -> dict:
    return {
        "cities": query_df("SELECT DISTINCT city FROM customers WHERE city IS NOT NULL ORDER BY city")["city"].tolist(),
        "segments": query_df("SELECT DISTINCT customer_segment FROM customers WHERE customer_segment IS NOT NULL ORDER BY customer_segment")["customer_segment"].tolist(),
        "risk_levels": query_df("SELECT DISTINCT risk_level FROM customer_churn_scores WHERE risk_level IS NOT NULL ORDER BY risk_level")["risk_level"].tolist(),
        "value_segments": query_df("SELECT DISTINCT value_segment FROM customer_value_segments WHERE value_segment IS NOT NULL ORDER BY value_segment")["value_segment"].tolist(),
        "service_types": query_df("SELECT DISTINCT service_type FROM subscriptions WHERE service_type IS NOT NULL ORDER BY service_type")["service_type"].tolist(),
        "months": query_df("SELECT DISTINCT summary_month FROM customer_monthly_summary ORDER BY summary_month")["summary_month"].tolist(),
    }


def build_customer_where(cities, segments, risk_levels, value_segments):
    clauses = ["1=1"]
    params = []
    if cities:
        clauses.append("c.city IN (" + ",".join("?" for _ in cities) + ")")
        params.extend(cities)
    if segments:
        clauses.append("c.customer_segment IN (" + ",".join("?" for _ in segments) + ")")
        params.extend(segments)
    if risk_levels:
        clauses.append("ch.risk_level IN (" + ",".join("?" for _ in risk_levels) + ")")
        params.extend(risk_levels)
    if value_segments:
        clauses.append("vs.value_segment IN (" + ",".join("?" for _ in value_segments) + ")")
        params.extend(value_segments)
    return " AND ".join(clauses), tuple(params)


@st.cache_data(show_spinner=False)
def analytics_kpis(cities, segments, risk_levels, value_segments, month_start, month_end):
    where, params = build_customer_where(cities, segments, risk_levels, value_segments)
    customer_base = f"""
        FROM customers c
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = c.customer_id
        LEFT JOIN customer_value_segments vs ON vs.customer_id = c.customer_id
        WHERE {where}
    """
    total_customers = scalar(f"SELECT COUNT(DISTINCT c.customer_id) AS value {customer_base}", params) or 0
    high_risk = scalar(
        f"SELECT COUNT(DISTINCT c.customer_id) AS value {customer_base} AND ch.risk_level = 'High'", params
    ) or 0
    avg_churn = scalar(f"SELECT AVG(ch.churn_score) AS value {customer_base}", params) or 0
    avg_arpu = scalar(f"SELECT AVG(vs.arpu_jod) AS value {customer_base}", params) or 0

    monthly_sql = f"""
        SELECT
            COALESCE(SUM(ms.total_revenue_jod), 0) AS revenue,
            COALESCE(SUM(ms.data_used_gb), 0) AS data_gb,
            COALESCE(SUM(ms.complaints_count), 0) AS complaints,
            COALESCE(AVG(ms.payment_delay_days), 0) AS avg_payment_delay
        FROM customer_monthly_summary ms
        JOIN customers c ON c.customer_id = ms.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = c.customer_id
        LEFT JOIN customer_value_segments vs ON vs.customer_id = c.customer_id
        WHERE {where}
          AND ms.summary_month BETWEEN ? AND ?
    """
    df = query_df(monthly_sql, params + (month_start, month_end))
    revenue = float(df.loc[0, "revenue"]) if not df.empty else 0
    data_gb = float(df.loc[0, "data_gb"]) if not df.empty else 0
    complaints = int(df.loc[0, "complaints"]) if not df.empty else 0
    avg_payment_delay = float(df.loc[0, "avg_payment_delay"]) if not df.empty else 0

    return {
        "Customers": int(total_customers),
        "High Risk": int(high_risk),
        "Avg Churn": round(float(avg_churn), 3),
        "Avg ARPU": round(float(avg_arpu), 2),
        "Revenue": round(revenue, 2),
        "Data Used GB": round(data_gb, 2),
        "Complaints": complaints,
        "Avg Delay": round(avg_payment_delay, 1),
    }


@st.cache_data(show_spinner=False)
def dynamic_chart_data(metric, group_by, cities, segments, risk_levels, value_segments, month_start, month_end, top_n):
    where, params = build_customer_where(cities, segments, risk_levels, value_segments)
    group_map = {
        "City": "c.city",
        "Customer Segment": "c.customer_segment",
        "Risk Level": "ch.risk_level",
        "Value Segment": "vs.value_segment",
        "Service Type": "s.service_type",
        "Plan Category": "p.plan_category",
        "Plan Technology": "p.technology",
    }
    group_expr = group_map.get(group_by, "c.city")

    metric_map = {
        "Customers": ("COUNT(DISTINCT c.customer_id)", "customers"),
        "Revenue": ("SUM(ms.total_revenue_jod)", "revenue_jod"),
        "Avg Churn Score": ("AVG(ch.churn_score)", "avg_churn_score"),
        "Avg ARPU": ("AVG(vs.arpu_jod)", "avg_arpu_jod"),
        "Data Usage GB": ("SUM(ms.data_used_gb)", "data_used_gb"),
        "Complaints": ("SUM(ms.complaints_count)", "complaints"),
        "Payment Delay Days": ("AVG(ms.payment_delay_days)", "avg_payment_delay"),
    }
    metric_expr, metric_alias = metric_map.get(metric, metric_map["Customers"])

    # Subscription/plan joins can multiply monthly rows. For the current demo DB this is acceptable for directional analysis,
    # and the grouped customer count uses DISTINCT to protect the customer metric.
    sql = f"""
        SELECT
            COALESCE({group_expr}, 'Unknown') AS label,
            ROUND(COALESCE({metric_expr}, 0), 3) AS value
        FROM customers c
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = c.customer_id
        LEFT JOIN customer_value_segments vs ON vs.customer_id = c.customer_id
        LEFT JOIN subscriptions s ON s.customer_id = c.customer_id
        LEFT JOIN plans p ON p.plan_id = s.plan_id
        LEFT JOIN customer_monthly_summary ms ON ms.customer_id = c.customer_id
            AND ms.summary_month BETWEEN ? AND ?
        WHERE {where}
        GROUP BY label
        ORDER BY value DESC
        LIMIT ?
    """
    df = query_df(sql, (month_start, month_end) + params + (top_n,))
    return df, sql, metric_alias


@st.cache_data(show_spinner=False)
def get_smart_segments(top_n: int = 15) -> dict:
    high_value_risk = query_df(
        """
        SELECT
            c.customer_id, c.full_name, c.city, c.customer_segment,
            vs.value_segment, ROUND(vs.arpu_jod, 2) AS arpu_jod,
            ROUND(ch.churn_score, 3) AS churn_score, ch.risk_level, ch.main_risk_reason, ch.recommended_action
        FROM customers c
        JOIN customer_churn_scores ch ON ch.customer_id = c.customer_id
        JOIN customer_value_segments vs ON vs.customer_id = c.customer_id
        WHERE ch.risk_level = 'High'
        ORDER BY vs.arpu_jod DESC, ch.churn_score DESC
        LIMIT ?
        """,
        (top_n,),
    )
    overdue_risk = query_df(
        """
        SELECT
            c.customer_id, c.full_name, c.city,
            COUNT(i.invoice_id) AS overdue_invoices,
            ROUND(SUM(i.total_amount_jod), 2) AS overdue_amount_jod,
            ROUND(ch.churn_score, 3) AS churn_score,
            ch.risk_level
        FROM invoices i
        JOIN accounts a ON a.account_id = i.account_id
        JOIN customers c ON c.customer_id = a.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = c.customer_id
        WHERE i.payment_status != 'Paid' AND i.days_overdue > 0
        GROUP BY c.customer_id, c.full_name, c.city, ch.churn_score, ch.risk_level
        ORDER BY overdue_amount_jod DESC, churn_score DESC
        LIMIT ?
        """,
        (top_n,),
    )
    five_g_ready = query_df(
        """
        SELECT DISTINCT
            c.customer_id, c.full_name, c.city, p.plan_name, p.technology AS plan_technology,
            d.brand, d.model, d.device_5g_capable_flag
        FROM devices d
        JOIN customers c ON c.customer_id = d.customer_id
        JOIN subscriptions s ON s.subscription_id = d.subscription_id
        JOIN plans p ON p.plan_id = s.plan_id
        WHERE d.device_5g_capable_flag = 1 AND COALESCE(p.technology, '') != '5G'
        LIMIT ?
        """,
        (top_n,),
    )
    unresolved_complaints = query_df(
        """
        SELECT
            c.customer_id, c.full_name, c.city,
            COUNT(cp.complaint_id) AS open_complaints,
            MAX(cp.complaint_date) AS latest_complaint,
            GROUP_CONCAT(DISTINCT cp.complaint_category) AS categories
        FROM complaints cp
        JOIN customers c ON c.customer_id = cp.customer_id
        WHERE cp.status != 'Resolved'
        GROUP BY c.customer_id, c.full_name, c.city
        ORDER BY open_complaints DESC, latest_complaint DESC
        LIMIT ?
        """,
        (top_n,),
    )
    return {
        "High value + high churn risk": high_value_risk,
        "Overdue invoices + churn signal": overdue_risk,
        "5G capable device, non-5G plan": five_g_ready,
        "Open complaints needing follow-up": unresolved_complaints,
    }


@st.cache_data(show_spinner=False)
def customer_snapshot(customer_id: int) -> dict:
    profile = query_df(
        """
        SELECT
            c.customer_id, c.full_name, c.phone_number, c.email, c.city, c.governorate,
            c.customer_segment, c.customer_type, c.status, c.preferred_language,
            ch.risk_level, ROUND(ch.churn_score, 3) AS churn_score, ch.main_risk_reason, ch.recommended_action,
            vs.value_segment, ROUND(vs.arpu_jod, 2) AS arpu_jod, ROUND(vs.total_revenue_6m_jod, 2) AS revenue_6m_jod
        FROM customers c
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = c.customer_id
        LEFT JOIN customer_value_segments vs ON vs.customer_id = c.customer_id
        WHERE c.customer_id = ?
        """,
        (customer_id,),
    )
    subscriptions = query_df(
        """
        SELECT s.subscription_id, s.msisdn, s.service_type, s.status, p.plan_name, p.plan_category,
               p.technology, p.monthly_fee_jod, s.activation_date, s.contract_end_date
        FROM subscriptions s
        LEFT JOIN plans p ON p.plan_id = s.plan_id
        WHERE s.customer_id = ?
        ORDER BY s.primary_subscription_flag DESC, s.activation_date DESC
        """,
        (customer_id,),
    )
    invoices = query_df(
        """
        SELECT i.invoice_id, i.issue_date, i.due_date, ROUND(i.total_amount_jod, 2) AS total_amount_jod,
               i.payment_status, i.days_overdue
        FROM invoices i
        JOIN accounts a ON a.account_id = i.account_id
        WHERE a.customer_id = ?
        ORDER BY i.issue_date DESC
        LIMIT 10
        """,
        (customer_id,),
    )
    complaints = query_df(
        """
        SELECT complaint_date, complaint_category, severity, status, compensation_amount_jod
        FROM complaints
        WHERE customer_id = ?
        ORDER BY complaint_date DESC
        LIMIT 10
        """,
        (customer_id,),
    )
    usage = query_df(
        """
        SELECT summary_month, ROUND(total_revenue_jod, 2) AS revenue_jod,
               ROUND(data_used_gb, 2) AS data_used_gb, voice_minutes, complaints_count, payment_delay_days
        FROM customer_monthly_summary
        WHERE customer_id = ?
        ORDER BY summary_month
        """,
        (customer_id,),
    )
    return {"profile": profile, "subscriptions": subscriptions, "invoices": invoices, "complaints": complaints, "usage": usage}


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------

PALETTES = {
    "Dark": {
        "app_bg": "#07111f",
        "app_bg_2": "#0b1020",
        "surface": "rgba(15, 23, 42, 0.84)",
        "surface_2": "rgba(30, 41, 59, 0.72)",
        "surface_solid": "#111827",
        "sidebar": "rgba(10, 15, 30, 0.96)",
        "text": "#f8fafc",
        "muted": "#94a3b8",
        "subtle": "#cbd5e1",
        "border": "rgba(255,255,255,0.11)",
        "accent": "#7c3aed",
        "accent_2": "#00a3e0",
        "success": "#34d399",
        "warning": "#fbbf24",
        "danger": "#fb7185",
        "shadow": "0 24px 70px rgba(0, 0, 0, .38)",
        "plotly": "plotly_dark",
    },
    "Light": {
        "app_bg": "#f6f8fb",
        "app_bg_2": "#eef2ff",
        "surface": "rgba(255, 255, 255, 0.86)",
        "surface_2": "rgba(248, 250, 252, 0.92)",
        "surface_solid": "#ffffff",
        "sidebar": "rgba(255, 255, 255, 0.92)",
        "text": "#0f172a",
        "muted": "#64748b",
        "subtle": "#334155",
        "border": "rgba(15, 23, 42, 0.10)",
        "accent": "#6d28d9",
        "accent_2": "#0284c7",
        "success": "#059669",
        "warning": "#b45309",
        "danger": "#e11d48",
        "shadow": "0 20px 55px rgba(15, 23, 42, .12)",
        "plotly": "plotly_white",
    },
}

CHART_TYPES = {
    "Bar": "bar",
    "Horizontal bar": "horizontal_bar",
    "Line": "line",
    "Area": "area",
    "Pie": "pie",
    "Doughnut": "doughnut",
}

NAV_ITEMS = [
    ("Chat", "💬", "Copilot Chat"),
    ("Analytics", "📊", "Dynamic Analytics"),
    ("Customer", "👤", "Customer Workspace"),
    ("Chart Builder", "📈", "AI Chart Builder"),
    ("SQL Query Builder", "🧮", "SQL Lab"),
    ("Suggested Questions", "✨", "Prompt Library"),
]

SUGGESTED_QUESTIONS = [
    "Show me the full profile, plan, complaints, churn risk, invoices, and recommended action for customer ID 9.",
    "Find the top 10 customers with the highest churn score and explain why they are at risk.",
    "Which customers have overdue invoices and high churn risk?",
    "What are the most common complaint categories and how many are still unresolved?",
    "Which marketing campaigns have the best conversion rate?",
    "Which customer segments bring the most revenue in the last 6 months?",
    "Which cities have the highest number of affected customers from network events?",
    "Which customers have the highest data usage this month?",
    "Which customers generated the highest roaming cost?",
    "Which customers have 5G capable devices but are not on a 5G plan?",
    "Which add-ons are most used by customers?",
    "Which payment channels are used the most?",
    "Summarize recent support interactions by channel, reason, sentiment, and priority.",
]


def get_logo_data_uri() -> str:
    if not LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


LOGO_DATA_URI = get_logo_data_uri()


def apply_theme(theme_name: str) -> None:
    palette = PALETTES[theme_name]
    css = f"""
    <style>
    :root {{
      --app-bg: {palette['app_bg']};
      --app-bg-2: {palette['app_bg_2']};
      --surface: {palette['surface']};
      --surface-2: {palette['surface_2']};
      --surface-solid: {palette['surface_solid']};
      --sidebar: {palette['sidebar']};
      --text: {palette['text']};
      --muted: {palette['muted']};
      --subtle: {palette['subtle']};
      --border: {palette['border']};
      --accent: {palette['accent']};
      --accent-2: {palette['accent_2']};
      --success: {palette['success']};
      --warning: {palette['warning']};
      --danger: {palette['danger']};
      --shadow: {palette['shadow']};
      --radius-xl: 30px;
      --radius-lg: 22px;
      --radius-md: 16px;
      --sidebar-width: 340px;
    }}

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {{
      background:
        radial-gradient(circle at 82% 5%, color-mix(in srgb, var(--accent) 20%, transparent), transparent 32%),
        radial-gradient(circle at 10% 10%, color-mix(in srgb, var(--accent-2) 14%, transparent), transparent 30%),
        linear-gradient(135deg, var(--app-bg) 0%, var(--app-bg-2) 55%, var(--app-bg) 100%) !important;
      color: var(--text) !important;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    }}
    * {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important; }}
    header[data-testid="stHeader"] {{ background: transparent !important; }}
    [data-testid="stToolbar"] {{ color: var(--text) !important; }}
    .block-container {{ max-width: 1280px !important; padding: 1.15rem 1.6rem 7rem !important; }}
    h1, h2, h3, h4, h5, h6 {{ color: var(--text) !important; letter-spacing: -0.035em !important; }}
    p, label, span, li, div {{ color: inherit; }}

    section[data-testid="stSidebar"] {{
      width: var(--sidebar-width) !important;
      min-width: var(--sidebar-width) !important;
      background: transparent !important;
      border-right: 1px solid var(--border) !important;
    }}
    section[data-testid="stSidebar"] > div {{ padding: .85rem !important; background: transparent !important; }}
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
      min-height: calc(100vh - 1.7rem) !important;
      padding: 1rem .95rem 1.15rem !important;
      border-radius: 28px !important;
      background: var(--sidebar) !important;
      border: 1px solid var(--border) !important;
      box-shadow: var(--shadow) !important;
      backdrop-filter: blur(24px) saturate(140%);
    }}

    .brand-card, .hero-card, .glass-card {{
      background: var(--surface) !important;
      border: 1px solid var(--border) !important;
      box-shadow: var(--shadow) !important;
      backdrop-filter: blur(24px) saturate(140%);
      border-radius: var(--radius-xl);
    }}
    .brand-card {{ padding: 1rem; overflow: hidden; position: relative; margin-bottom: .95rem; }}
    .brand-card::after {{ content:""; position:absolute; right:-55px; bottom:-70px; width:150px; height:150px; border-radius:999px; background: color-mix(in srgb, var(--accent) 22%, transparent); }}
    .brand-row {{ display:flex; gap:.8rem; align-items:center; position:relative; z-index:2; }}
    .brand-icon {{ width:46px; height:46px; min-width:46px; border-radius:17px; display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg,var(--accent),var(--accent-2)); color:#fff; font-weight:900; box-shadow:0 14px 35px color-mix(in srgb, var(--accent) 28%, transparent); }}
    .brand-title {{ font-size:1rem; font-weight:900; line-height:1.08; color:var(--text); }}
    .brand-subtitle {{ font-size:.72rem; font-weight:800; color:var(--muted); margin-top:.15rem; }}
    .brand-copy {{ position:relative; z-index:2; color:var(--muted); font-size:.76rem; line-height:1.55; margin-top:.75rem; }}
    .sidebar-label {{ color:var(--muted); font-size:.66rem; font-weight:900; letter-spacing:.13em; text-transform:uppercase; margin:1.05rem 0 .45rem .15rem; }}
    .sidebar-divider {{ height:1px; margin:.9rem 0; background:linear-gradient(90deg,transparent,var(--border),transparent); }}
    .sidebar-footnote {{ margin-top:1rem; padding:.85rem; border-radius:18px; background:var(--surface-2); border:1px solid var(--border); color:var(--muted); font-size:.72rem; line-height:1.5; }}

    .hero-card {{
      position:relative; overflow:hidden; padding:1.55rem 1.7rem; min-height:158px; margin-bottom:1.25rem;
      background:
        radial-gradient(circle at 88% 22%, color-mix(in srgb, var(--accent) 26%, transparent), transparent 34%),
        radial-gradient(circle at 98% 88%, color-mix(in srgb, var(--accent-2) 18%, transparent), transparent 30%),
        var(--surface) !important;
    }}
    .hero-card::before {{ content:""; position:absolute; inset:0; background:linear-gradient(135deg, color-mix(in srgb, var(--surface-solid) 18%, transparent), transparent); pointer-events:none; }}
    .hero-inner {{ position:relative; z-index:2; max-width:850px; }}
    .eyebrow {{ color:var(--accent-2); font-weight:900; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; margin-bottom:.45rem; }}
    .hero-title {{ color:var(--text); font-weight:950; font-size:2rem; letter-spacing:-.055em; line-height:1.04; margin-bottom:.5rem; }}
    .hero-copy {{ color:var(--muted); font-size:.96rem; line-height:1.62; max-width:760px; }}
    .hero-logo {{ position:absolute; right:1.6rem; top:50%; transform:translateY(-50%); width:148px; opacity:.18; filter:grayscale(1); }}

    .mini-card {{ padding:1rem; border-radius:22px; background:var(--surface); border:1px solid var(--border); box-shadow:var(--shadow); min-height:110px; }}
    .mini-label {{ color:var(--muted); font-size:.74rem; font-weight:800; letter-spacing:.03em; }}
    .mini-value {{ color:var(--text); font-size:1.65rem; font-weight:950; letter-spacing:-.04em; margin-top:.18rem; }}
    .mini-note {{ color:var(--muted); font-size:.72rem; margin-top:.25rem; }}
    .status-pill {{ display:inline-flex; align-items:center; gap:.35rem; padding:.35rem .62rem; border-radius:999px; font-size:.72rem; font-weight:850; color:var(--text); background:var(--surface-2); border:1px solid var(--border); }}
    .chip-row {{ display:flex; gap:.45rem; flex-wrap:wrap; margin:.45rem 0 .9rem; }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.9rem; margin:.7rem 0 1.1rem; }}

    div[data-testid="stMetric"] {{ background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:22px !important; padding:17px !important; box-shadow:var(--shadow) !important; }}
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricDelta"] {{ color:var(--muted) !important; }}
    div[data-testid="stMetricValue"] {{ color:var(--text) !important; font-weight:950 !important; }}
    .stPlotlyChart {{ background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:24px !important; padding:.9rem !important; box-shadow:var(--shadow) !important; }}
    div[data-testid="stDataFrame"] {{ border-radius:20px !important; overflow:hidden !important; border:1px solid var(--border) !important; box-shadow:var(--shadow) !important; }}
    div[data-testid="stExpander"] {{ background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:20px !important; box-shadow:var(--shadow) !important; overflow:hidden !important; }}
    div[data-testid="stExpander"] summary {{ color:var(--text) !important; font-weight:850 !important; }}
    div[data-testid="stTabs"] button {{ color:var(--muted) !important; font-weight:850 !important; }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{ color:var(--text) !important; }}
    div[data-testid="stCodeBlock"] {{ border-radius:18px !important; overflow:hidden !important; border:1px solid var(--border) !important; }}
    div[data-testid="stAlert"] {{ border-radius:18px !important; }}

    input, textarea, [data-baseweb="select"] > div {{
      border-radius:16px !important;
      border-color:var(--border) !important;
      background:var(--surface-2) !important;
      color:var(--text) !important;
    }}
    input:focus, textarea:focus {{ box-shadow:0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent) !important; border-color:var(--accent) !important; }}

    div.stButton > button, div[data-testid="stDownloadButton"] > button {{
      min-height:42px !important; border-radius:15px !important; border:1px solid var(--border) !important;
      background:var(--surface-2) !important; color:var(--text) !important; font-weight:850 !important; box-shadow:none !important;
      transition: all .16s ease !important;
    }}
    div.stButton > button:hover, div[data-testid="stDownloadButton"] > button:hover {{
      transform:translateY(-1px); border-color:color-mix(in srgb, var(--accent) 45%, var(--border)) !important;
      background:color-mix(in srgb, var(--surface-solid) 72%, var(--accent) 8%) !important;
    }}
    div.stButton > button[kind="primary"], div[data-testid="stDownloadButton"] > button[kind="primary"] {{
      background:linear-gradient(135deg,var(--accent),var(--accent-2)) !important; color:#fff !important; border-color:transparent !important;
      box-shadow:0 16px 34px color-mix(in srgb, var(--accent) 28%, transparent) !important;
    }}
    section[data-testid="stSidebar"] div.stButton > button {{ width:100% !important; justify-content:flex-start !important; text-align:left !important; padding:.58rem .7rem !important; min-height:42px !important; box-shadow:none !important; }}
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{ justify-content:flex-start !important; }}
    section[data-testid="stSidebar"] .stTextInput input {{ min-height:42px !important; }}

    .st-key-chat_input_shell {{
      position: fixed; left: calc(var(--sidebar-width) + 50%); transform: translateX(-50%); bottom: 1.15rem; z-index: 1000;
      width: min(820px, calc(100vw - 445px)); padding:.42rem .48rem .42rem 1rem; border-radius:999px; background:var(--surface) !important;
      border:1px solid var(--border); box-shadow:var(--shadow); backdrop-filter:blur(24px) saturate(140%);
    }}
    .st-key-chat_input_shell [data-testid="stForm"] {{ border:0 !important; padding:0 !important; background:transparent !important; }}
    .st-key-chat_input_shell div[data-testid="stHorizontalBlock"] {{ align-items:center !important; gap:.45rem !important; }}
    .st-key-chat_input_shell input {{ min-height:48px !important; border:0 !important; background:transparent !important; box-shadow:none !important; color:var(--text) !important; }}
    .st-key-chat_input_shell button {{ width:48px !important; min-width:48px !important; height:48px !important; min-height:48px !important; padding:0 !important; border-radius:999px !important; justify-content:center !important; text-align:center !important; font-size:0 !important; }}
    .st-key-chat_input_shell button::after {{ content:"➜"; font-size:1.1rem; color:#fff; }}
    div[data-testid="stChatMessage"] {{ padding:.45rem 0 !important; background:transparent !important; }}
    div[data-testid="stChatMessageContent"] {{ color:var(--subtle) !important; line-height:1.68 !important; font-size:.95rem !important; }}
    div[data-testid="chatAvatarIcon-assistant"] {{ background:linear-gradient(135deg,var(--accent),var(--accent-2)) !important; }}

    @media (max-width: 980px) {{
      :root {{ --sidebar-width: 0px; }}
      section[data-testid="stSidebar"] {{ width:auto !important; min-width:auto !important; }}
      .block-container {{ padding-left:1rem !important; padding-right:1rem !important; }}
      .st-key-chat_input_shell {{ left:1rem; right:1rem; transform:none; width:auto; }}
      .hero-card {{ padding:1.2rem; min-height:auto; }}
      .hero-title {{ font-size:1.48rem; }}
      .hero-logo {{ position:relative; right:auto; top:auto; transform:none; width:92px; margin-top:1rem; }}
      .metric-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    }}
    @media (max-width: 620px) {{ .metric-grid {{ grid-template-columns:1fr; }} }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def plotly_layout(fig):
    theme_name = st.session_state.get("theme", "Dark")
    palette = PALETTES[theme_name]
    fig.update_layout(
        template=palette["plotly"],
        margin=dict(l=20, r=20, t=38, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=palette["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_chart(df: pd.DataFrame, chart_type: str, title: str, metric_label: str = "Value"):
    if df is None or df.empty:
        st.info("No matching data found for the selected filters.")
        return
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    if chart_type == "pie":
        fig = px.pie(df, names="label", values="value", title=title, height=430)
    elif chart_type == "doughnut":
        fig = px.pie(df, names="label", values="value", hole=0.52, title=title, height=430)
    elif chart_type == "line":
        fig = px.line(df, x="label", y="value", markers=True, title=title, height=430)
    elif chart_type == "area":
        fig = px.area(df, x="label", y="value", title=title, height=430)
    elif chart_type == "horizontal_bar":
        fig = px.bar(df.sort_values("value"), x="value", y="label", orientation="h", title=title, height=430, labels={"value": metric_label, "label": ""})
    else:
        fig = px.bar(df, x="label", y="value", title=title, height=430, labels={"value": metric_label, "label": ""})
    st.plotly_chart(plotly_layout(fig), use_container_width=True)


def safe_html(value) -> str:  # noqa: ANN001
    return html.escape("" if value is None else str(value))


def hero(title: str, subtitle: str, eyebrow: str = "ZAIN 360 COPILOT"):
    logo_html = f'<img class="hero-logo" src="{LOGO_DATA_URI}" alt="Zain Logo">' if LOGO_DATA_URI else ""
    st.markdown(
        f"""
        <div class="hero-card">
          <div class="hero-inner">
            <div class="eyebrow">{safe_html(eyebrow)}</div>
            <div class="hero-title">{safe_html(title)}</div>
            <div class="hero-copy">{safe_html(subtitle)}</div>
          </div>
          {logo_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="mini-card">
          <div class="mini-label">{safe_html(label)}</div>
          <div class="mini-value">{safe_html(value)}</div>
          <div class="mini-note">{safe_html(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_number(value, decimals=0, suffix="") -> str:  # noqa: ANN001
    try:
        val = float(value)
    except Exception:
        return str(value)
    if decimals == 0:
        return f"{val:,.0f}{suffix}"
    return f"{val:,.{decimals}f}{suffix}"


def dataframe_download(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    if df is not None and not df.empty:
        st.download_button(
            label,
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# Chat state
# -----------------------------------------------------------------------------


def default_assistant_message():
    return {
        "role": "assistant",
        "content": "Hello. Ask me a business question about customers, churn, billing, complaints, campaigns, network events, or usage insights.",
    }


def init_session_state():
    st.session_state.setdefault("theme", "Dark")
    st.session_state.setdefault("page", "Chat")
    st.session_state.setdefault("chat_search", "")
    st.session_state.setdefault("open_chat_menu_id", None)
    st.session_state.setdefault("rename_chat_id", None)
    st.session_state.setdefault("rename_chat_value", "")
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = [
            {"id": "chat_1", "title": "New Chat", "messages": [default_assistant_message()], "created_at": datetime.now().isoformat(timespec="seconds")}
        ]
        st.session_state.current_chat_id = "chat_1"
    st.session_state.setdefault("current_chat_id", st.session_state.chat_sessions[0]["id"])


def current_chat():
    init_session_state()
    for chat in st.session_state.chat_sessions:
        if chat["id"] == st.session_state.current_chat_id:
            return chat
    st.session_state.current_chat_id = st.session_state.chat_sessions[0]["id"]
    return st.session_state.chat_sessions[0]


def title_from_question(question: str) -> str:
    cleaned = " ".join(question.split())
    return cleaned[:38] + "..." if len(cleaned) > 38 else cleaned or "New Chat"


def create_new_chat():
    next_id = f"chat_{int(time.time() * 1000)}"
    st.session_state.chat_sessions.insert(
        0,
        {"id": next_id, "title": "New Chat", "messages": [default_assistant_message()], "created_at": datetime.now().isoformat(timespec="seconds")},
    )
    st.session_state.current_chat_id = next_id
    st.session_state.page = "Chat"
    st.session_state.open_chat_menu_id = None
    st.session_state.rename_chat_id = None


def rename_chat(chat_id: str, new_title: str):
    title = " ".join(new_title.split()).strip()
    if not title:
        return
    for chat in st.session_state.chat_sessions:
        if chat["id"] == chat_id:
            chat["title"] = title
            break
    st.session_state.rename_chat_id = None
    st.session_state.open_chat_menu_id = None


def delete_chat(chat_id: str):
    st.session_state.chat_sessions = [c for c in st.session_state.chat_sessions if c["id"] != chat_id]
    if not st.session_state.chat_sessions:
        create_new_chat()
    if st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat_id = st.session_state.chat_sessions[0]["id"]
    st.session_state.open_chat_menu_id = None


def export_chat_markdown(chat: dict) -> str:
    lines = [f"# {chat['title']}", "", f"Exported: {datetime.now().isoformat(timespec='seconds')}", ""]
    for message in chat["messages"]:
        lines.append(f"## {message['role'].title()}")
        lines.append(message.get("content", ""))
        if message.get("sql"):
            lines.append("\n```sql")
            lines.append(message["sql"])
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------


def show_chat():
    chat = current_chat()
    hero(
        "Customer 360 Chat",
        f"{chat['title']} · Ask natural-language questions and inspect the SQL behind the answer when available.",
    )

    quick_cols = st.columns(4)
    quick_prompts = [
        "Top churn risks",
        "Open complaints",
        "Overdue invoices",
        "Best campaigns",
    ]
    quick_questions = [
        "Find the top 10 customers with the highest churn score and explain why they are at risk.",
        "What are the most common complaint categories and how many are still unresolved?",
        "Which customers have overdue invoices and high churn risk?",
        "Which marketing campaigns have the best conversion rate?",
    ]
    for col, label, q in zip(quick_cols, quick_prompts, quick_questions):
        if col.button(label, use_container_width=True):
            submit_prompt(q)
            st.rerun()

    for message in chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sql"):
                with st.expander("SQL query used"):
                    st.code(message["sql"], language="sql")

    chat_bar = st.container(key="chat_input_shell")
    with chat_bar:
        with st.form("chat_form", clear_on_submit=True):
            input_col, send_col = st.columns([8, 1])
            with input_col:
                prompt = st.text_input(
                    "Question",
                    placeholder="Ask Zain 360 Copilot anything about the database...",
                    label_visibility="collapsed",
                )
            with send_col:
                submitted = st.form_submit_button("Send", type="primary")

    if submitted and prompt.strip():
        submit_prompt(prompt.strip())
        st.rerun()


def submit_prompt(prompt: str):
    chat = current_chat()
    if chat["title"] == "New Chat":
        chat["title"] = title_from_question(prompt)
    chat["messages"].append({"role": "user", "content": prompt})
    try:
        with st.spinner("Analyzing database..."):
            payload = ask_sql_agent_payload(prompt)
        answer = payload.get("answer", "No answer was returned.")
        sql = payload.get("sql", "")
    except Exception as exc:
        answer = (
            "I could not complete the AI answer. "
            f"Reason: {type(exc).__name__}: {exc}\n\n"
            "Tip: confirm that OPENAI_API_KEY is configured in Streamlit secrets and that the database file is present."
        )
        sql = ""
    chat["messages"].append({"role": "assistant", "content": answer, "sql": sql})


def show_analytics():
    hero(
        "Dynamic Analytics",
        "Adjust filters, select metrics, change grouping, switch chart types, and export the filtered analysis as CSV.",
    )
    opts = get_filter_options()
    if not opts["months"]:
        st.error("No monthly summary data was found.")
        return

    with st.expander("Analytics controls", expanded=True):
        row1 = st.columns(4)
        cities = row1[0].multiselect("City", opts["cities"], placeholder="All cities")
        segments = row1[1].multiselect("Customer segment", opts["segments"], placeholder="All segments")
        risk_levels = row1[2].multiselect("Risk level", opts["risk_levels"], placeholder="All risks")
        value_segments = row1[3].multiselect("Value segment", opts["value_segments"], placeholder="All values")

        row2 = st.columns([1.2, 1.2, 1, 1, 1])
        month_start = row2[0].selectbox("Start month", opts["months"], index=0)
        month_end = row2[1].selectbox("End month", opts["months"], index=len(opts["months"]) - 1)
        metric = row2[2].selectbox(
            "Metric",
            ["Customers", "Revenue", "Avg Churn Score", "Avg ARPU", "Data Usage GB", "Complaints", "Payment Delay Days"],
            index=1,
        )
        group_by = row2[3].selectbox(
            "Group by",
            ["City", "Customer Segment", "Risk Level", "Value Segment", "Service Type", "Plan Category", "Plan Technology"],
        )
        top_n = row2[4].slider("Top results", 3, 25, 10)

        chart_label = st.radio("Chart style", list(CHART_TYPES.keys()), horizontal=True, index=0)

    if month_start > month_end:
        st.warning("Start month is after end month. Please adjust the range.")
        return

    kpis = analytics_kpis(cities, segments, risk_levels, value_segments, month_start, month_end)
    cols = st.columns(4)
    with cols[0]:
        kpi_card("Customers", format_number(kpis["Customers"]), "Filtered customer base")
    with cols[1]:
        kpi_card("Revenue", format_number(kpis["Revenue"], 2, " JOD"), f"{month_start} to {month_end}")
    with cols[2]:
        kpi_card("Avg Churn", format_number(kpis["Avg Churn"], 3), f"High-risk: {kpis['High Risk']:,}")
    with cols[3]:
        kpi_card("Avg ARPU", format_number(kpis["Avg ARPU"], 2, " JOD"), f"Avg delay: {kpis['Avg Delay']} days")

    chart_df, sql, metric_alias = dynamic_chart_data(
        metric,
        group_by,
        cities,
        segments,
        risk_levels,
        value_segments,
        month_start,
        month_end,
        top_n,
    )
    render_chart(chart_df, CHART_TYPES[chart_label], f"{metric} by {group_by}", metric_alias)

    action_cols = st.columns([1, 1, 2])
    with action_cols[0]:
        dataframe_download(chart_df, "zain_dynamic_analytics.csv")
    with action_cols[1]:
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with st.expander("SQL behind this analysis"):
        st.code(sql, language="sql")

    st.markdown("### Smart Segments")
    smart = get_smart_segments(top_n=top_n)
    tabs = st.tabs(list(smart.keys()))
    for tab, (title, df) in zip(tabs, smart.items()):
        with tab:
            st.caption("Use these lists for next-best-action workflows and retention follow-up.")
            st.dataframe(df, use_container_width=True, hide_index=True)
            dataframe_download(df, f"{title.lower().replace(' ', '_').replace('+', 'and')}.csv", "Download segment")


def show_customer_workspace():
    hero(
        "Customer Workspace",
        "Look up a customer, review their profile, churn risk, subscriptions, invoices, complaints, and monthly trend in one place.",
    )
    total_customers = int(scalar("SELECT COUNT(*) AS value FROM customers") or 0)
    col_a, col_b = st.columns([1, 3])
    with col_a:
        customer_id = st.number_input("Customer ID", min_value=1, max_value=max(total_customers, 1), value=9, step=1)
    with col_b:
        st.markdown('<div class="chip-row"><span class="status-pill">Search by ID</span><span class="status-pill">360 profile</span><span class="status-pill">Retention action</span><span class="status-pill">Billing view</span></div>', unsafe_allow_html=True)

    data = customer_snapshot(int(customer_id))
    profile = data["profile"]
    if profile.empty:
        st.warning("No customer found with this ID.")
        return
    p = profile.iloc[0].to_dict()

    st.markdown("### Profile Summary")
    cols = st.columns(4)
    with cols[0]:
        kpi_card("Customer", p.get("full_name", "-"), f"ID {p.get('customer_id')}")
    with cols[1]:
        kpi_card("Risk Level", p.get("risk_level", "-"), f"Score {p.get('churn_score', '-')}")
    with cols[2]:
        kpi_card("Value Segment", p.get("value_segment", "-"), f"ARPU {p.get('arpu_jod', 0)} JOD")
    with cols[3]:
        kpi_card("City", p.get("city", "-"), p.get("customer_segment", "-"))

    st.info(f"Recommended action: {p.get('recommended_action') or 'No recommendation found.'}")
    st.caption(f"Main risk reason: {p.get('main_risk_reason') or 'Not available.'}")

    usage = data["usage"]
    if not usage.empty:
        usage_long = usage[["summary_month", "revenue_jod", "data_used_gb"]].melt(id_vars="summary_month", var_name="Metric", value_name="value")
        fig = px.line(usage_long, x="summary_month", y="value", color="Metric", markers=True, title="Monthly customer trend", height=390)
        st.plotly_chart(plotly_layout(fig), use_container_width=True)

    tabs = st.tabs(["Subscriptions", "Invoices", "Complaints", "Raw Profile"])
    with tabs[0]:
        st.dataframe(data["subscriptions"], use_container_width=True, hide_index=True)
    with tabs[1]:
        st.dataframe(data["invoices"], use_container_width=True, hide_index=True)
    with tabs[2]:
        st.dataframe(data["complaints"], use_container_width=True, hide_index=True)
    with tabs[3]:
        st.dataframe(profile, use_container_width=True, hide_index=True)

    prompt = f"Show me the full profile, plan, complaints, churn risk, invoices, and recommended action for customer ID {int(customer_id)}."
    if st.button("Ask Copilot about this customer", type="primary", use_container_width=True):
        st.session_state.page = "Chat"
        submit_prompt(prompt)
        st.rerun()


def show_chart_builder():
    hero(
        "AI Chart Builder",
        "Describe the chart you want, pick a chart type, and the app will generate a safe database-backed visualization.",
    )
    examples = [
        "Build a chart based on customer with ID = 9 by their complaints type and number.",
        "Show the top 10 cities by affected customers from network events.",
        "Compare campaigns by conversion rate.",
        "Show average revenue by value segment.",
    ]
    col1, col2 = st.columns([2, 1])
    with col1:
        question = st.text_area("Chart request", value=examples[0], height=130)
    with col2:
        chart_label = st.selectbox("Chart type", list(CHART_TYPES.keys()))
        st.caption("Tip: use top N, a customer ID, a city, or a business topic such as churn, complaints, revenue, campaigns, or network impact.")

    if st.button("Create chart", type="primary", use_container_width=True):
        with st.spinner("Creating chart..."):
            chart = build_chart_from_question(question, CHART_TYPES[chart_label])
        st.session_state.last_ai_chart = chart

    chart = st.session_state.get("last_ai_chart")
    if chart:
        df = pd.DataFrame(chart.get("rows") or [])
        if "label" in df.columns and "value" in df.columns:
            render_chart(df[["label", "value"]], chart.get("chart_type", CHART_TYPES[chart_label]), chart.get("title", "Generated chart"), chart.get("metric", "Value"))
            st.caption(chart.get("summary", ""))
            with st.expander("Generated chart data"):
                st.dataframe(df, use_container_width=True, hide_index=True)
                dataframe_download(df, "generated_chart_data.csv")
        else:
            st.warning(chart.get("summary") or "No chart data returned.")


def run_sql_callback(key_prefix: str):
    sql = st.session_state.get(f"{key_prefix}_sql_editor", "").strip()
    try:
        st.session_state[f"{key_prefix}_sql_result"] = execute_sql_query(sql)
        st.session_state[f"{key_prefix}_sql_error"] = ""
    except Exception as exc:
        st.session_state[f"{key_prefix}_sql_result"] = None
        st.session_state[f"{key_prefix}_sql_error"] = f"{type(exc).__name__}: {exc}"


def show_sql_builder():
    hero(
        "SQL Lab",
        "Run safe read-only SELECT queries, inspect returned rows, and export results for validation or reporting.",
    )
    default_sql = "SELECT COUNT(*) AS total_customers FROM customers"
    key_prefix = "sql_lab"
    editor_key = f"{key_prefix}_sql_editor"
    st.session_state.setdefault(editor_key, default_sql)

    with st.expander("Database tables", expanded=False):
        st.dataframe(get_table_counts(), use_container_width=True, hide_index=True)

    st.text_area("SQL", key=editor_key, height=200)
    if st.button("Run read-only query", type="primary", use_container_width=True):
        run_sql_callback(key_prefix)

    error = st.session_state.get(f"{key_prefix}_sql_error", "")
    result = st.session_state.get(f"{key_prefix}_sql_result")
    if error:
        st.error(f"Query failed: {error}")
    elif result:
        rows = result.get("rows", [])
        st.success(f"Returned {len(rows)} rows.")
        df = pd.DataFrame(rows)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
            dataframe_download(df, "sql_lab_results.csv")
        else:
            st.info("Query ran successfully but returned no rows.")
        with st.expander("Executed SQL"):
            st.code(result.get("sql", ""), language="sql")


def show_suggested_questions():
    hero(
        "Prompt Library",
        "Use ready-made telecom analytics prompts, then continue the conversation in the Copilot chat.",
    )
    cols = st.columns(2)
    for idx, question in enumerate(SUGGESTED_QUESTIONS):
        with cols[idx % 2]:
            with st.container(border=True):
                st.markdown(f"**{question}**")
                if st.button("Send to chat", key=f"suggested_{idx}", use_container_width=True):
                    st.session_state.page = "Chat"
                    submit_prompt(question)
                    st.rerun()


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-card">
              <div class="brand-row">
                <div class="brand-icon">AI</div>
                <div>
                  <div class="brand-title">Zain 360 Copilot</div>
                  <div class="brand-subtitle">Customer intelligence workspace</div>
                </div>
              </div>
              <div class="brand-copy">Chat, dynamic analytics, customer view, SQL validation, and AI-generated charts in one Streamlit app.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.theme = st.radio("Theme", ["Dark", "Light"], horizontal=True, label_visibility="collapsed")

        if st.button("＋ New Chat", type="primary", use_container_width=True):
            create_new_chat()
            st.rerun()

        st.markdown('<div class="sidebar-label">Workspace</div>', unsafe_allow_html=True)
        for page, icon, label in NAV_ITEMS:
            btn_type = "primary" if st.session_state.page == page else "secondary"
            if st.button(f"{icon} {label}", key=f"nav_{page}", type=btn_type, use_container_width=True):
                st.session_state.page = page
                st.session_state.open_chat_menu_id = None
                st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">Conversations</div>', unsafe_allow_html=True)
        st.text_input("Search chats", key="chat_search", placeholder="Search chat history...", label_visibility="collapsed")
        search = st.session_state.chat_search.lower().strip()
        visible_chats = [c for c in st.session_state.chat_sessions if search in c["title"].lower()]

        for chat in visible_chats[:12]:
            cols = st.columns([8, 1.4])
            with cols[0]:
                label = "💬 " + chat["title"]
                btn_type = "primary" if st.session_state.current_chat_id == chat["id"] and st.session_state.page == "Chat" else "secondary"
                if st.button(label, key=f"select_{chat['id']}", type=btn_type, use_container_width=True):
                    st.session_state.current_chat_id = chat["id"]
                    st.session_state.page = "Chat"
                    st.session_state.open_chat_menu_id = None
                    st.rerun()
            with cols[1]:
                if st.button("⋯", key=f"menu_{chat['id']}", use_container_width=True):
                    st.session_state.open_chat_menu_id = None if st.session_state.open_chat_menu_id == chat["id"] else chat["id"]
                    st.rerun()

            if st.session_state.open_chat_menu_id == chat["id"]:
                action_cols = st.columns(3)
                if action_cols[0].button("Rename", key=f"rename_{chat['id']}", use_container_width=True):
                    st.session_state.rename_chat_id = chat["id"]
                    st.session_state.rename_chat_value = chat["title"]
                    st.session_state.open_chat_menu_id = None
                    st.rerun()
                if action_cols[1].download_button(
                    "Export",
                    data=export_chat_markdown(chat).encode("utf-8"),
                    file_name=f"{chat['title'].replace(' ', '_')[:40]}.md",
                    mime="text/markdown",
                    use_container_width=True,
                ):
                    pass
                if action_cols[2].button("Delete", key=f"delete_{chat['id']}", use_container_width=True):
                    delete_chat(chat["id"])
                    st.rerun()

            if st.session_state.rename_chat_id == chat["id"]:
                new_title = st.text_input(
                    "Rename chat",
                    value=st.session_state.rename_chat_value,
                    key=f"rename_input_{chat['id']}",
                    label_visibility="collapsed",
                )
                rename_cols = st.columns(2)
                if rename_cols[0].button("Save", key=f"save_{chat['id']}", type="primary", use_container_width=True):
                    rename_chat(chat["id"], new_title)
                    st.rerun()
                if rename_cols[1].button("Cancel", key=f"cancel_{chat['id']}", use_container_width=True):
                    st.session_state.rename_chat_id = None
                    st.rerun()

        st.markdown(
            f"""
            <div class="sidebar-footnote">
              <b>Database:</b> {safe_html(DB_PATH.name)}<br>
              <b>Status:</b> {'Connected' if DB_PATH.exists() else 'Missing'}<br>
              Built for customer view, churn, billing, complaints, campaigns, usage, and network impact.
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    init_session_state()
    apply_theme(st.session_state.theme)
    render_sidebar()

    if not DB_PATH.exists():
        hero("Database Missing", "Place zain_customer_360_ai_demo.db next to this Streamlit app, then rerun the app.")
        st.stop()

    page = st.session_state.page
    if page == "Chat":
        show_chat()
    elif page == "Analytics":
        show_analytics()
    elif page == "Customer":
        show_customer_workspace()
    elif page == "Chart Builder":
        show_chart_builder()
    elif page == "SQL Query Builder":
        show_sql_builder()
    else:
        show_suggested_questions()


if __name__ == "__main__":
    main()
