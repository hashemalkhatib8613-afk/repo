import os
import time
import base64
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "zain-logo.png"
DB_PATH = APP_DIR / "zain_customer_360_ai_demo.db"


def load_streamlit_secret():
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
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_logo_data_uri():
    if not LOGO_PATH.exists():
        return ""

    image_bytes = LOGO_PATH.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


LOGO_DATA_URI = get_logo_data_uri()


st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');

      :root {
        --horizon-bg: #fdfeff;
        --horizon-page-bg: #f4f7fe;
        --horizon-white: #ffffff;
        --horizon-navy-900: #0b1437;
        --horizon-navy-800: #111c44;
        --horizon-navy-700: #1b254b;
        --horizon-text: #1b2559;
        --horizon-muted: #707eae;
        --horizon-muted-2: #a3aed0;
        --horizon-border: #e9edf7;
        --horizon-border-2: #e0e5f2;
        --horizon-brand: #422afb;
        --horizon-brand-2: #7551ff;
        --horizon-brand-soft: #e9e3ff;
        --horizon-card-shadow: 14px 17px 40px 4px rgba(112, 144, 176, 0.08);
        --horizon-soft-shadow: 0 18px 40px rgba(112, 144, 176, 0.12);
      }

      html, body, [class*="css"] {
        font-family: "Plus Jakarta Sans", sans-serif !important;
      }

      .stApp {
        background: var(--horizon-page-bg);
        color: var(--horizon-text);
      }

      header[data-testid="stHeader"] {
        background: transparent;
      }

      .block-container {
        max-width: 1120px;
        padding-top: 1.25rem;
        padding-bottom: 8rem;
      }

      /* ================================
         SIDEBAR - HORIZON TEMPLATE STYLE
         ================================ */

      section[data-testid="stSidebar"] {
        background: var(--horizon-page-bg);
        border-right: 0;
        width: 305px !important;
      }

      section[data-testid="stSidebar"] > div {
        padding: 1rem 0.55rem 1rem 0.95rem;
      }

      section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        background: var(--horizon-white);
        border-radius: 14px;
        min-height: calc(100vh - 32px);
        margin-top: 0;
        padding: 1.15rem 1rem 1.3rem;
        box-shadow: var(--horizon-card-shadow);
        border: 1px solid rgba(226, 232, 240, 0.65);
      }

      .horizon-logo-wrap {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1.15rem 0 1.25rem;
      }

      .horizon-logo-text {
        color: var(--horizon-navy-700);
        font-size: 1.45rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        line-height: 1;
      }

      .horizon-logo-dot {
        color: var(--horizon-brand);
      }

      .horizon-logo-sub {
        color: var(--horizon-muted);
        font-size: 0.68rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: 0.17em;
        text-transform: uppercase;
        margin-top: 0.35rem;
      }

      .horizon-divider {
        height: 1px;
        background: var(--horizon-border);
        margin: 0 -1rem 1.2rem;
      }

      .sidebar-section-label {
        color: var(--horizon-muted);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        margin: 0.7rem 0 0.5rem 0.25rem;
      }

      .sidebar-footnote {
        margin-top: 1.1rem;
        border-top: 1px solid var(--horizon-border);
        padding-top: 0.85rem;
        color: var(--horizon-muted);
        font-size: 0.72rem;
        line-height: 1.55;
      }

      section[data-testid="stSidebar"] details {
        border: 0 !important;
      }

      section[data-testid="stSidebar"] details summary {
        color: var(--horizon-navy-700) !important;
        font-weight: 700;
        font-size: 0.86rem;
      }

      section[data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        min-height: 42px;
        border-radius: 14px;
        border: 1px solid transparent;
        background: transparent;
        color: var(--horizon-muted);
        font-weight: 700;
        font-size: 0.84rem;
        text-align: left;
        justify-content: flex-start;
        padding: 0.6rem 0.75rem;
        box-shadow: none;
        transition: all 0.18s ease;
      }

      section[data-testid="stSidebar"] div.stButton > button:hover {
        background: #f7f9ff;
        border-color: var(--horizon-border);
        color: var(--horizon-brand);
        transform: none;
      }

      section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: var(--horizon-brand-soft) !important;
        border-color: var(--horizon-brand-soft) !important;
        color: var(--horizon-brand) !important;
        box-shadow: none !important;
        font-weight: 800 !important;
      }

      section[data-testid="stSidebar"] [class*="st-key-sidebar_new_chat"] button {
        background: linear-gradient(135deg, var(--horizon-brand), var(--horizon-brand-2)) !important;
        color: #ffffff !important;
        justify-content: center !important;
        text-align: center !important;
        min-height: 44px !important;
        border-radius: 45px !important;
        box-shadow: 0 16px 28px rgba(66, 42, 251, 0.22) !important;
      }

      section[data-testid="stSidebar"] [class*="st-key-sidebar_search_fake"] button {
        background: var(--horizon-navy-700) !important;
        color: #ffffff !important;
        border-radius: 999px !important;
        min-height: 44px !important;
        padding: 0 !important;
        justify-content: center !important;
        text-align: center !important;
      }

      /* ================================
         CHAT ITEMS IN SIDEBAR
         ================================ */

      section[data-testid="stSidebar"] [class*="st-key-chat_item_"] {
        border-radius: 14px;
        padding: 0.1rem 0.05rem;
        margin-bottom: 0.2rem;
        background: transparent;
      }

      section[data-testid="stSidebar"] [class*="st-key-chat_item_"] div[data-testid="stHorizontalBlock"] {
        align-items: center;
        gap: 0.05rem;
      }

      section[data-testid="stSidebar"] [class*="st-key-chat_select_wrap_"] div.stButton > button {
        width: 100% !important;
        min-height: 38px !important;
        border-radius: 12px !important;
        padding: 0.48rem 0.6rem !important;
        font-size: 0.78rem !important;
        color: var(--horizon-muted) !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        overflow: hidden;
      }

      section[data-testid="stSidebar"] [class*="st-key-chat_select_wrap_"] div.stButton > button[kind="primary"] {
        color: var(--horizon-brand) !important;
        background: var(--horizon-brand-soft) !important;
        border: none !important;
        box-shadow: none !important;
      }

      section[data-testid="stSidebar"] [class*="st-key-chat_menu_wrap_"],
      section[data-testid="stSidebar"] [class*="st-key-chat_menu_wrap_"] div,
      section[data-testid="stSidebar"] [class*="st-key-chat_menu_wrap_"] div.stButton {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
      }

      section[data-testid="stSidebar"] [class*="st-key-chat_menu_wrap_"] button,
      section[data-testid="stSidebar"] [class*="st-key-menu_toggle_"] button {
        width: 24px !important;
        min-width: 24px !important;
        height: 24px !important;
        min-height: 24px !important;
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: var(--horizon-muted-2) !important;
        font-size: 1rem !important;
        font-weight: 900 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
      }

      section[data-testid="stSidebar"] [class*="st-key-chat_menu_wrap_"] button:hover,
      section[data-testid="stSidebar"] [class*="st-key-menu_toggle_"] button:hover {
        background: transparent !important;
        color: var(--horizon-brand) !important;
        transform: none !important;
      }

      section[data-testid="stSidebar"] [class*="st-key-chat_actions_wrap_"] {
        margin: 0.2rem 0 0.35rem 0.5rem;
      }

      section[data-testid="stSidebar"] [class*="st-key-chat_actions_wrap_"] div.stButton > button {
        min-height: 31px !important;
        border-radius: 10px !important;
        padding: 0.35rem 0.55rem !important;
        font-size: 0.74rem !important;
        justify-content: center !important;
        text-align: center !important;
        background: #f7f9ff !important;
        border: 1px solid var(--horizon-border) !important;
      }

      section[data-testid="stSidebar"] [class*="st-key-chat_rename_wrap_"] input {
        min-height: 38px;
        border-radius: 12px;
        background: #ffffff !important;
      }

      /* ================================
         MAIN HEADER
         ================================ */

      .page-brand-card {
        position: relative;
        width: 100%;
        min-height: 132px;
        border-radius: 24px;
        padding: 1.35rem 1.55rem;
        margin-bottom: 1.25rem;
        overflow: hidden;
        background: var(--horizon-white);
        border: 1px solid var(--horizon-border);
        box-shadow: var(--horizon-card-shadow);
      }

      .page-brand-card::before {
        content: "";
        position: absolute;
        right: -75px;
        top: -95px;
        width: 260px;
        height: 260px;
        border-radius: 999px;
        background: rgba(66, 42, 251, 0.08);
      }

      .page-brand-card::after {
        content: "";
        position: absolute;
        right: 110px;
        bottom: -125px;
        width: 260px;
        height: 260px;
        border-radius: 999px;
        background: rgba(117, 81, 255, 0.07);
      }

      .page-brand-content {
        position: relative;
        z-index: 2;
        max-width: 760px;
      }

      .page-brand-eyebrow {
        color: var(--horizon-brand);
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
      }

      .page-brand-title {
        color: var(--horizon-text);
        font-size: 1.95rem;
        font-weight: 900;
        letter-spacing: -0.05em;
        line-height: 1.08;
        margin-bottom: 0.5rem;
      }

      .page-brand-copy {
        color: var(--horizon-muted);
        font-size: 0.92rem;
        line-height: 1.6;
        max-width: 730px;
      }

      .page-brand-logo {
        position: absolute;
        z-index: 3;
        right: 1.4rem;
        top: 50%;
        transform: translateY(-50%);
        width: 128px;
        max-width: 128px;
        height: auto;
        object-fit: contain;
        opacity: 0.32;
        filter: grayscale(1);
      }

      /* ================================
         MAIN CARDS / CHARTS / DATA
         ================================ */

      div[data-testid="stMetric"] {
        background: var(--horizon-white);
        border: 1px solid var(--horizon-border);
        border-radius: 20px;
        padding: 18px;
        box-shadow: var(--horizon-card-shadow);
      }

      div[data-testid="stMetric"] label {
        color: var(--horizon-muted) !important;
        font-weight: 700 !important;
      }

      div[data-testid="stMetricValue"] {
        color: var(--horizon-text) !important;
        font-weight: 900 !important;
      }

      .stPlotlyChart {
        background: var(--horizon-white);
        border: 1px solid var(--horizon-border);
        border-radius: 22px;
        padding: 1rem;
        box-shadow: var(--horizon-card-shadow);
      }

      div[data-testid="stDataFrame"] {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid var(--horizon-border);
        box-shadow: var(--horizon-card-shadow);
      }

      div[data-testid="stExpander"] {
        border: 1px solid var(--horizon-border);
        border-radius: 18px;
        background: var(--horizon-white);
        box-shadow: var(--horizon-card-shadow);
        overflow: hidden;
      }

      div[data-testid="stCodeBlock"] {
        border-radius: 16px;
        border: 1px solid var(--horizon-border);
        overflow: hidden;
      }

      div[data-testid="stAlert"] {
        border-radius: 16px;
      }

      h1, h2, h3, h4 {
        color: var(--horizon-text) !important;
        letter-spacing: -0.04em;
      }

      /* ================================
         CHAT AREA
         ================================ */

      div[data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.35rem 0;
      }

      div[data-testid="stChatMessageContent"] {
        color: var(--horizon-text);
        font-size: 0.92rem;
        line-height: 1.65;
      }

      div[data-testid="stChatMessageContent"] p {
        margin-bottom: 0.7rem;
      }

      div[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
        background: linear-gradient(135deg, var(--horizon-brand), var(--horizon-brand-2));
      }

      div[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] {
        background: var(--horizon-navy-700);
      }

      /* ================================
         FORMS / BUTTONS
         ================================ */

      textarea,
      input {
        border-radius: 16px !important;
        border-color: var(--horizon-border-2) !important;
        background: var(--horizon-white) !important;
        color: var(--horizon-text) !important;
      }

      textarea:focus,
      input:focus {
        border-color: var(--horizon-brand) !important;
        box-shadow: 0 0 0 3px rgba(66, 42, 251, 0.10) !important;
      }

      .main div.stButton > button {
        width: auto;
        min-height: 42px;
        border-radius: 16px;
        border: 1px solid var(--horizon-border);
        background: var(--horizon-white);
        color: var(--horizon-text);
        font-weight: 800;
        justify-content: center;
        text-align: center;
        padding: 0.58rem 0.95rem;
        box-shadow: var(--horizon-card-shadow);
      }

      .main div.stButton > button:hover {
        border-color: var(--horizon-brand);
        color: var(--horizon-brand);
        background: var(--horizon-white);
        transform: translateY(-1px);
      }

      .main div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--horizon-brand), var(--horizon-brand-2)) !important;
        color: #ffffff !important;
        border-color: var(--horizon-brand) !important;
        box-shadow: 0 14px 28px rgba(66, 42, 251, 0.22) !important;
      }

      /* ================================
         FIXED CHAT INPUT
         ================================ */

      .st-key-fixed_chat_bar {
        position: fixed;
        left: calc(305px + 50%);
        transform: translateX(-50%);
        bottom: 1.25rem;
        z-index: 1001;
        width: min(740px, calc(100vw - 410px));
        border: 1px solid var(--horizon-border-2);
        border-radius: 45px;
        padding: 0.42rem 0.45rem 0.42rem 0.95rem;
        background: rgba(255, 255, 255, 0.97);
        box-shadow: 0 18px 42px rgba(112, 144, 176, 0.22);
        backdrop-filter: blur(18px);
      }

      .st-key-fixed_chat_bar [data-testid="stForm"] {
        border: 0;
        padding: 0;
        background: transparent;
      }

      .st-key-fixed_chat_bar div[data-testid="stHorizontalBlock"] {
        align-items: center;
        gap: 0.5rem;
      }

      .st-key-fixed_chat_bar div[data-testid="column"] {
        padding-left: 0 !important;
        padding-right: 0 !important;
      }

      .st-key-fixed_chat_bar input {
        border: 0 !important;
        min-height: 46px;
        background: transparent !important;
        box-shadow: none !important;
        font-size: 0.86rem !important;
        color: var(--horizon-text) !important;
      }

      .st-key-fixed_chat_bar input:focus {
        border: 0 !important;
        box-shadow: none !important;
        outline: none !important;
      }

      .st-key-fixed_chat_bar button {
        width: 46px !important;
        min-width: 46px !important;
        height: 46px !important;
        min-height: 46px !important;
        border-radius: 999px !important;
        padding: 0 !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        background: linear-gradient(135deg, var(--horizon-brand), var(--horizon-brand-2)) !important;
        color: #ffffff !important;
        border: 0 !important;
        box-shadow: 0 12px 28px rgba(66, 42, 251, 0.30) !important;
        font-size: 0 !important;
      }

      .st-key-fixed_chat_bar button::after {
        content: "➜";
        font-size: 1.05rem;
        line-height: 1;
      }

      @media (max-width: 900px) {
        section[data-testid="stSidebar"] {
          width: auto !important;
        }

        .block-container {
          padding-left: 1rem;
          padding-right: 1rem;
        }

        .st-key-fixed_chat_bar {
          left: 1rem;
          right: 1rem;
          transform: none;
          width: auto;
        }

        .page-brand-card {
          padding: 1.15rem;
          min-height: auto;
        }

        .page-brand-logo {
          position: relative;
          top: auto;
          right: auto;
          transform: none;
          display: block;
          width: 92px;
          margin-top: 1rem;
        }

        .page-brand-title {
          font-size: 1.42rem;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


CHART_TYPES = {
    "Bar chart": "bar",
    "Horizontal bar": "horizontal_bar",
    "Pie chart": "pie",
    "Doughnut chart": "doughnut",
    "Line chart": "line",
    "Area chart": "area",
}

SUGGESTED_QUESTIONS = [
    "Show me the full profile, plan, complaints, churn risk, invoices, and recommended action for customer ID 9.",
    "Find the top 10 customers with the highest churn score and explain why they are at risk.",
    "Which customers have overdue invoices and high churn risk?",
    "What are the most common complaint categories and how many are still unresolved?",
    "Which marketing campaigns have the best conversion rate?",
    "Which customer segments bring the most revenue in the last 6 months?",
    "Which cities have the highest number of affected customers from network events?",
    "Show customer distribution by city.",
    "Which customers have the highest data usage this month?",
    "Which customers generated the highest roaming cost?",
    "Which customers have 5G capable devices but are not on a 5G plan?",
    "Which add-ons are most used by customers?",
    "Which payment channels are used the most?",
    "Summarize recent support interactions by channel, reason, sentiment, and priority.",
]

NAV_ITEMS = [
    ("Overview", "✨ Overview"),
    ("Chart Builder", "📊 Chart Builder"),
    ("Suggested Questions", "💡 Suggested Questions"),
    ("SQL Query Builder", "🧮 SQL Query Builder"),
]


def default_assistant_message():
    return {
        "role": "assistant",
        "content": "Hello. Ask me a business question about the Zain Customer 360 database.",
    }


def title_from_question(question):
    cleaned = " ".join(question.split())
    return cleaned[:34] + "..." if len(cleaned) > 34 else cleaned or "New Chat"


def render_page_brand(page_key, title, subtitle):
    logo_html = ""

    if LOGO_DATA_URI:
        logo_html = f'<img class="page-brand-logo" src="{LOGO_DATA_URI}" alt="Zain Logo">'

    st.markdown(
        f"""
        <div class="page-brand-card" id="page-brand-{page_key}">
          <div class="page-brand-content">
            <div class="page-brand-eyebrow">Zain 360 Copilot</div>
            <div class="page-brand-title">{title}</div>
            <div class="page-brand-copy">{subtitle}</div>
          </div>
          {logo_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_chat_sessions():
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = [
            {
                "id": "chat_1",
                "title": "New Chat",
                "messages": [default_assistant_message()],
            }
        ]
        st.session_state.current_chat_id = "chat_1"

    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = st.session_state.chat_sessions[0]["id"]

    if "rename_chat_id" not in st.session_state:
        st.session_state.rename_chat_id = None

    if "rename_chat_value" not in st.session_state:
        st.session_state.rename_chat_value = ""

    if "open_chat_menu_id" not in st.session_state:
        st.session_state.open_chat_menu_id = None


def current_chat():
    init_chat_sessions()

    for chat in st.session_state.chat_sessions:
        if chat["id"] == st.session_state.current_chat_id:
            return chat

    st.session_state.current_chat_id = st.session_state.chat_sessions[0]["id"]
    return st.session_state.chat_sessions[0]


def create_new_chat():
    init_chat_sessions()

    next_id = f"chat_{len(st.session_state.chat_sessions) + 1}_{int(time.time())}"

    chat = {
        "id": next_id,
        "title": "New Chat",
        "messages": [default_assistant_message()],
    }

    st.session_state.chat_sessions.insert(0, chat)
    st.session_state.current_chat_id = next_id
    st.session_state.page = "Chat"
    st.session_state.open_chat_menu_id = None
    st.session_state.rename_chat_id = None
    st.session_state.rename_chat_value = ""


def rename_chat(chat_id, new_title):
    new_title = " ".join(new_title.split()).strip()

    if not new_title:
        return

    for chat in st.session_state.chat_sessions:
        if chat["id"] == chat_id:
            chat["title"] = new_title
            break

    st.session_state.rename_chat_id = None
    st.session_state.rename_chat_value = ""
    st.session_state.open_chat_menu_id = None


def delete_chat(chat_id):
    st.session_state.chat_sessions = [
        chat for chat in st.session_state.chat_sessions if chat["id"] != chat_id
    ]

    if not st.session_state.chat_sessions:
        st.session_state.chat_sessions = [
            {
                "id": f"chat_1_{int(time.time())}",
                "title": "New Chat",
                "messages": [default_assistant_message()],
            }
        ]

    if st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat_id = st.session_state.chat_sessions[0]["id"]
        st.session_state.page = "Chat"

    if st.session_state.rename_chat_id == chat_id:
        st.session_state.rename_chat_id = None
        st.session_state.rename_chat_value = ""

    if st.session_state.open_chat_menu_id == chat_id:
        st.session_state.open_chat_menu_id = None


def render_chart(chart):
    rows = chart.get("rows") or []

    if not rows:
        st.warning(chart.get("summary") or "No matching data was found for this chart request.")
        return

    df = pd.DataFrame(rows)
    title = chart.get("title", "Chart")
    chart_type = chart.get("chart_type", "bar")

    st.subheader(title)

    if chart_type == "pie":
        st.plotly_chart(px.pie(df, names="label", values="value", height=430), width="stretch")
    elif chart_type == "doughnut":
        st.plotly_chart(px.pie(df, names="label", values="value", hole=0.52, height=430), width="stretch")
    elif chart_type == "line":
        st.plotly_chart(px.line(df, x="label", y="value", markers=True, height=430), width="stretch")
    elif chart_type == "area":
        st.plotly_chart(px.area(df, x="label", y="value", height=430), width="stretch")
    elif chart_type == "horizontal_bar":
        st.plotly_chart(px.bar(df, x="value", y="label", orientation="h", height=430), width="stretch")
    else:
        st.plotly_chart(px.bar(df, x="label", y="value", height=430), width="stretch")

    if chart.get("summary"):
        st.caption(chart["summary"])

    with st.expander("Chart data"):
        st.dataframe(df, width="stretch", hide_index=True)


def stream_markdown(text):
    placeholder = st.empty()
    rendered = ""

    for token in text.split(" "):
        rendered += token + " "
        placeholder.markdown(rendered)
        time.sleep(0.015)


def run_sql_callback(key_prefix):
    sql = st.session_state.get(f"{key_prefix}_sql_editor", "").strip()

    try:
        st.session_state[f"{key_prefix}_sql_result"] = execute_sql_query(sql)
        st.session_state[f"{key_prefix}_sql_error"] = ""
    except Exception as exc:
        st.session_state[f"{key_prefix}_sql_result"] = None
        st.session_state[f"{key_prefix}_sql_error"] = f"{type(exc).__name__}: {exc}"


def render_sql_viewer(sql):
    st.code(sql, language="sql")


def render_sql_runner(default_sql="", key_prefix="sql_runner"):
    st.markdown("#### SQL Query Builder")

    editor_key = f"{key_prefix}_sql_editor"

    if editor_key not in st.session_state:
        st.session_state[editor_key] = default_sql

    st.text_area("SQL", height=180, key=editor_key)

    st.button(
        "Run Query",
        type="primary",
        key=f"{key_prefix}_run_button",
        on_click=run_sql_callback,
        args=(key_prefix,),
    )

    error = st.session_state.get(f"{key_prefix}_sql_error", "")
    result = st.session_state.get(f"{key_prefix}_sql_result")

    if error:
        st.error(f"Query failed: {error}")
    elif result:
        rows = result.get("rows", [])
        st.success(f"Returned {len(rows)} rows.")

        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("Query ran successfully but returned no rows.")


def show_overview():
    render_page_brand(
        "overview",
        "Database Overview",
        "Monitor customer volume, churn distribution, city concentration, and key database activity from one clean executive view.",
    )

    data = get_database_overview()
    st.caption(data["summary"])

    cols = st.columns(len(data["kpis"]))

    for col, item in zip(cols, data["kpis"]):
        col.metric(item["label"], f"{item['value']:,}")

    chart_cols = st.columns(2)

    for index, chart in enumerate(data["charts"]):
        with chart_cols[index % 2]:
            render_chart(chart)

    st.subheader("Tables and Row Counts")
    st.dataframe(pd.DataFrame(data["tables"]), width="stretch", hide_index=True)


def show_chart_builder():
    render_page_brand(
        "chart_builder",
        "Chart Builder",
        "Create charts directly from natural-language business questions using the Customer 360 database.",
    )

    question = st.text_area(
        "Chart inquiry",
        value="Build a chart based on customer with ID = 9 by their complaints type and number.",
        height=110,
    )

    chart_label = st.selectbox("Chart type", list(CHART_TYPES.keys()))

    if st.button("Create Chart", type="primary"):
        with st.spinner("Building chart from database..."):
            chart = build_chart_from_question(question, CHART_TYPES[chart_label])

        st.session_state.last_chart = chart

    if st.session_state.get("last_chart"):
        render_chart(st.session_state.last_chart)


def show_chat():
    chat = current_chat()

    render_page_brand(
        "chat",
        "Customer 360 Chat",
        f"{chat['title']} · Ask about customers, churn, billing, complaints, campaigns, network events, and usage insights.",
    )

    for index, message in enumerate(chat["messages"]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sql"):
                with st.expander("SQL Query"):
                    render_sql_viewer(message["sql"])

    chat_bar = st.container(key="fixed_chat_bar")

    with chat_bar:
        with st.form("fixed_chat_form", clear_on_submit=True):
            input_col, send_col = st.columns([8, 1])

            with input_col:
                prompt = st.text_input(
                    "Question",
                    placeholder="Ask Zain 360 Copilot anything about the database...",
                    label_visibility="collapsed",
                )

            with send_col:
                submitted = st.form_submit_button("Send", type="primary", width="stretch")

    if submitted and prompt.strip():
        prompt = prompt.strip()

        if chat["title"] == "New Chat":
            chat["title"] = title_from_question(prompt)

        chat["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                payload = ask_sql_agent_payload(prompt)

            stream_markdown(payload["answer"])

            if payload.get("sql"):
                with st.expander("SQL Query"):
                    render_sql_viewer(payload["sql"])

        chat["messages"].append(
            {
                "role": "assistant",
                "content": payload["answer"],
                "sql": payload.get("sql", ""),
            }
        )


def show_suggested_questions():
    chat = current_chat()

    render_page_brand(
        "suggested_questions",
        "Suggested Questions",
        "Start with guided use cases for churn, complaints, revenue, campaigns, network impact, and customer profiling.",
    )

    for question in SUGGESTED_QUESTIONS:
        if st.button(question):
            if chat["title"] == "New Chat":
                chat["title"] = title_from_question(question)

            chat["messages"].append({"role": "user", "content": question})

            with st.spinner("Thinking..."):
                payload = ask_sql_agent_payload(question)

            chat["messages"].append(
                {
                    "role": "assistant",
                    "content": payload["answer"],
                    "sql": payload.get("sql", ""),
                }
            )

            st.success("Question sent to Chat. Open the Chat page to view the answer.")
            st.session_state.page = "Chat"
            time.sleep(0.8)
            st.rerun()


def show_sql_builder_page():
    render_page_brand(
        "sql_builder",
        "SQL Query Builder",
        "Run safe read-only SQL queries, inspect returned rows, and validate the data behind business insights.",
    )

    st.caption("Safe runner for read-only SELECT queries.")
    render_sql_runner("SELECT COUNT(*) AS total_customers FROM customers", key_prefix="standalone")


with st.sidebar:
    st.markdown(
        """
        <div class="horizon-logo-wrap">
          <div>
            <div class="horizon-logo-text">ZAIN<span class="horizon-logo-dot">.</span>AI</div>
            <div class="horizon-logo-sub">Customer 360</div>
          </div>
        </div>
        <div class="horizon-divider"></div>
        """,
        unsafe_allow_html=True,
    )

    if "page" not in st.session_state:
        st.session_state.page = "Chat"

    init_chat_sessions()

    top_col_1, top_col_2 = st.columns([4, 1.15])

    with top_col_1:
        if st.button("＋ New Chat", key="sidebar_new_chat", type="primary"):
            create_new_chat()
            st.rerun()

    with top_col_2:
        st.button("⌕", key="sidebar_search_fake")

    st.markdown('<div class="sidebar-section-label">Conversations</div>', unsafe_allow_html=True)

    with st.expander("Chats", expanded=True):
        for chat in list(st.session_state.chat_sessions):
            with st.container(key=f"chat_item_{chat['id']}"):
                row_col, menu_col = st.columns([8, 1])

                with row_col:
                    with st.container(key=f"chat_select_wrap_{chat['id']}"):
                        label = "✨ " + chat["title"]
                        chat_type = (
                            "primary"
                            if st.session_state.page == "Chat"
                            and st.session_state.current_chat_id == chat["id"]
                            else "secondary"
                        )

                        if st.button(label, key=f"select_{chat['id']}", type=chat_type):
                            st.session_state.current_chat_id = chat["id"]
                            st.session_state.page = "Chat"
                            st.session_state.open_chat_menu_id = None
                            st.rerun()

                with menu_col:
                    with st.container(key=f"chat_menu_wrap_{chat['id']}"):
                        if st.button("⋯", key=f"menu_toggle_{chat['id']}", help="Chat options"):
                            if st.session_state.open_chat_menu_id == chat["id"]:
                                st.session_state.open_chat_menu_id = None
                            else:
                                st.session_state.open_chat_menu_id = chat["id"]

                            st.rerun()

                if st.session_state.open_chat_menu_id == chat["id"]:
                    with st.container(key=f"chat_actions_wrap_{chat['id']}"):
                        action_col1, action_col2 = st.columns(2)

                        with action_col1:
                            if st.button("Rename", key=f"rename_button_{chat['id']}"):
                                st.session_state.rename_chat_id = chat["id"]
                                st.session_state.rename_chat_value = chat["title"]
                                st.session_state.open_chat_menu_id = None
                                st.rerun()

                        with action_col2:
                            if st.button("Delete", key=f"delete_button_{chat['id']}"):
                                delete_chat(chat["id"])
                                st.rerun()

                if st.session_state.rename_chat_id == chat["id"]:
                    with st.container(key=f"chat_rename_wrap_{chat['id']}"):
                        new_title = st.text_input(
                            "Rename chat",
                            value=st.session_state.rename_chat_value,
                            key=f"rename_input_{chat['id']}",
                            label_visibility="collapsed",
                            placeholder="Enter new chat name",
                        )

                        save_col, cancel_col = st.columns(2)

                        with save_col:
                            if st.button("Save", key=f"save_rename_{chat['id']}", type="primary"):
                                rename_chat(chat["id"], new_title)
                                st.rerun()

                        with cancel_col:
                            if st.button("Cancel", key=f"cancel_rename_{chat['id']}"):
                                st.session_state.rename_chat_id = None
                                st.session_state.rename_chat_value = ""
                                st.rerun()

    st.markdown('<div class="horizon-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Workspace</div>', unsafe_allow_html=True)

    for page_name, label in NAV_ITEMS:
        nav_button_type = "primary" if st.session_state.page == page_name else "secondary"

        if st.button(label, key=f"nav_{page_name}", type=nav_button_type):
            st.session_state.page = page_name
            st.session_state.open_chat_menu_id = None
            st.rerun()

    st.markdown(
        """
        <div class="sidebar-footnote">
          Built for telecom insight: customer view, churn, revenue, complaints, campaigns, and network impact.
        </div>
        """,
        unsafe_allow_html=True,
    )


page = st.session_state.page

if page == "Chat":
    show_chat()
elif page == "Overview":
    show_overview()
elif page == "Chart Builder":
    show_chart_builder()
elif page == "Suggested Questions":
    show_suggested_questions()
else:
    show_sql_builder_page()
