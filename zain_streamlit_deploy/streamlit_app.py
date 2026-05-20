import os
import time
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
    page_title="Zain Customer 360",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp {
        background: #0b0d12;
        color: #f4f4f5;
      }
      section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #11141b, #0b0d12);
        border-right: 1px solid #2b303b;
      }
      section[data-testid="stSidebar"] > div {
        padding: 1.2rem 1rem;
      }
      section[data-testid="stSidebar"] img {
        max-width: 190px;
        margin: 0 auto 0.65rem;
        display: block;
      }
      .sidebar-hero {
        border: 1px solid #303746;
        border-radius: 16px;
        padding: 0.95rem;
        margin: 0.35rem 0 1rem;
        background:
          radial-gradient(circle at 15% 0%, rgba(215, 25, 32, 0.24), transparent 34%),
          linear-gradient(180deg, rgba(32, 38, 53, 0.98), rgba(17, 20, 27, 0.98));
        box-shadow: 0 18px 42px rgba(0, 0, 0, 0.24);
      }
      .sidebar-hero-title {
        color: #ffffff;
        font-size: 1rem;
        font-weight: 900;
        line-height: 1.25;
        margin-bottom: 0.35rem;
      }
      .sidebar-hero-copy {
        color: #b6bcc8;
        font-size: 0.78rem;
        line-height: 1.45;
      }
      .sidebar-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.38rem;
        margin-top: 0.75rem;
      }
      .sidebar-chip {
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 999px;
        padding: 0.26rem 0.48rem;
        color: #f4f4f5;
        background: rgba(255, 255, 255, 0.06);
        font-size: 0.67rem;
        font-weight: 800;
      }
      .sidebar-title {
        color: #8f96a3;
        font-size: 0.74rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0.6rem 0 0.5rem;
      }
      .nav-active {
        border: 1px solid rgba(215, 25, 32, 0.62);
        border-radius: 14px;
        padding: 0.72rem 0.85rem;
        margin-bottom: 0.55rem;
        color: #ffffff;
        background: linear-gradient(135deg, rgba(215, 25, 32, 0.28), rgba(32, 38, 53, 0.96));
        box-shadow: 0 0 0 3px rgba(215, 25, 32, 0.15);
        font-weight: 900;
      }
      .block-container {
        padding-top: 1.4rem;
      }
      div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #202635, #171b25);
        border: 1px solid #303746;
        border-radius: 10px;
        padding: 14px;
      }
      div.stButton > button {
        width: 100%;
        min-height: 44px;
        border-radius: 12px;
        border: 1px solid #303746;
        background: linear-gradient(180deg, #202635, #171b25);
        color: #f4f4f5;
        font-weight: 800;
        text-align: left;
        justify-content: flex-start;
        padding-left: 0.9rem;
      }
      div.stButton > button:hover {
        border-color: rgba(215, 25, 32, 0.65);
        color: #ffffff;
        background: linear-gradient(180deg, #2d3548, #1d2330);
        transform: translateY(-1px);
      }
      .sidebar-footnote {
        margin-top: 1rem;
        border-top: 1px solid #2b303b;
        padding-top: 0.8rem;
        color: #8f96a3;
        font-size: 0.72rem;
        line-height: 1.45;
      }
      .st-key-fixed_chat_bar {
        position: fixed;
        left: 22rem;
        right: 1.5rem;
        bottom: 0.9rem;
        z-index: 1001;
        border: 1px solid #303746;
        border-radius: 16px;
        padding: 0.75rem;
        background: rgba(17, 20, 27, 0.98);
        box-shadow: 0 -18px 42px rgba(0, 0, 0, 0.28);
      }
      .st-key-fixed_chat_bar [data-testid="stForm"] {
        border: 0;
        padding: 0;
        background: transparent;
      }
      .st-key-fixed_chat_bar div[data-testid="stHorizontalBlock"] {
        align-items: end;
      }
      .st-key-fixed_chat_bar input {
        min-height: 48px;
        border-radius: 12px;
      }
      .st-key-fixed_chat_bar button {
        min-height: 48px !important;
        justify-content: center !important;
        border-radius: 12px !important;
      }
      @media (max-width: 900px) {
        .st-key-fixed_chat_bar {
          left: 1rem;
          right: 1rem;
        }
      }
      div[data-testid="stAlert"] {
        border-radius: 10px;
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
    "Find the top 10 customers with the highest churn score and explain why they are at risk.",
    "Which customer segments bring the most revenue in the last 6 months?",
    "What are the most common complaint categories and which ones are still unresolved?",
    "Which cities have the highest number of affected customers from network events?",
    "Which marketing campaigns have the best conversion rate?",
    "Show me the full profile, plan, complaints, churn risk, and recommended action for customer 42.",
    "Which customers have overdue invoices and high churn risk?",
    "Summarize recent support interactions by channel, reason, sentiment, and priority.",
]

NAV_ITEMS = [
    ("Overview", "📊 Overview"),
    ("Chart Builder", "📈 Chart Builder"),
    ("Suggested Questions", "✨ Suggested Questions"),
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
    st.title("Database Overview")
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
    st.title("Chart Builder")
    st.caption("Ask for one chart in business language. The app queries the database and builds only that chart.")
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
    st.title("Customer 360 Chat")
    st.caption(f"{chat['title']} · Ask about customers, churn, billing, complaints, campaigns, and network events.")

    for index, message in enumerate(chat["messages"]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sql"):
                with st.expander("SQL Query"):
                    render_sql_runner(message["sql"], key_prefix=f"{chat['id']}_history_{index}")

    chat_bar = st.container(key="fixed_chat_bar")
    with chat_bar:
        with st.form("fixed_chat_form", clear_on_submit=True):
            input_col, send_col = st.columns([9, 1])
            with input_col:
                prompt = st.text_input(
                    "Question",
                    placeholder="Ask a question about the Customer 360 database",
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
            with st.spinner("Asking the SQL agent..."):
                payload = ask_sql_agent_payload(prompt)
            stream_markdown(payload["answer"])
            if payload.get("sql"):
                with st.expander("SQL Query"):
                    render_sql_runner(payload["sql"], key_prefix=f"{chat['id']}_current_{len(chat['messages'])}")
        chat["messages"].append(
            {"role": "assistant", "content": payload["answer"], "sql": payload.get("sql", "")}
        )


def show_suggested_questions():
    chat = current_chat()
    st.title("Suggested Questions")
    st.caption("Use these predictable customer inputs to guide useful database-backed questions.")
    for question in SUGGESTED_QUESTIONS:
        if st.button(question):
            if chat["title"] == "New Chat":
                chat["title"] = title_from_question(question)
            chat["messages"].append({"role": "user", "content": question})
            with st.spinner("Asking the SQL agent..."):
                payload = ask_sql_agent_payload(question)
            chat["messages"].append(
                {"role": "assistant", "content": payload["answer"], "sql": payload.get("sql", "")}
            )
            st.success("Question sent to Chat. Open the Chat page to view the answer.")
            st.session_state.page = "Chat"
            time.sleep(0.8)
            st.rerun()


with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width="stretch")
    else:
        st.warning("Logo image was not found in the app folder.")

    st.markdown(
        """
        <div class="sidebar-hero">
          <div class="sidebar-hero-title">Customer 360 AI Copilot</div>
          <div class="sidebar-hero-copy">
            Explore subscribers, churn, complaints, campaigns, billing, and network signals from one conversational workspace.
          </div>
          <div class="sidebar-chips">
            <span class="sidebar-chip">SQL-backed</span>
            <span class="sidebar-chip">Charts</span>
            <span class="sidebar-chip">Overview</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "page" not in st.session_state:
        st.session_state.page = "Chat"
    init_chat_sessions()

    active_label = current_chat()["title"] if st.session_state.page == "Chat" else dict(NAV_ITEMS)[st.session_state.page]
    st.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="nav-active">{active_label}</div>', unsafe_allow_html=True)

    if st.button("＋ New Chat", key="sidebar_new_chat", type="primary"):
        create_new_chat()
        st.rerun()

    with st.expander("Chats", expanded=True):
        for chat in st.session_state.chat_sessions:
            label = "💬 " + chat["title"]
            if st.button(label, key=f"select_{chat['id']}", type="secondary"):
                st.session_state.current_chat_id = chat["id"]
                st.session_state.page = "Chat"
                st.rerun()

    for page_name, label in NAV_ITEMS:
        if st.button(label, key=f"nav_{page_name}", type="secondary"):
            st.session_state.page = page_name
            st.rerun()

    st.markdown(
        """
        <div class="sidebar-footnote">
          Built for fast telecom insight: ask, inspect the SQL, and turn answers into action.
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
    st.title("SQL Query Builder")
    st.caption("Safe runner for read-only SELECT queries.")
    render_sql_runner("SELECT COUNT(*) AS total_customers FROM customers", key_prefix="standalone")
