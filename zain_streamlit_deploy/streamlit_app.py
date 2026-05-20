def render_sql_viewer(sql):
    st.code(sql, language="sql")


def show_chat():
    chat = current_chat()

    render_page_brand(
        "chat",
        "Customer 360 Chat",
        f"{chat['title']} · Ask about customers, churn, billing, complaints, campaigns, and network events.",
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
            input_col, send_col = st.columns([8, 1.4])

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
                    render_sql_viewer(payload["sql"])

        chat["messages"].append(
            {
                "role": "assistant",
                "content": payload["answer"],
                "sql": payload.get("sql", ""),
            }
        )
