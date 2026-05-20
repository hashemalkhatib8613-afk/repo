import json
import os
import re
import sqlite3
from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "zain_customer_360_ai_demo.db"

MODEL_NAME = "gpt-4.1-mini"
TOP_K = 10

NON_DATABASE_PATTERNS = (
    r"^\s*(hi|hello|hey|مرحبا|اهلا|أهلا|السلام عليكم)\s*[!.؟?]*\s*$",
    r"^\s*(thanks|thank you|شكرا|شكرًا)\s*[!.؟?]*\s*$",
    r"^\s*(how are you|كيفك|كيف الحال)\s*[!.؟?]*\s*$",
)


def load_openai_key():
    if os.environ.get("OPENAI_API_KEY"):
        return

    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)

        if key.strip() == "OPENAI_API_KEY" and value.strip():
            os.environ["OPENAI_API_KEY"] = value.strip().strip('"').strip("'")
            return


def extract_final_text(result):
    last_message = result["messages"][-1]

    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        return last_message.content

    if hasattr(last_message, "content_blocks"):
        parts = []

        for block in last_message.content_blocks:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(block["text"])
                elif "content" in block:
                    parts.append(str(block["content"]))
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))

        return "\n".join(parts)

    return str(last_message)


def _connect():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_markdown(rows):
    if not rows:
        return "No matching rows found."

    headers = rows[0].keys()

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for row in rows:
        values = []
        for key in headers:
            value = row[key]
            if value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _dict_rows(rows):
    return [dict(row) for row in rows]


def _run_rows(query, params=()):
    with _connect() as conn:
        return conn.execute(query, params).fetchall()


def _run_one(query, params=()):
    rows = _run_rows(query, params)
    return rows[0] if rows else None


def _table_exists(table_name):
    row = _run_one(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table_name,),
    )
    return row is not None


def _schema_summary():
    with _connect() as conn:
        table_names = [
            row["name"]
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                ORDER BY name
                """
            )
        ]

        parts = []

        for table in table_names:
            columns = [
                f"{row['name']} {row['type']}"
                for row in conn.execute(f"PRAGMA table_info({table})")
            ]
            parts.append(f"{table}: " + ", ".join(columns))

        return "\n".join(parts)


def _extract_json_object(text):
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("The model did not return valid JSON.")

    return json.loads(text[start : end + 1])


def _normalize_sql(sql):
    return sql.strip().rstrip(";")


def validate_read_only_sql(sql):
    normalized = _normalize_sql(sql)
    lowered = normalized.lower()

    banned = (
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "create",
        "replace",
        "attach",
        "detach",
        "pragma",
        "vacuum",
        "reindex",
    )

    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("Only SELECT queries are allowed.")

    if any(re.search(rf"\b{word}\b", lowered) for word in banned):
        raise ValueError("Only read-only SELECT queries are allowed.")

    return normalized


def _validate_chart_sql(sql):
    normalized = validate_read_only_sql(sql)

    lowered = normalized.lower()

    if " label" not in lowered and " as label" not in lowered:
        raise ValueError("Chart SQL must return a label column.")

    if " value" not in lowered and " as value" not in lowered:
        raise ValueError("Chart SQL must return a value column.")

    return normalized


def _has_limit(sql):
    lowered = f" {_normalize_sql(sql).lower()} "
    return " limit " in lowered


def _is_grouped_query(sql):
    lowered = f" {_normalize_sql(sql).lower()} "
    return (
        " group by " in lowered
        or " count(" in lowered
        or " sum(" in lowered
        or " avg(" in lowered
        or " min(" in lowered
        or " max(" in lowered
    )


def execute_sql_query(sql, limit=50):
    sql = validate_read_only_sql(sql)

    limited_sql = sql

    if not _has_limit(sql) and not _is_grouped_query(sql):
        safe_limit = max(1, min(int(limit), 100))
        limited_sql = f"{sql}\nLIMIT {safe_limit}"

    rows = _dict_rows(_run_rows(limited_sql))
    columns = list(rows[0].keys()) if rows else []

    return {
        "columns": columns,
        "rows": rows,
        "sql": limited_sql,
    }


def get_database_overview():
    with _connect() as conn:
        table_rows = [
            {
                "label": row["name"],
                "value": conn.execute(f"SELECT COUNT(*) AS count FROM {row['name']}").fetchone()["count"],
            }
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                ORDER BY name
                """
            )
        ]

    kpis = [
        {
            "label": "Customers",
            "value": _run_rows("SELECT COUNT(*) AS value FROM customers")[0]["value"],
        },
        {
            "label": "Complaints",
            "value": _run_rows("SELECT COUNT(*) AS value FROM complaints")[0]["value"],
        },
        {
            "label": "Campaign Responses",
            "value": _run_rows("SELECT COUNT(*) AS value FROM customer_campaign_responses")[0]["value"],
        },
        {
            "label": "Network Events",
            "value": _run_rows("SELECT COUNT(*) AS value FROM network_events")[0]["value"],
        },
    ]

    charts = [
        {
            "title": "Customers by Segment",
            "metric": "Customers",
            "chart_type": "pie",
            "rows": _dict_rows(
                _run_rows(
                    """
                    SELECT customer_segment AS label, COUNT(*) AS value
                    FROM customers
                    GROUP BY customer_segment
                    ORDER BY value DESC
                    """
                )
            ),
        },
        {
            "title": "Customers by City",
            "metric": "Customers",
            "chart_type": "bar",
            "rows": _dict_rows(
                _run_rows(
                    """
                    SELECT city AS label, COUNT(*) AS value
                    FROM customers
                    GROUP BY city
                    ORDER BY value DESC
                    LIMIT 8
                    """
                )
            ),
        },
        {
            "title": "Churn Risk Levels",
            "metric": "Customers",
            "chart_type": "pie",
            "rows": _dict_rows(
                _run_rows(
                    """
                    SELECT risk_level AS label, COUNT(*) AS value
                    FROM customer_churn_scores
                    GROUP BY risk_level
                    ORDER BY value DESC
                    """
                )
            ),
        },
        {
            "title": "Largest Tables",
            "metric": "Rows",
            "chart_type": "bar",
            "rows": sorted(table_rows, key=lambda row: row["value"], reverse=True)[:10],
        },
    ]

    return {
        "kpis": kpis,
        "charts": charts,
        "tables": sorted(table_rows, key=lambda row: row["label"]),
        "summary": "This overview highlights database size, customer distribution, churn risk, and major activity tables.",
    }


def _extract_limit(question, default=10, maximum=25):
    q = question.lower()
    limit_match = re.search(r"\btop\s+(\d+)|\blimit\s+(\d+)", q)

    if not limit_match:
        return default

    value = int(next(value for value in limit_match.groups() if value))
    return max(1, min(value, maximum))


def _extract_customer_id(question):
    q = question.lower()

    customer_match = re.search(
        r"\bcustomer(?:\s+with)?\s+id\s*=?\s*(\d+)"
        r"|\bcustomer_id\s*=?\s*(\d+)"
        r"|\bcustomer\s*#\s*(\d+)"
        r"|\bid\s*=\s*(\d+)",
        q,
    )

    if not customer_match:
        return None

    return int(next(value for value in customer_match.groups() if value))


def _latest_customer_monthly_summary(customer_id):
    return _run_one(
        """
        SELECT
            summary_month,
            total_revenue_jod,
            voice_minutes,
            data_used_gb,
            sms_count,
            support_interactions_count,
            complaints_count,
            payment_delay_days,
            churn_score
        FROM customer_monthly_summary
        WHERE customer_id = ?
        ORDER BY summary_month DESC
        LIMIT 1
        """,
        (customer_id,),
    )


def _latest_customer_satisfaction(customer_id):
    if not _table_exists("customer_satisfaction"):
        return None

    return _run_one(
        """
        SELECT
            survey_date,
            nps_score,
            csat_score,
            sentiment
        FROM customer_satisfaction
        WHERE customer_id = ?
        ORDER BY survey_date DESC
        LIMIT 1
        """,
        (customer_id,),
    )


def _customer_churn(customer_id):
    return _run_one(
        """
        SELECT
            churn_score,
            risk_level,
            main_risk_reason,
            recommended_action
        FROM customer_churn_scores
        WHERE customer_id = ?
        """,
        (customer_id,),
    )


def _customer_accounts(customer_id):
    return _run_rows(
        """
        SELECT
            account_id,
            account_number,
            account_type,
            billing_cycle_day,
            credit_limit_jod,
            account_status,
            created_date
        FROM accounts
        WHERE customer_id = ?
        ORDER BY created_date DESC, account_id DESC
        """,
        (customer_id,),
    )


def _customer_subscriptions(customer_id):
    return _run_rows(
        """
        SELECT
            s.subscription_id,
            s.msisdn,
            s.service_type,
            s.activation_date,
            s.contract_end_date,
            s.status AS subscription_status,
            p.plan_name,
            p.plan_category,
            p.monthly_fee_jod,
            p.data_allowance_gb,
            p.local_minutes,
            p.international_minutes,
            p.roaming_minutes,
            p.sms_allowance,
            p.technology AS plan_technology,
            p.contract_months,
            p.is_business_plan
        FROM subscriptions s
        LEFT JOIN plans p ON s.plan_id = p.plan_id
        WHERE s.customer_id = ?
        ORDER BY s.activation_date DESC, s.subscription_id DESC
        """,
        (customer_id,),
    )


def _customer_recent_complaints(customer_id, limit=5):
    return _run_rows(
        """
        SELECT
            complaint_id,
            complaint_date,
            complaint_category,
            complaint_type,
            status,
            priority,
            channel,
            resolution_date
        FROM complaints
        WHERE customer_id = ?
        ORDER BY complaint_date DESC
        LIMIT ?
        """,
        (customer_id, limit),
    )


def _customer_recent_invoices(customer_id, limit=5):
    return _run_rows(
        """
        SELECT
            i.invoice_id,
            i.invoice_date,
            i.due_date,
            i.total_amount_jod,
            i.payment_status,
            i.days_overdue
        FROM invoices i
        JOIN accounts a ON i.account_id = a.account_id
        WHERE a.customer_id = ?
        ORDER BY i.invoice_date DESC
        LIMIT ?
        """,
        (customer_id, limit),
    )


def _customer_profile_answer(customer_id):
    customer = _run_one(
        """
        SELECT
            customer_id,
            full_name,
            gender,
            date_of_birth,
            age_group,
            city,
            governorate,
            customer_type,
            customer_segment,
            preferred_language,
            email,
            phone_number,
            signup_date,
            status
        FROM customers
        WHERE customer_id = ?
        """,
        (customer_id,),
    )

    if not customer:
        return {
            "answer": f"Direct Answer\nNo customer was found with customer_id = {customer_id}.",
            "sql": f"SELECT * FROM customers WHERE customer_id = {customer_id}",
        }

    accounts = _customer_accounts(customer_id)
    subscriptions = _customer_subscriptions(customer_id)
    monthly = _latest_customer_monthly_summary(customer_id)
    satisfaction = _latest_customer_satisfaction(customer_id)
    churn = _customer_churn(customer_id)
    complaints = _customer_recent_complaints(customer_id)
    invoices = _customer_recent_invoices(customer_id)

    answer_parts = []

    answer_parts.append(
        f"""
Direct Answer

Customer {customer['full_name']} has customer_id = {customer_id}. Below is a clean customer profile built from separate focused queries to avoid duplicated rows from one-to-many joins.
""".strip()
    )

    answer_parts.append("\n\nCustomer Details\n\n" + _rows_to_markdown([customer]))

    if accounts:
        answer_parts.append("\n\nAccounts\n\n" + _rows_to_markdown(accounts))
    else:
        answer_parts.append("\n\nAccounts\n\nNo accounts found.")

    if subscriptions:
        answer_parts.append("\n\nSubscriptions and Plans\n\n" + _rows_to_markdown(subscriptions))
    else:
        answer_parts.append("\n\nSubscriptions and Plans\n\nNo subscriptions found.")

    if monthly:
        answer_parts.append("\n\nLatest Monthly Summary\n\n" + _rows_to_markdown([monthly]))
    else:
        answer_parts.append("\n\nLatest Monthly Summary\n\nNo monthly summary found.")

    if satisfaction:
        answer_parts.append("\n\nLatest Satisfaction Survey\n\n" + _rows_to_markdown([satisfaction]))
    else:
        answer_parts.append("\n\nLatest Satisfaction Survey\n\nNo satisfaction survey found.")

    if churn:
        answer_parts.append("\n\nChurn Risk\n\n" + _rows_to_markdown([churn]))
    else:
        answer_parts.append("\n\nChurn Risk\n\nNo churn score found.")

    if complaints:
        answer_parts.append("\n\nRecent Complaints\n\n" + _rows_to_markdown(complaints))
    else:
        answer_parts.append("\n\nRecent Complaints\n\nNo recent complaints found.")

    if invoices:
        answer_parts.append("\n\nRecent Invoices\n\n" + _rows_to_markdown(invoices))
    else:
        answer_parts.append("\n\nRecent Invoices\n\nNo recent invoices found.")

    interpretation = "\n\nBusiness Interpretation\n"

    if churn:
        interpretation += (
            f"- Risk level: {churn['risk_level']}\n"
            f"- Churn score: {churn['churn_score']}\n"
            f"- Main risk reason: {churn['main_risk_reason']}\n"
        )
    else:
        interpretation += "- No churn risk record is available for this customer.\n"

    if monthly:
        interpretation += (
            f"- Latest monthly revenue: {monthly['total_revenue_jod']} JOD\n"
            f"- Latest data usage: {monthly['data_used_gb']} GB\n"
            f"- Payment delay days: {monthly['payment_delay_days']}\n"
        )

    recommended_action = "\nRecommended Next Action\n"

    if churn and churn["recommended_action"]:
        recommended_action += f"- {churn['recommended_action']}"
    elif complaints:
        recommended_action += "- Review recent complaints and contact the customer with a service recovery action."
    else:
        recommended_action += "- Monitor the customer normally and keep the account experience stable."

    answer_parts.append(interpretation)
    answer_parts.append(recommended_action)

    display_sql = f"""
-- Customer profile is intentionally built using separate queries to avoid duplicate rows.

SELECT * FROM customers WHERE customer_id = {customer_id};

SELECT * FROM accounts WHERE customer_id = {customer_id};

SELECT s.*, p.*
FROM subscriptions s
LEFT JOIN plans p ON s.plan_id = p.plan_id
WHERE s.customer_id = {customer_id};

SELECT *
FROM customer_monthly_summary
WHERE customer_id = {customer_id}
ORDER BY summary_month DESC
LIMIT 1;

SELECT *
FROM customer_churn_scores
WHERE customer_id = {customer_id};
""".strip()

    return {
        "answer": "\n".join(answer_parts),
        "sql": display_sql,
    }


def _customer_profile_flat_sql(customer_id):
    return f"""
WITH
latest_monthly AS (
    SELECT *
    FROM customer_monthly_summary
    WHERE customer_id = {customer_id}
    ORDER BY summary_month DESC
    LIMIT 1
),
latest_satisfaction AS (
    SELECT *
    FROM customer_satisfaction
    WHERE customer_id = {customer_id}
    ORDER BY survey_date DESC
    LIMIT 1
),
account_summary AS (
    SELECT
        customer_id,
        COUNT(*) AS account_count,
        GROUP_CONCAT(account_number, ', ') AS account_numbers,
        GROUP_CONCAT(account_type, ', ') AS account_types
    FROM accounts
    WHERE customer_id = {customer_id}
    GROUP BY customer_id
),
subscription_summary AS (
    SELECT
        s.customer_id,
        COUNT(*) AS subscription_count,
        GROUP_CONCAT(s.msisdn, ', ') AS msisdns,
        GROUP_CONCAT(p.plan_name, ', ') AS plan_names,
        ROUND(SUM(COALESCE(p.monthly_fee_jod, 0)), 2) AS total_monthly_fees_jod
    FROM subscriptions s
    LEFT JOIN plans p ON s.plan_id = p.plan_id
    WHERE s.customer_id = {customer_id}
    GROUP BY s.customer_id
)
SELECT
    c.customer_id,
    c.full_name,
    c.gender,
    c.date_of_birth,
    c.age_group,
    c.city,
    c.governorate,
    c.customer_type,
    c.customer_segment,
    c.preferred_language,
    c.email,
    c.phone_number,
    c.signup_date,
    c.status,
    acc.account_count,
    acc.account_numbers,
    acc.account_types,
    sub.subscription_count,
    sub.msisdns,
    sub.plan_names,
    sub.total_monthly_fees_jod,
    lm.summary_month,
    lm.total_revenue_jod,
    lm.voice_minutes,
    lm.data_used_gb,
    lm.sms_count,
    lm.support_interactions_count,
    lm.complaints_count,
    lm.payment_delay_days,
    lm.churn_score AS monthly_churn_score,
    sat.survey_date,
    sat.nps_score,
    sat.csat_score,
    sat.sentiment,
    ch.churn_score AS churn_score_value,
    ch.risk_level,
    ch.main_risk_reason,
    ch.recommended_action
FROM customers c
LEFT JOIN account_summary acc ON c.customer_id = acc.customer_id
LEFT JOIN subscription_summary sub ON c.customer_id = sub.customer_id
LEFT JOIN latest_monthly lm ON c.customer_id = lm.customer_id
LEFT JOIN latest_satisfaction sat ON c.customer_id = sat.customer_id
LEFT JOIN customer_churn_scores ch ON c.customer_id = ch.customer_id
WHERE c.customer_id = {customer_id}
LIMIT 1
""".strip()


def _run_dynamic_chart_query(question, chart_type, limit):
    load_openai_key()

    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "title": "Chart unavailable",
            "metric": "No data",
            "chart_type": chart_type,
            "rows": [],
            "summary": "OPENAI_API_KEY is not set, so the dynamic chart query could not be generated.",
        }

    model = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)

    prompt = f"""
You are a chart SQL planner for a SQLite telecom Customer 360 database.

Database schema:
{_schema_summary()}

User chart request:
{question}

Return only one JSON object with these fields:
- title: short chart title
- metric: short label for the numeric metric
- chart_type: "bar", "horizontal_bar", "pie", "doughnut", "line", or "area"
- sql: one safe SQLite SELECT query
- summary: one short business explanation

Rules:
- SQL must be read-only SELECT only.
- SQL must return exactly two required aliases: label and value.
- label must be human-readable text.
- value must be numeric.
- Use joins when needed.
- Avoid joining several one-to-many tables directly.
- If joining one-to-many tables, aggregate them first using GROUP BY, CTEs, COUNT, SUM, AVG, or GROUP_CONCAT.
- If the request asks about a specific customer, filter by that customer_id.
- Limit the chart to at most {limit} rows unless grouping naturally returns fewer.
- If the requested data is not available in the schema, set sql to an empty string and explain that in summary.
"""

    result = model.invoke([{"role": "user", "content": prompt}])
    content = result.content if hasattr(result, "content") else str(result)

    spec = _extract_json_object(content)

    sql = str(spec.get("sql", "")).strip()

    if not sql:
        return {
            "title": spec.get("title") or "Chart unavailable",
            "metric": spec.get("metric") or "No data",
            "chart_type": chart_type,
            "rows": [],
            "summary": spec.get("summary") or "The requested chart cannot be created from the available database fields.",
        }

    sql = _validate_chart_sql(sql)
    rows = _dict_rows(_run_rows(sql))

    normalized_rows = []

    for row in rows[:limit]:
        if "label" not in row or "value" not in row:
            continue

        try:
            value = float(row["value"])
        except (TypeError, ValueError):
            continue

        normalized = dict(row)
        normalized["label"] = str(row["label"])
        normalized["value"] = value
        normalized_rows.append(normalized)

    return {
        "title": spec.get("title") or "Custom Database Chart",
        "metric": spec.get("metric") or "Value",
        "chart_type": chart_type
        if chart_type in {"bar", "horizontal_bar", "pie", "doughnut", "line", "area"}
        else spec.get("chart_type", "bar"),
        "rows": normalized_rows,
        "summary": spec.get("summary") or "Chart created from the database.",
    }


def answer_direct_query(question):
    q = question.lower()

    if any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in NON_DATABASE_PATTERNS):
        return {
            "answer": "Hello. Ask me a question about customers, churn, complaints, billing, campaigns, network events, or anything in the Customer 360 database.",
            "sql": "",
        }

    limit = _extract_limit(question, default=10, maximum=25)
    customer_id = _extract_customer_id(question)

    wants_profile = (
        customer_id is not None
        and (
            "profile" in q
            or "full" in q
            or "details" in q
            or "plan" in q
            or "summary" in q
            or "customer" in q
            or "show me" in q
        )
    )

    if wants_profile:
        return _customer_profile_answer(customer_id)

    if "how many" in q and "customer" in q:
        sql = "SELECT COUNT(*) AS total_customers FROM customers"
        total = _run_rows(sql)[0]["total_customers"]

        answer = (
            f"Direct Answer\nThere are {total:,} customers in the database.\n\n"
            f"Key Numbers\n- Total customers: {total:,}\n\n"
            "Business Interpretation\nThis is the current Customer 360 base available for segmentation, churn, revenue, billing, and network analysis."
        )

        return {
            "answer": answer,
            "sql": sql,
        }

    if "churn" in q and ("highest" in q or "top" in q or "risk" in q):
        sql = """
SELECT
    c.customer_id,
    c.full_name,
    c.city,
    c.customer_segment,
    ch.churn_score,
    ch.risk_level,
    ch.main_risk_reason,
    ch.recommended_action
FROM customer_churn_scores ch
JOIN customers c ON ch.customer_id = c.customer_id
ORDER BY ch.churn_score DESC
LIMIT ?
""".strip()

        rows = _run_rows(sql, (limit,))

        answer = (
            f"Direct Answer\nThese are the top {limit} customers with the highest churn scores.\n\n"
            f"{_rows_to_markdown(rows)}\n\n"
            "Business Interpretation\nCustomers at the top of this list should be prioritized because their churn scores and risk reasons indicate a higher likelihood of leaving.\n\n"
            "Recommended Next Action\nContact them with retention offers matched to the recommended action and monitor complaint or service-quality signals."
        )

        return {
            "answer": answer,
            "sql": sql.replace("?", str(limit)),
        }

    if "complaint" in q:
        if customer_id is not None:
            sql = """
SELECT
    complaint_category,
    COUNT(*) AS total_complaints
FROM complaints
WHERE customer_id = ?
GROUP BY complaint_category
ORDER BY total_complaints DESC
""".strip()

            rows = _run_rows(sql, (customer_id,))
            customer = _run_rows("SELECT full_name FROM customers WHERE customer_id = ?", (customer_id,))

            if not customer:
                return {
                    "answer": f"Direct Answer\nNo customer was found with customer_id = {customer_id}.",
                    "sql": f"SELECT full_name FROM customers WHERE customer_id = {customer_id}",
                }

            customer_name = customer[0]["full_name"]

            if not rows:
                return {
                    "answer": f"Direct Answer\nNo complaint records were found for {customer_name} with customer_id = {customer_id}.",
                    "sql": sql.replace("?", str(customer_id)),
                }

            return {
                "answer": f"Direct Answer\nComplaint types for {customer_name} with customer_id = {customer_id}:\n\n" + _rows_to_markdown(rows),
                "sql": sql.replace("?", str(customer_id)),
            }

        sql = """
SELECT
    complaint_category,
    COUNT(*) AS total_complaints
FROM complaints
GROUP BY complaint_category
ORDER BY total_complaints DESC
LIMIT ?
""".strip()

        rows = _run_rows(sql, (limit,))

        return {
            "answer": "Direct Answer\nTop complaint categories:\n\n" + _rows_to_markdown(rows),
            "sql": sql.replace("?", str(limit)),
        }

    if "campaign" in q and ("conversion" in q or "best" in q):
        sql = """
SELECT
    ca.campaign_name,
    ca.campaign_type,
    ca.target_segment,
    COUNT(cr.response_id) AS total_sent,
    SUM(cr.converted_flag) AS total_converted,
    ROUND(100.0 * SUM(cr.converted_flag) / COUNT(cr.response_id), 2) AS conversion_rate_percent
FROM campaigns ca
JOIN customer_campaign_responses cr ON ca.campaign_id = cr.campaign_id
GROUP BY ca.campaign_id, ca.campaign_name, ca.campaign_type, ca.target_segment
ORDER BY conversion_rate_percent DESC
LIMIT ?
""".strip()

        rows = _run_rows(sql, (limit,))

        return {
            "answer": "Direct Answer\nBest campaigns by conversion rate:\n\n" + _rows_to_markdown(rows),
            "sql": sql.replace("?", str(limit)),
        }

    if "city" in q and "customer" in q:
        sql = """
SELECT
    city,
    COUNT(*) AS total_customers
FROM customers
GROUP BY city
ORDER BY total_customers DESC
LIMIT ?
""".strip()

        rows = _run_rows(sql, (limit,))

        return {
            "answer": "Direct Answer\nCustomer distribution by city:\n\n" + _rows_to_markdown(rows),
            "sql": sql.replace("?", str(limit)),
        }

    if "segment" in q and "revenue" in q:
        sql = """
SELECT
    value_segment,
    COUNT(*) AS total_customers,
    ROUND(AVG(arpu_jod), 2) AS avg_arpu,
    ROUND(AVG(total_revenue_6m_jod), 2) AS avg_revenue_6m
FROM customer_value_segments
GROUP BY value_segment
ORDER BY avg_revenue_6m DESC
LIMIT ?
""".strip()

        rows = _run_rows(sql, (limit,))

        return {
            "answer": "Direct Answer\nRevenue by value segment:\n\n" + _rows_to_markdown(rows),
            "sql": sql.replace("?", str(limit)),
        }

    if "network" in q or "affected" in q:
        sql = """
SELECT
    nt.city,
    SUM(ne.affected_customers) AS total_affected_customers
FROM network_events ne
JOIN network_towers nt ON ne.tower_id = nt.tower_id
GROUP BY nt.city
ORDER BY total_affected_customers DESC
LIMIT ?
""".strip()

        rows = _run_rows(sql, (limit,))

        return {
            "answer": "Direct Answer\nCities with highest network impact:\n\n" + _rows_to_markdown(rows),
            "sql": sql.replace("?", str(limit)),
        }

    return None


def build_chart_from_question(question, chart_type="bar"):
    q = question.lower()
    limit = _extract_limit(question, default=10, maximum=15)
    customer_id = _extract_customer_id(question)

    chart = chart_type if chart_type in {"bar", "horizontal_bar", "pie", "doughnut", "line", "area"} else "bar"

    if "churn" in q:
        rows = _run_rows(
            """
            SELECT
                c.full_name AS label,
                ROUND(ch.churn_score, 2) AS value,
                c.customer_id,
                c.city,
                c.customer_segment,
                ch.risk_level,
                ch.main_risk_reason
            FROM customer_churn_scores ch
            JOIN customers c ON ch.customer_id = c.customer_id
            ORDER BY ch.churn_score DESC
            LIMIT ?
            """,
            (limit,),
        )

        return {
            "title": f"Top {limit} Customers by Churn Score",
            "metric": "Churn score",
            "chart_type": chart,
            "rows": _dict_rows(rows),
            "summary": "The chart ranks customers by churn score. Higher scores indicate customers needing faster retention attention.",
        }

    if "complaint" in q:
        if customer_id is not None:
            rows = _run_rows(
                """
                SELECT
                    complaint_category AS label,
                    COUNT(*) AS value
                FROM complaints
                WHERE customer_id = ?
                GROUP BY complaint_category
                ORDER BY value DESC
                """,
                (customer_id,),
            )

            customer = _run_rows("SELECT full_name FROM customers WHERE customer_id = ?", (customer_id,))
            customer_name = customer[0]["full_name"] if customer else f"Customer {customer_id}"

            return {
                "title": f"Complaint Types for {customer_name}",
                "metric": "Complaints",
                "chart_type": chart,
                "rows": _dict_rows(rows),
                "summary": (
                    f"This chart shows complaint volume by complaint type for customer ID {customer_id}."
                    if rows
                    else f"No complaint records were found for customer ID {customer_id}."
                ),
            }

        rows = _run_rows(
            """
            SELECT
                complaint_category AS label,
                COUNT(*) AS value
            FROM complaints
            GROUP BY complaint_category
            ORDER BY value DESC
            LIMIT ?
            """,
            (limit,),
        )

        return {
            "title": "Top Complaint Categories",
            "metric": "Complaints",
            "chart_type": chart,
            "rows": _dict_rows(rows),
            "summary": "The chart highlights the complaint themes customers raise most often.",
        }

    if "campaign" in q:
        rows = _run_rows(
            """
            SELECT
                ca.campaign_name AS label,
                ROUND(100.0 * SUM(cr.converted_flag) / COUNT(cr.response_id), 2) AS value
            FROM campaigns ca
            JOIN customer_campaign_responses cr ON ca.campaign_id = cr.campaign_id
            GROUP BY ca.campaign_id, ca.campaign_name
            ORDER BY value DESC
            LIMIT ?
            """,
            (limit,),
        )

        return {
            "title": "Best Campaign Conversion Rates",
            "metric": "Conversion %",
            "chart_type": chart,
            "rows": _dict_rows(rows),
            "summary": "The chart compares campaigns by conversion rate.",
        }

    if "network" in q or "affected" in q:
        rows = _run_rows(
            """
            SELECT
                nt.city AS label,
                SUM(ne.affected_customers) AS value
            FROM network_events ne
            JOIN network_towers nt ON ne.tower_id = nt.tower_id
            GROUP BY nt.city
            ORDER BY value DESC
            LIMIT ?
            """,
            (limit,),
        )

        return {
            "title": "Network Impact by City",
            "metric": "Affected customers",
            "chart_type": chart,
            "rows": _dict_rows(rows),
            "summary": "The chart shows where network events affected the most customers.",
        }

    if "segment" in q and "revenue" in q:
        rows = _run_rows(
            """
            SELECT
                value_segment AS label,
                ROUND(AVG(total_revenue_6m_jod), 2) AS value
            FROM customer_value_segments
            GROUP BY value_segment
            ORDER BY value DESC
            LIMIT ?
            """,
            (limit,),
        )

        return {
            "title": "Average 6M Revenue by Segment",
            "metric": "JOD/customer",
            "chart_type": chart,
            "rows": _dict_rows(rows),
            "summary": "The chart compares value segments by average six-month revenue.",
        }

    if "city" in q and "customer" in q:
        rows = _run_rows(
            """
            SELECT
                city AS label,
                COUNT(*) AS value
            FROM customers
            GROUP BY city
            ORDER BY value DESC
            LIMIT ?
            """,
            (limit,),
        )

        return {
            "title": "Customers by City",
            "metric": "Customers",
            "chart_type": chart,
            "rows": _dict_rows(rows),
            "summary": "The chart shows the largest customer concentrations by city.",
        }

    try:
        return _run_dynamic_chart_query(question, chart, limit)
    except Exception as exc:
        return {
            "title": "Chart unavailable",
            "metric": "No data",
            "chart_type": chart,
            "rows": [],
            "summary": f"I could not create this chart from the available database fields. Details: {type(exc).__name__}: {exc}",
        }


def create_sql_agent():
    load_openai_key()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH.as_posix()}")
    model = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    sql_tools = toolkit.get_tools()

    system_prompt = f"""
You are a professional telecom business intelligence SQL agent for Zain Jordan.

You are connected to a SQLite telecom Customer 360 database.

Your job:
- Answer business questions using SQL.
- Inspect tables and schemas before writing SQL.
- Generate syntactically correct SQLite queries.
- Double-check SQL queries before execution.
- Execute queries only after checking them.
- Explain results in clear business language.

Important safety rules:
- Only use SELECT queries.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, REPLACE, ATTACH, DETACH, PRAGMA, VACUUM, or REINDEX.
- Do not modify the database.
- Do not query all columns from a table unless absolutely necessary.
- Unless the user requests a specific number, limit detailed query results to at most {TOP_K} rows.
- Use relevant columns only.
- If the question is ambiguous, explain your assumption.
- If a table or column does not exist, inspect schema and correct the query.
- Do not guess facts that are not in the database.

Very important query quality rules:
- Avoid joining multiple one-to-many tables directly.
- A customer can have multiple accounts, subscriptions, invoices, complaints, monthly summaries, satisfaction surveys, and interactions.
- If you join several one-to-many tables directly, you may multiply rows and return misleading results.
- For customer profile questions, prefer separate focused queries or CTEs that aggregate child tables first.
- For latest monthly data, use ORDER BY summary_month DESC LIMIT 1.
- For latest satisfaction data, use ORDER BY survey_date DESC LIMIT 1.
- For accounts or subscriptions, aggregate with COUNT and GROUP_CONCAT if returning one row per customer.
- Use GROUP BY when calculating counts, totals, or rates.
- Use COUNT(DISTINCT ...) when duplicate joins may inflate counts.

Useful telecom context:
- Churn analysis usually involves customers and customer_churn_scores.
- Revenue analysis usually involves customer_value_segments, invoices, payments, transactions, or customer_monthly_summary.
- Complaint analysis usually involves complaints and support_interactions.
- Campaign analysis usually involves campaigns and customer_campaign_responses.
- Network analysis usually involves network_towers and network_events.
- Customer profile analysis usually involves customers, accounts, subscriptions, plans, latest monthly summary, latest satisfaction, and churn scores.

After querying, provide:
1. Direct Answer
2. Key Numbers
3. Business Interpretation
4. Recommended Next Action, when useful
"""

    return create_agent(model=model, tools=sql_tools, system_prompt=system_prompt)


_SQL_AGENT = None


def plan_sql_for_question(question):
    q = question.lower()

    if any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in NON_DATABASE_PATTERNS):
        return ""

    customer_id = _extract_customer_id(question)

    if customer_id is not None and (
        "profile" in q
        or "full" in q
        or "details" in q
        or "plan" in q
        or "summary" in q
        or "customer" in q
        or "show me" in q
    ):
        return _customer_profile_flat_sql(customer_id)

    load_openai_key()

    if not os.environ.get("OPENAI_API_KEY"):
        return ""

    model = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)

    prompt = f"""
You are a SQLite SQL planner for a telecom Customer 360 database.

Database schema:
{_schema_summary()}

Question:
{question}

Return only one JSON object:
{{"sql": "SELECT ..."}}

Rules:
- SQL must be read-only SELECT only.
- Use the same business intent as the question.
- Use joins when needed.
- Use relevant columns only.
- Limit detailed row results to 25 rows unless the query is grouped.
- Avoid joining multiple one-to-many tables directly.
- If the question needs multiple one-to-many tables, aggregate them first using CTEs.
- Use COUNT(DISTINCT ...) when joins can multiply records.
- For one row per customer, aggregate accounts/subscriptions using GROUP_CONCAT or COUNT.
- For latest monthly summary, use ORDER BY summary_month DESC LIMIT 1.
- For latest satisfaction, use ORDER BY survey_date DESC LIMIT 1.
- If the needed data is not in the schema, return {{"sql": ""}}.
"""

    result = model.invoke([{"role": "user", "content": prompt}])
    content = result.content if hasattr(result, "content") else str(result)

    spec = _extract_json_object(content)
    sql = str(spec.get("sql", "")).strip()

    return validate_read_only_sql(sql) if sql else ""


def ask_sql_agent_payload(question):
    direct_payload = answer_direct_query(question)

    if direct_payload:
        return direct_payload

    global _SQL_AGENT

    if _SQL_AGENT is None:
        _SQL_AGENT = create_sql_agent()

    result = _SQL_AGENT.invoke({"messages": [{"role": "user", "content": question}]})
    answer = extract_final_text(result)

    try:
        sql = plan_sql_for_question(question)
    except Exception:
        sql = ""

    return {
        "answer": answer,
        "sql": sql,
    }


def ask_sql_agent(question):
    return ask_sql_agent_payload(question)["answer"]
