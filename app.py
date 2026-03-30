import re
import boto3
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

# ---------------------------
# AWS config
# ---------------------------
AWS_REGION = st.secrets.get("AWS_REGION", "us-east-2")
BEDROCK_MODEL_ID = st.secrets.get("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")
AWS_ACCESS_KEY_ID = st.secrets.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    st.error("AWS secrets are missing in Streamlit app settings.")
    st.stop()

try:
    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
except Exception as e:
    st.error(f"Could not create Bedrock client: {e}")
    st.stop()

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="Maryland Pedestrian Injury Chatbot",
    # page_icon="panda.jpg",
    layout="wide"
)

col1, col2 = st.columns([1, 10])

with col1:
    pass
with col2:
    st.title("Maryland Pedestrian Injury Chatbot")
   

# ---------------------------
# Load and clean data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("SHIP_Pedestrian_Injury_Rate_on_Public_Roads_2009-2022_20260319.csv")

    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("/", "_", regex=False)
    df.columns = df.columns.str.replace(" ", "_", regex=False)

    if "Race__ethnicity" in df.columns:
        df = df.rename(columns={"Race__ethnicity": "Race_Ethnicity"})
    elif "Race_ethnicity" in df.columns:
        df = df.rename(columns={"Race_ethnicity": "Race_Ethnicity"})

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Jurisdiction"] = df["Jurisdiction"].astype(str).str.strip()

    df = df.dropna(subset=["Year", "Value", "Jurisdiction"])
    return df

df = load_data()
jurisdictions = sorted(df["Jurisdiction"].dropna().unique().tolist())

# ---------------------------
# Query helpers
# ---------------------------
def extract_year_range(user_query):
    years = re.findall(r"\b(20\d{2}|19\d{2})\b", user_query)
    years = [int(y) for y in years]

    if len(years) >= 2:
        return min(years), max(years)
    elif len(years) == 1:
        return years[0], years[0]
    return None, None

def extract_top_n(user_query, default=3):
    match = re.search(r"top\s+(\d+)", user_query.lower())
    if match:
        return int(match.group(1))
    return default

def is_lowest_trend_request(user_prompt: str) -> bool:
    q = user_prompt.lower()
    return (
        "trend" in q and "lowest" in q
    ) or (
        "show" in q and "lowest pedestrian injury rate" in q
    )

def is_highest_trend_request(user_prompt: str) -> bool:
    q = user_prompt.lower()
    return (
        "trend" in q and "highest" in q
    ) or (
        "show" in q and "highest pedestrian injury rate" in q
    )

def is_top_bottom_request(user_prompt: str) -> bool:
    q = user_prompt.lower()

    if is_lowest_trend_request(q) or is_highest_trend_request(q):
        return False

    has_top_bottom_words = any(k in q for k in ["top", "bottom", "lowest", "highest"])
    has_range = len(re.findall(r"\b(20\d{2}|19\d{2})\b", q)) >= 1

    return has_top_bottom_words and has_range

# ---------------------------
# Data functions
# ---------------------------
def find_best_jurisdiction_match(name: str):
    if not name:
        return None

    name_clean = str(name).strip().lower()

    exact_matches = [j for j in jurisdictions if j.lower() == name_clean]
    if exact_matches:
        return exact_matches[0]

    contains_matches = [j for j in jurisdictions if name_clean in j.lower()]
    if contains_matches:
        return contains_matches[0]

    reverse_contains = [j for j in jurisdictions if j.lower() in name_clean]
    if reverse_contains:
        return reverse_contains[0]

    return None

def get_rate(df_in, jurisdiction, year):
    matched = find_best_jurisdiction_match(jurisdiction)
    if matched is None:
        return f"I could not find a matching jurisdiction for '{jurisdiction}'."

    data = df_in[
        (df_in["Jurisdiction"].str.lower() == matched.lower()) &
        (df_in["Year"] == year)
    ]

    if data.empty:
        return f"No data found for {matched} in {year}."

    avg_value = data["Value"].mean()
    return f"The pedestrian injury rate in {matched} in {year} was {avg_value:.2f}."

def highest_rate(df_in, year):
    data = df_in[df_in["Year"] == year]
    if data.empty:
        return f"No data found for {year}."

    row = data.sort_values("Value", ascending=False).iloc[0]
    race_val = row["Race_Ethnicity"] if "Race_Ethnicity" in row.index else "All"

    return (
        f"In {year}, the highest pedestrian injury rate was in "
        f"{row['Jurisdiction']} ({race_val}), with a rate of {row['Value']:.2f}."
    )

def jurisdiction_trend(df_in, jurisdiction):
    matched = find_best_jurisdiction_match(jurisdiction)
    if matched is None:
        return f"I could not find a matching jurisdiction for '{jurisdiction}'."

    data = df_in[df_in["Jurisdiction"].str.lower() == matched.lower()]
    if data.empty:
        return f"No data found for {matched}."

    summary = data.groupby("Year")["Value"].mean().reset_index().sort_values("Year")
    lines = [f"{int(row['Year'])}: {row['Value']:.2f}" for _, row in summary.iterrows()]
    return f"Trend for {matched}:\n" + "\n".join(lines)

def compare_jurisdictions(df_in, jurisdiction1, jurisdiction2, year):
    matched1 = find_best_jurisdiction_match(jurisdiction1)
    matched2 = find_best_jurisdiction_match(jurisdiction2)

    if matched1 is None:
        return f"I could not find a matching jurisdiction for '{jurisdiction1}'."
    if matched2 is None:
        return f"I could not find a matching jurisdiction for '{jurisdiction2}'."

    data1 = df_in[
        (df_in["Jurisdiction"].str.lower() == matched1.lower()) &
        (df_in["Year"] == year)
    ]
    data2 = df_in[
        (df_in["Jurisdiction"].str.lower() == matched2.lower()) &
        (df_in["Year"] == year)
    ]

    if data1.empty or data2.empty:
        return f"Missing data for one or both jurisdictions in {year}."

    val1 = data1["Value"].mean()
    val2 = data2["Value"].mean()

    return (
        f"In {year}, {matched1} had a pedestrian injury rate of {val1:.2f}, "
        f"while {matched2} had a rate of {val2:.2f}."
    )

def generate_data_summary(df_in, jurisdiction=None):
    data = df_in.copy()

    matched = None
    if jurisdiction:
        matched = find_best_jurisdiction_match(jurisdiction)
        if matched is None:
            return None
        data = data[data["Jurisdiction"].str.lower() == matched.lower()]

    if data.empty:
        return None

    summary = (
        data.groupby("Year", as_index=False)["Value"]
        .mean()
        .sort_values("Year")
    )

    if summary.empty:
        return None

    first_year = int(summary["Year"].iloc[0])
    last_year = int(summary["Year"].iloc[-1])
    first_value = float(summary["Value"].iloc[0])
    last_value = float(summary["Value"].iloc[-1])
    avg_value = float(summary["Value"].mean())
    max_value = float(summary["Value"].max())
    min_value = float(summary["Value"].min())

    if last_value > first_value:
        trend = "increasing"
    elif last_value < first_value:
        trend = "decreasing"
    else:
        trend = "stable"

    pct_change = None
    if first_value != 0:
        pct_change = ((last_value - first_value) / first_value) * 100

    max_row = summary.loc[summary["Value"].idxmax()]
    min_row = summary.loc[summary["Value"].idxmin()]

    return {
        "jurisdiction": matched if matched else "Maryland overall",
        "trend": trend,
        "first_year": first_year,
        "last_year": last_year,
        "first_value": round(first_value, 2),
        "last_value": round(last_value, 2),
        "average": round(avg_value, 2),
        "max_value": round(max_value, 2),
        "min_value": round(min_value, 2),
        "max_year": int(max_row["Year"]),
        "min_year": int(min_row["Year"]),
        "pct_change": round(pct_change, 2) if pct_change is not None else None
    }

def extract_counties_and_years(user_query, available_counties):
    query_lower = user_query.lower()
    found_counties = []

    for county in available_counties:
        county_clean = county.lower().replace(" county", "").strip()
        if county_clean in query_lower:
            found_counties.append(county)

    years = re.findall(r"\b(20\d{2}|19\d{2})\b", user_query)
    years = [int(y) for y in years]

    if len(years) >= 2:
        start_year, end_year = min(years), max(years)
    elif len(years) == 1:
        start_year = end_year = years[0]
    else:
        start_year, end_year = None, None

    return found_counties, start_year, end_year

def multi_jurisdiction_trend_text(df_in, counties, start_year, end_year):
    matched_counties = []
    for c in counties:
        matched = find_best_jurisdiction_match(c)
        if matched:
            matched_counties.append(matched)

    matched_counties = list(dict.fromkeys(matched_counties))

    if len(matched_counties) < 2:
        return "I could not identify at least two valid jurisdictions to compare."

    filtered = df_in[
        (df_in["Jurisdiction"].isin(matched_counties)) &
        (df_in["Year"] >= start_year) &
        (df_in["Year"] <= end_year)
    ].copy()

    if filtered.empty:
        return "No data found for the selected jurisdictions and year range."

    summary = (
        filtered.groupby(["Jurisdiction", "Year"], as_index=False)["Value"]
        .mean()
        .sort_values(["Jurisdiction", "Year"])
    )

    lines = [f"Here are the pedestrian injury trends from {start_year} to {end_year}:"]
    for j in matched_counties:
        lines.append(f"\n{j}:")
        sub = summary[summary["Jurisdiction"] == j]
        for _, row in sub.iterrows():
            lines.append(f"- {int(row['Year'])}: {row['Value']:.2f}")

    return "\n".join(lines)

def top_bottom_jurisdictions_by_year(df_in, start_year, end_year, top_n=3):
    data = df_in[
        (df_in["Year"] >= start_year) &
        (df_in["Year"] <= end_year)
    ].copy()

    if data.empty:
        return {"text": f"No data found from {start_year} to {end_year}."}

    yearly_avg = (
        data.groupby(["Year", "Jurisdiction"], as_index=False)["Value"]
        .mean()
    )

    lines = []

    for year in range(start_year, end_year + 1):
        year_data = yearly_avg[yearly_avg["Year"] == year].copy()

        if year_data.empty:
            lines.append(f"\n{year}: No data found.")
            continue

        highest = year_data.sort_values("Value", ascending=False).head(top_n)
        lowest = year_data.sort_values("Value", ascending=True).head(top_n)

        lines.append(f"\n{year}")
        lines.append(f"Top {top_n} highest:")
        for _, row in highest.iterrows():
            lines.append(f"- {row['Jurisdiction']}: {row['Value']:.2f}")

        lines.append(f"Top {top_n} lowest:")
        for _, row in lowest.iterrows():
            lines.append(f"- {row['Jurisdiction']}: {row['Value']:.2f}")

    return {"text": "\n".join(lines)}

# ---------------------------
# Plotting functions
# ---------------------------
def make_trend_figure(df_in, jurisdiction):
    matched = find_best_jurisdiction_match(jurisdiction)
    if matched is None:
        return None

    data = df_in[df_in["Jurisdiction"].str.lower() == matched.lower()].copy()
    if data.empty:
        return None

    summary = (
        data.groupby("Year", as_index=False)["Value"]
        .mean()
        .sort_values("Year")
    )
    summary["Year"] = summary["Year"].astype(int)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(summary["Year"], summary["Value"], marker="o", linewidth=2, markersize=5)
    ax.set_title(f"Trend: {matched}", fontsize=12, pad=10)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_xticks(summary["Year"].tolist())
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

def make_compare_figure(df_in, jurisdiction1, jurisdiction2, year):
    matched1 = find_best_jurisdiction_match(jurisdiction1)
    matched2 = find_best_jurisdiction_match(jurisdiction2)

    if matched1 is None or matched2 is None:
        return None

    data1 = df_in[
        (df_in["Jurisdiction"].str.lower() == matched1.lower()) &
        (df_in["Year"] == year)
    ]
    data2 = df_in[
        (df_in["Jurisdiction"].str.lower() == matched2.lower()) &
        (df_in["Year"] == year)
    ]

    if data1.empty or data2.empty:
        return None

    val1 = data1["Value"].mean()
    val2 = data2["Value"].mean()

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.bar([matched1, matched2], [val1, val2], width=0.55)
    ax.set_title(f"Comparison in {year}", fontsize=12, pad=10)
    ax.set_ylabel("Rate", fontsize=10)
    ax.tick_params(axis="x", labelrotation=15, labelsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

def plot_county_trend(df_in, counties, start_year=2015, end_year=2020,
                      county_col="Jurisdiction", year_col="Year", value_col="Value"):
    matched_counties = []
    for c in counties:
        matched = find_best_jurisdiction_match(c)
        if matched:
            matched_counties.append(matched)

    matched_counties = list(dict.fromkeys(matched_counties))

    if len(matched_counties) < 2:
        return None, None

    data = df_in.copy()
    data[year_col] = pd.to_numeric(data[year_col], errors="coerce")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")

    filtered = data[
        (data[county_col].isin(matched_counties)) &
        (data[year_col] >= start_year) &
        (data[year_col] <= end_year)
    ].copy()

    if filtered.empty:
        return None, None

    trend = (
        filtered.groupby([year_col, county_col], as_index=False)[value_col]
        .mean()
        .sort_values([county_col, year_col])
    )
    trend[year_col] = trend[year_col].astype(int)

    pivot_df = trend.pivot(index=year_col, columns=county_col, values=value_col)
    pivot_df.index = pivot_df.index.astype(int)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for county in pivot_df.columns:
        ax.plot(pivot_df.index, pivot_df[county], marker="o", linewidth=2, label=county)

    ax.set_title(f"Trend Comparison ({start_year}–{end_year})", fontsize=12, pad=10)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Pedestrian Injury Rate", fontsize=10)
    ax.set_xticks(pivot_df.index.tolist())
    ax.legend(title="Jurisdiction")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return fig, pivot_df.reset_index()

def plot_lowest_by_year(df_in, start_year, end_year):
    data = df_in[
        (df_in["Year"] >= start_year) &
        (df_in["Year"] <= end_year)
    ].copy()

    if data.empty:
        return None, f"No data found from {start_year} to {end_year}."

    yearly_avg = (
        data.groupby(["Year", "Jurisdiction"], as_index=False)["Value"]
        .mean()
    )

    lowest_each_year = (
        yearly_avg.sort_values(["Year", "Value"], ascending=[True, True])
        .groupby("Year", as_index=False)
        .first()
    )
    lowest_each_year["Year"] = lowest_each_year["Year"].astype(int)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        lowest_each_year["Year"],
        lowest_each_year["Value"],
        marker="o",
        linewidth=2
    )

    for _, row in lowest_each_year.iterrows():
        ax.annotate(
            row["Jurisdiction"],
            (row["Year"], row["Value"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8
        )

    ax.set_title(f"Lowest Pedestrian Injury Rate by Year ({start_year}–{end_year})", fontsize=12, pad=10)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Lowest Rate", fontsize=10)
    ax.set_xticks(lowest_each_year["Year"].tolist())
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    text_lines = [f"Lowest jurisdiction in each year from {start_year} to {end_year}:"]
    for _, row in lowest_each_year.iterrows():
        text_lines.append(f"- {int(row['Year'])}: {row['Jurisdiction']} ({row['Value']:.2f})")

    return fig, "\n".join(text_lines)

def plot_highest_by_year(df_in, start_year, end_year):
    data = df_in[
        (df_in["Year"] >= start_year) &
        (df_in["Year"] <= end_year)
    ].copy()

    if data.empty:
        return None, f"No data found from {start_year} to {end_year}."

    yearly_avg = (
        data.groupby(["Year", "Jurisdiction"], as_index=False)["Value"]
        .mean()
    )

    highest_each_year = (
        yearly_avg.sort_values(["Year", "Value"], ascending=[True, False])
        .groupby("Year", as_index=False)
        .first()
    )
    highest_each_year["Year"] = highest_each_year["Year"].astype(int)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        highest_each_year["Year"],
        highest_each_year["Value"],
        marker="o",
        linewidth=2
    )

    for _, row in highest_each_year.iterrows():
        ax.annotate(
            row["Jurisdiction"],
            (row["Year"], row["Value"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8
        )

    ax.set_title(f"Highest Pedestrian Injury Rate by Year ({start_year}–{end_year})", fontsize=12, pad=10)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Highest Rate", fontsize=10)
    ax.set_xticks(highest_each_year["Year"].tolist())
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    text_lines = [f"Highest jurisdiction in each year from {start_year} to {end_year}:"]
    for _, row in highest_each_year.iterrows():
        text_lines.append(f"- {int(row['Year'])}: {row['Jurisdiction']} ({row['Value']:.2f})")

    return fig, "\n".join(text_lines)

# ---------------------------
# Bedrock tool config
# ---------------------------
SYSTEM_PROMPT = f"""
You are a Maryland pedestrian injury data assistant.

You help users answer questions about a dataset of pedestrian injury rates.

Available jurisdictions:
{', '.join(jurisdictions)}

You have access to tools for:
- one jurisdiction in one year
- highest jurisdiction in a year
- trend over time for one jurisdiction
- comparison between two jurisdictions in one year
- multi-jurisdiction trend comparison across a range of years
- statistical summary / analysis

Rules:
- Use tools whenever a user asks about data.
- Do not invent any numeric value.
- Use exact jurisdiction names when possible.
- If the user asks for explanation or analysis, first call analysis_summary.
- Keep answers concise, clear, and grounded in the data.
- If a user asks for a chart, trend, visualization, graph, plot, or compare visually, use the correct tool.
- If a user asks to compare two or more jurisdictions over multiple years or on the same plot, use compare_trend_multiple.
"""

def get_tool_config():
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": "get_rate",
                    "description": "Get the pedestrian injury rate for one jurisdiction in one year.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "jurisdiction": {"type": "string"},
                                "year": {"type": "integer"}
                            },
                            "required": ["jurisdiction", "year"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "highest_rate",
                    "description": "Find the jurisdiction with the highest pedestrian injury rate in a given year.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "year": {"type": "integer"}
                            },
                            "required": ["year"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "trend_summary",
                    "description": "Summarize the pedestrian injury trend over time for a jurisdiction.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "jurisdiction": {"type": "string"}
                            },
                            "required": ["jurisdiction"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "compare_jurisdictions",
                    "description": "Compare two jurisdictions in a given year.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "jurisdiction1": {"type": "string"},
                                "jurisdiction2": {"type": "string"},
                                "year": {"type": "integer"}
                            },
                            "required": ["jurisdiction1", "jurisdiction2", "year"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "compare_trend_multiple",
                    "description": "Compare trend lines for multiple jurisdictions across a range of years on the same plot.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "jurisdictions": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "start_year": {"type": "integer"},
                                "end_year": {"type": "integer"}
                            },
                            "required": ["jurisdictions", "start_year", "end_year"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "analysis_summary",
                    "description": "Generate a structured summary of the data for one jurisdiction or Maryland overall.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "jurisdiction": {"type": "string"}
                            }
                        }
                    }
                }
            }
        ]
    }

# ---------------------------
# Tool executor
# ---------------------------
def execute_tool(tool_name, tool_input):
    if tool_name == "get_rate":
        jurisdiction = tool_input["jurisdiction"]
        year = int(tool_input["year"])
        text = get_rate(df, jurisdiction, year)
        return {"text": text}

    elif tool_name == "highest_rate":
        year = int(tool_input["year"])
        text = highest_rate(df, year)
        return {"text": text}

    elif tool_name == "trend_summary":
        jurisdiction = tool_input["jurisdiction"]
        text = jurisdiction_trend(df, jurisdiction)
        matched = find_best_jurisdiction_match(jurisdiction)
        return {
            "text": text,
            "jurisdiction": matched or jurisdiction,
            "show_trend_chart": True
        }

    elif tool_name == "compare_jurisdictions":
        j1 = tool_input["jurisdiction1"]
        j2 = tool_input["jurisdiction2"]
        year = int(tool_input["year"])
        text = compare_jurisdictions(df, j1, j2, year)

        matched1 = find_best_jurisdiction_match(j1) or j1
        matched2 = find_best_jurisdiction_match(j2) or j2

        return {
            "text": text,
            "jurisdiction1": matched1,
            "jurisdiction2": matched2,
            "year": year,
            "show_compare_chart": True
        }

    elif tool_name == "compare_trend_multiple":
        req_jurisdictions = tool_input["jurisdictions"]
        start_year = int(tool_input["start_year"])
        end_year = int(tool_input["end_year"])

        text = multi_jurisdiction_trend_text(df, req_jurisdictions, start_year, end_year)

        matched_list = []
        for j in req_jurisdictions:
            matched = find_best_jurisdiction_match(j)
            if matched:
                matched_list.append(matched)

        matched_list = list(dict.fromkeys(matched_list))

        return {
            "text": text,
            "jurisdictions": matched_list,
            "start_year": start_year,
            "end_year": end_year,
            "show_multi_trend_chart": True
        }

    elif tool_name == "analysis_summary":
        jurisdiction = tool_input.get("jurisdiction")
        summary = generate_data_summary(df, jurisdiction)
        if summary is None:
            return {"text": "I could not generate a structured summary from the data."}
        return {"summary": summary}

    return {"error": f"Unknown tool: {tool_name}"}

# ---------------------------
# Bedrock conversation
# ---------------------------
def extract_text_from_content_blocks(content_blocks):
    parts = []
    for block in content_blocks:
        if "text" in block:
            parts.append(block["text"])
    return "\n".join(parts).strip()

def ask_bedrock_with_tools(user_prompt):
    messages = [
        {
            "role": "user",
            "content": [{"text": user_prompt}]
        }
    ]

    pending_chart = None
    loops = 0
    max_loops = 6

    try:
        response = bedrock.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig=get_tool_config()
        )
    except (BotoCoreError, ClientError) as e:
        return {"text": f"Bedrock request failed: {e}", "chart": None}
    except Exception as e:
        return {"text": f"Unexpected Bedrock error: {e}", "chart": None}

    while loops < max_loops:
        loops += 1

        output_message = response["output"]["message"]
        stop_reason = response.get("stopReason", "")
        messages.append(output_message)

        if stop_reason == "end_turn":
            final_text = extract_text_from_content_blocks(output_message["content"])
            if not final_text:
                final_text = "I could not generate a final answer."
            return {"text": final_text, "chart": pending_chart}

        if stop_reason == "tool_use":
            tool_result_content = []

            for block in output_message["content"]:
                if "toolUse" not in block:
                    continue

                tool_use = block["toolUse"]
                tool_name = tool_use["name"]
                tool_input = tool_use["input"]
                tool_use_id = tool_use["toolUseId"]

                result = execute_tool(tool_name, tool_input)

                if result.get("show_trend_chart"):
                    pending_chart = {
                        "type": "trend",
                        "jurisdiction": result["jurisdiction"]
                    }

                if result.get("show_compare_chart"):
                    pending_chart = {
                        "type": "compare",
                        "jurisdiction1": result["jurisdiction1"],
                        "jurisdiction2": result["jurisdiction2"],
                        "year": result["year"]
                    }

                if result.get("show_multi_trend_chart"):
                    pending_chart = {
                        "type": "multi_trend",
                        "jurisdictions": result["jurisdictions"],
                        "start_year": result["start_year"],
                        "end_year": result["end_year"]
                    }

                tool_result_content.append({
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"json": result}]
                    }
                })

            if not tool_result_content:
                return {
                    "text": "The model requested tool use, but no valid tool call was returned.",
                    "chart": None
                }

            messages.append({
                "role": "user",
                "content": tool_result_content
            })

            try:
                response = bedrock.converse(
                    modelId=BEDROCK_MODEL_ID,
                    system=[{"text": SYSTEM_PROMPT}],
                    messages=messages,
                    toolConfig=get_tool_config()
                )
            except (BotoCoreError, ClientError) as e:
                return {"text": f"Bedrock follow-up request failed: {e}", "chart": None}
            except Exception as e:
                return {"text": f"Unexpected Bedrock follow-up error: {e}", "chart": None}

            continue

        return {"text": "I could not complete the request.", "chart": None}

    return {"text": "The Bedrock tool loop reached its limit.", "chart": None}

# ---------------------------
# Main app logic
# ---------------------------
def answer_question(question):
    result = ask_bedrock_with_tools(question)

    fig = None
    chart = result.get("chart")

    if chart:
        if chart["type"] == "trend":
            fig = make_trend_figure(df, chart["jurisdiction"])

        elif chart["type"] == "compare":
            fig = make_compare_figure(
                df,
                chart["jurisdiction1"],
                chart["jurisdiction2"],
                chart["year"]
            )

        elif chart["type"] == "multi_trend":
            fig, _ = plot_county_trend(
                df,
                chart["jurisdictions"],
                start_year=chart["start_year"],
                end_year=chart["end_year"],
                county_col="Jurisdiction",
                year_col="Year",
                value_col="Value"
            )

    return {"text": result["text"], "figure": fig}

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.subheader("Dataset")
    st.write(f"Rows: {len(df):,}")
    st.write(f"Jurisdictions: {df['Jurisdiction'].nunique():,}")
    st.write(f"Years: {int(df['Year'].min())}–{int(df['Year'].max())}")
    st.write(f"AWS Region: {AWS_REGION}")
    st.write("Example questions:")
    st.markdown("""
    - What was the pedestrian injury rate in Baltimore City in 2022?
    - Which jurisdiction had the highest rate in 2021?
    - Show trend for Baltimore City
    - Compare Maryland and Baltimore City in 2020
    - Show Washington and Baltimore County together from 2015 to 2020
    - Show top 5 counties with highest injury rate and top 5 with lowest injury rate from 2017 to 2020
    - Show trend for the lowest pedestrian injury rate for each year from 2017 to 2020
    - Show trend for the highest pedestrian injury rate for each year from 2017 to 2020
    """)

# ---------------------------
# Chat history
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me about pedestrian injury rates, trends, comparisons, or analysis in Maryland."
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "figure_key" in message and message["figure_key"] in st.session_state:
            st.pyplot(st.session_state[message["figure_key"]])

# ---------------------------
# Chat input
# ---------------------------
user_prompt = st.chat_input("Ask a question about the dataset...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.write(user_prompt)

    start_year, end_year = extract_year_range(user_prompt)

    if is_lowest_trend_request(user_prompt):
        if start_year is None or end_year is None:
            result = {
                "text": "Please specify a year range like 2017 to 2020.",
                "figure": None
            }
        else:
            fig, text = plot_lowest_by_year(df, start_year, end_year)
            result = {
                "text": text,
                "figure": fig
            }

    elif is_highest_trend_request(user_prompt):
        if start_year is None or end_year is None:
            result = {
                "text": "Please specify a year range like 2017 to 2020.",
                "figure": None
            }
        else:
            fig, text = plot_highest_by_year(df, start_year, end_year)
            result = {
                "text": text,
                "figure": fig
            }

    elif is_top_bottom_request(user_prompt):
        top_n = extract_top_n(user_prompt, default=3)

        if start_year is None or end_year is None:
            result = {
                "text": "Please specify a year range like 2017 to 2020.",
                "figure": None
            }
        else:
            top_bottom_result = top_bottom_jurisdictions_by_year(df, start_year, end_year, top_n)
            result = {
                "text": top_bottom_result["text"],
                "figure": None
            }

    else:
        result = answer_question(user_prompt)

    assistant_message = {
        "role": "assistant",
        "content": result["text"]
    }

    with st.chat_message("assistant"):
        st.write(result["text"])
        if result.get("figure") is not None:
            figure_key = f"fig_{len(st.session_state.messages)}"
            st.session_state[figure_key] = result["figure"]
            assistant_message["figure_key"] = figure_key
            st.pyplot(result["figure"])

    st.session_state.messages.append(assistant_message)
