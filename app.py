import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from google import genai

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Maryland Pedestrian Injury Chatbot", page_icon="🚶", layout="wide")
st.title("🚶 Maryland Pedestrian Injury Chatbot")
st.caption("Gemini + Pandas + Streamlit")

# ---------------------------
# API key input
# ---------------------------

# Try Streamlit secrets first, then fall back to user input
api_key = st.secrets.get("GEMINI_API_KEY", None)

if not api_key:
    api_key = st.text_input("Enter your Gemini API key", type="password")

if not api_key:
    st.info("Enter your Gemini API key to start.")
    st.stop()

client = genai.Client(api_key=api_key)
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
    df = df.dropna(subset=["Year", "Value"])

    return df

df = load_data()
jurisdictions = sorted(df["Jurisdiction"].dropna().unique().tolist())

# ---------------------------
# Helpers
# ---------------------------
def clean_json_text(text):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return text

def parse_question_with_gemini(question, jurisdictions):
    prompt = f"""
You are helping a pedestrian injury data chatbot.

Available jurisdictions:
{jurisdictions}

Return ONLY valid JSON in this exact format:
{{
  "intent": "rate" | "highest" | "trend" | "compare" | "unknown",
  "year": null or integer,
  "jurisdiction": null or string,
  "jurisdiction_2": null or string
}}

Rules:
- Use "rate" for questions asking for one place in one year.
- Use "highest" for questions asking which place had the highest rate in a year.
- Use "trend" for questions asking about change over time, plotting, graphing, or visualizing a jurisdiction across years.
- Use "compare" for questions comparing two jurisdictions in one year.
- If possible, match jurisdiction names exactly from the list provided.
- Return JSON only.

User question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    text = clean_json_text(response.text)

    try:
        return json.loads(text)
    except Exception:
        return {
            "intent": "unknown",
            "year": None,
            "jurisdiction": None,
            "jurisdiction_2": None
        }

def get_rate(df, jurisdiction, year):
    data = df[
        (df["Jurisdiction"].str.lower() == jurisdiction.lower()) &
        (df["Year"] == year)
    ]

    if data.empty:
        return f"No data found for {jurisdiction} in {year}."

    avg_value = data["Value"].mean()
    return f"The pedestrian injury rate in {jurisdiction} in {year} was {avg_value:.2f}."

def highest_rate(df, year):
    data = df[df["Year"] == year]
    if data.empty:
        return f"No data found for {year}."

    row = data.sort_values("Value", ascending=False).iloc[0]
    race_val = row["Race_Ethnicity"] if "Race_Ethnicity" in row.index else "All"
    return (
        f"In {year}, the highest pedestrian injury rate was in "
        f"{row['Jurisdiction']} ({race_val}), with a rate of {row['Value']:.2f}."
    )

def jurisdiction_trend(df, jurisdiction):
    data = df[df["Jurisdiction"].str.lower() == jurisdiction.lower()]
    if data.empty:
        return f"No data found for {jurisdiction}."

    summary = data.groupby("Year")["Value"].mean().reset_index().sort_values("Year")
    lines = [f"{int(row['Year'])}: {row['Value']:.2f}" for _, row in summary.iterrows()]
    return f"Trend for {jurisdiction}:\n" + "\n".join(lines)

def compare_jurisdictions(df, jurisdiction1, jurisdiction2, year):
    data1 = df[
        (df["Jurisdiction"].str.lower() == jurisdiction1.lower()) &
        (df["Year"] == year)
    ]
    data2 = df[
        (df["Jurisdiction"].str.lower() == jurisdiction2.lower()) &
        (df["Year"] == year)
    ]

    if data1.empty or data2.empty:
        return f"Missing data for one or both jurisdictions in {year}."

    val1 = data1["Value"].mean()
    val2 = data2["Value"].mean()

    return (
        f"In {year}, {jurisdiction1} had a pedestrian injury rate of {val1:.2f}, "
        f"while {jurisdiction2} had a rate of {val2:.2f}."
    )

def make_trend_figure(df, jurisdiction):
    data = df[df["Jurisdiction"].str.lower() == jurisdiction.lower()].copy()
    if data.empty:
        return None

    summary = (
        data.groupby("Year", as_index=False)["Value"]
        .mean()
        .sort_values("Year")
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(summary["Year"], summary["Value"], marker="o")
    ax.set_title(f"Pedestrian Injury Rate Trend: {jurisdiction}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Pedestrian Injury Rate")
    ax.grid(True)
    return fig

def make_compare_figure(df, jurisdiction1, jurisdiction2, year):
    data1 = df[
        (df["Jurisdiction"].str.lower() == jurisdiction1.lower()) &
        (df["Year"] == year)
    ]
    data2 = df[
        (df["Jurisdiction"].str.lower() == jurisdiction2.lower()) &
        (df["Year"] == year)
    ]

    if data1.empty or data2.empty:
        return None

    val1 = data1["Value"].mean()
    val2 = data2["Value"].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([jurisdiction1, jurisdiction2], [val1, val2])
    ax.set_title(f"Comparison in {year}")
    ax.set_ylabel("Pedestrian Injury Rate")
    return fig

def answer_question(question):
    parsed = parse_question_with_gemini(question, jurisdictions)

    intent = parsed.get("intent")
    year = parsed.get("year")
    jurisdiction = parsed.get("jurisdiction")
    jurisdiction_2 = parsed.get("jurisdiction_2")

    if intent == "rate":
        if year is None or not jurisdiction:
            return {"text": "I understood this as a rate question, but I need a valid year and jurisdiction."}
        return {"text": get_rate(df, jurisdiction, year)}

    if intent == "highest":
        if year is None:
            return {"text": "I understood this as a highest-rate question, but I need a year."}
        return {"text": highest_rate(df, year)}

    if intent == "trend":
        if not jurisdiction:
            return {"text": "I understood this as a trend question, but I need a valid jurisdiction."}
        fig = make_trend_figure(df, jurisdiction)
        return {"text": jurisdiction_trend(df, jurisdiction), "figure": fig}

    if intent == "compare":
        if year is None or not jurisdiction or not jurisdiction_2:
            return {"text": "I understood this as a comparison question, but I need two jurisdictions and a year."}
        fig = make_compare_figure(df, jurisdiction, jurisdiction_2, year)
        return {"text": compare_jurisdictions(df, jurisdiction, jurisdiction_2, year), "figure": fig}

    return {"text": "I could not understand the question. Try asking about a rate, highest value, trend, or comparison."}

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.subheader("Dataset")
    st.write(f"Rows: {len(df):,}")
    st.write(f"Jurisdictions: {df['Jurisdiction'].nunique():,}")
    st.write(f"Years: {int(df['Year'].min())}–{int(df['Year'].max())}")
    st.write("Example questions:")
    st.markdown("""
- What was the pedestrian injury rate in Baltimore City in 2022?
- Which jurisdiction had the highest rate in 2021?
- Show trend for Baltimore City
- Compare Maryland and Baltimore City in 2020
""")

# ---------------------------
# Chat history
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me about pedestrian injury rates, trends, or comparisons in Maryland."
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

    result = answer_question(user_prompt)

    assistant_message = {"role": "assistant", "content": result["text"]}

    with st.chat_message("assistant"):
        st.write(result["text"])
        if "figure" in result and result["figure"] is not None:
            figure_key = f"fig_{len(st.session_state.messages)}"
            st.session_state[figure_key] = result["figure"]
            assistant_message["figure_key"] = figure_key
            st.pyplot(result["figure"])

    st.session_state.messages.append(assistant_message)