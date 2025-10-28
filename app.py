# app.py
import os
import pandas as pd
import streamlit as st
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv


load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    st.error("CLAUDE_API_KEY not found in environment variables!")
    st.stop()

client = Anthropic(api_key=CLAUDE_API_KEY)


CSV_PATH = "Structured data/cleaned_sales_products.csv"
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    st.error(f"CSV file not found at path: {CSV_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()


st.title("Business Insights AI")
st.write("Ask questions about your sales/product data and get actionable insights.")

query = st.text_input("Enter your business query:")

def ask_claude(query_text, df_data):
    
    data_str = df_data.head(500).to_string()
    prompt = f"""
You are an expert business analyst. Analyze the following company sales/product data:
{data_str}

Provide insights, suggest strategies, or answer this query:
{query_text}
"""
    response = client.completions.create(
        model="claude-3-opus-20240229",
        prompt=HUMAN_PROMPT + prompt + AI_PROMPT,
        max_tokens_to_sample=500
    )
    return response["completion"]

if query:
    with st.spinner("Analyzing your data..."):
        try:
            answer = ask_claude(query, df)
            st.subheader("AI Response:")
            st.write(answer)
        except Exception as e:
            st.error(f"Error generating response: {e}")
