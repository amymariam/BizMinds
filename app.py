import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

if not CLAUDE_API_KEY:
    st.error("CLAUDE_API_KEY not found in environment variables! Please add it to your .env file.")
    st.stop()

CSV_PATH = "Structured data/cleaned_sales_products.csv"
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    st.error(f"CSV file not found at path: {CSV_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

st.title("💼 BizMinds — Business Insights AI")
st.write("Ask any question about your sales/product data and get instant business insights powered by Claude + LlamaIndex.")

query = st.text_input("🔍 Enter your question about the data:")

# LlamaIndex setup
try:
    from llama_index.experimental.query_engine import PandasQueryEngine
    from llama_index.llms.anthropic import Anthropic as LlamaClaude

    use_llamaindex = True
except Exception as e:
    st.warning(f"LlamaIndex unavailable, falling back to direct Claude API.\nError: {e}")
    use_llamaindex = False

def analyze_with_llamaindex(df, query):
    #LlamaIndex PandasQueryEngine with Claude
    llm = LlamaClaude(api_key=CLAUDE_API_KEY, model="claude-3-opus-20240229")
    query_engine = PandasQueryEngine(df=df, llm=llm, verbose=False)
    response = query_engine.query(query)
    return str(response)

def analyze_with_claude_api(df, query):
    #Direct Claude API as fallback
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
    client = Anthropic(api_key=CLAUDE_API_KEY)

    data_preview = df.head(300).to_string() 
    prompt = f"""
You are an expert business analyst. Analyze the following company sales/product data:
{data_preview}

Now answer this query clearly and analytically:
{query}
"""

    response = client.completions.create(
        model="claude-3-opus-20240229",
        prompt=HUMAN_PROMPT + prompt + AI_PROMPT,
        max_tokens_to_sample=500
    )
    return response.completion

if query:
    with st.spinner("🤔 Analyzing your data..."):
        try:
            if use_llamaindex:
                answer = analyze_with_llamaindex(df, query)
            else:
                answer = analyze_with_claude_api(df, query)
            st.success("✅ Analysis Complete")
            st.subheader("📊 AI Insights:")
            st.write(answer)
        except Exception as e:
            st.error(f"Error generating response: {e}")
